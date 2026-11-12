#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

COMPOSE_FILE="$SCRIPT_DIR/app/docker-compose.yml"
FRONTEND_URL="http://localhost:5173"
API_HEALTH_URL="http://localhost:8000/health"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed or is not on PATH. Install Docker Desktop (macOS) or Docker Engine (Linux), then try again." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Start Docker Desktop or the Docker service, then try again." >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    if [ ! -f "$SCRIPT_DIR/.env.example" ]; then
        echo "Cannot create .env: .env.example was not found." >&2
        exit 1
    fi
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "Created .env from .env.example."
fi

compose() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

dotenv_value() {
    name=$1
    default_value=$2
    value=$(awk -v key="$name" '
        $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
            sub(/^[^=]*=/, "")
            gsub(/^[[:space:]]+|[[:space:]]+$/, "")
            if ((substr($0, 1, 1) == "\"" && substr($0, length($0), 1) == "\"") ||
                (substr($0, 1, 1) == "\047" && substr($0, length($0), 1) == "\047")) {
                $0 = substr($0, 2, length($0) - 2)
            }
            value = $0
        }
        END { if (value != "") print value }
    ' "$SCRIPT_DIR/.env")
    if [ -z "$value" ]; then
        value=$default_value
    fi
    printf '%s' "$value"
}

ensure_model() {
    model=$1
    if [ -z "$model" ]; then
        echo "Model names in .env must not be empty." >&2
        exit 1
    fi

    model_list=$(compose exec -T ollama ollama list)
    installed_models=$(printf '%s\n' "$model_list" | awk 'NR > 1 {print $1}')
    latest_model=$model
    case "$model" in
        *:*) ;;
        *) latest_model="${model}:latest" ;;
    esac

    if printf '%s\n' "$installed_models" | grep -F -x -e "$model" >/dev/null ||
        printf '%s\n' "$installed_models" | grep -F -x -e "$latest_model" >/dev/null; then
        echo "Model $model is already installed."
    else
        echo "Downloading required model $model (first launch may take a while)..."
        compose exec -T ollama ollama pull "$model"
    fi
}

echo "Building and starting the application..."
compose up --build --detach

echo "Waiting for Ollama..."
deadline=$(( $(date +%s) + 300 ))
while ! model_list=$(compose exec -T ollama ollama list 2>/dev/null); do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "Ollama did not become ready. Check the Ollama container logs." >&2
        exit 1
    fi
    sleep 5
done

primary_model=$(dotenv_value PRIMARY_MODEL llama3.2:1b)
fallback_model=$(dotenv_value FALLBACK_MODEL llama3.2:1b)
embedding_model=$(dotenv_value EMBEDDING_MODEL nomic-embed-text)
ensure_model "$primary_model"
if [ "$fallback_model" != "$primary_model" ]; then
    ensure_model "$fallback_model"
fi
ensure_model "$embedding_model"

wait_for_endpoint() {
    url=$1
    name=$2
    deadline=$(( $(date +%s) + 300 ))

    while ! curl --silent --show-error --fail --max-time 5 "$url" >/dev/null 2>&1; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "$name did not become ready at $url within 300 seconds." >&2
            exit 1
        fi
        sleep 3
    done
    echo "$name is ready."
}

wait_for_endpoint "$API_HEALTH_URL" "API"
wait_for_endpoint "$FRONTEND_URL" "Frontend"

echo "Opening $FRONTEND_URL"
case "$(uname -s)" in
    Darwin) open "$FRONTEND_URL" || echo "Open $FRONTEND_URL in your browser." ;;
    Linux)
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$FRONTEND_URL" >/dev/null 2>&1 || echo "Open $FRONTEND_URL in your browser."
        else
            echo "Open $FRONTEND_URL in your browser."
        fi
        ;;
    *) echo "Open $FRONTEND_URL in your browser." ;;
esac

echo "Application is running. Stop it with: docker compose -f app/docker-compose.yml down"
