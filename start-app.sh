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

if ! printf '%s\n' "$model_list" | grep -Eq '^llama3\.2:1b([[:space:]]|$)'; then
    echo "Downloading required model llama3.2:1b (first launch may take a while)..."
    compose exec -T ollama ollama pull llama3.2:1b
else
    echo "Model llama3.2:1b is already installed."
fi

if ! printf '%s\n' "$model_list" | grep -Eq '^nomic-embed-text(:latest)?([[:space:]]|$)'; then
    echo "Downloading required model nomic-embed-text (first launch may take a while)..."
    compose exec -T ollama ollama pull nomic-embed-text
else
    echo "Model nomic-embed-text is already installed."
fi

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
