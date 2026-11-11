from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def _read_yaml(path: str) -> dict:
    with (ROOT / path).open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def test_compose_separates_api_worker_dispatcher_and_dependencies():
    compose = _read_yaml("app/docker-compose.yml")
    services = compose["services"]

    assert {
        "frontend",
        "uploads-init",
        "agent-api",
        "ingestion-worker",
        "job-dispatcher",
    } <= services.keys()
    assert {"db", "redis", "ollama"} <= services.keys()
    assert services["frontend"]["build"]["context"] == "../frontend"
    assert services["frontend"]["ports"] == ["127.0.0.1:5173:5173"]
    assert services["frontend"]["depends_on"]["agent-api"][
        "condition"
    ] == "service_healthy"
    assert services["uploads-init"]["volumes"] == ["uploads:/data/uploads"]
    assert services["agent-api"]["depends_on"]["uploads-init"][
        "condition"
    ] == "service_completed_successfully"
    assert services["ingestion-worker"]["depends_on"]["uploads-init"][
        "condition"
    ] == "service_completed_successfully"
    assert services["ingestion-worker"]["command"][-1] == (
        "--queues=document-ingestion"
    )
    assert "beat" in services["job-dispatcher"]["command"]
    assert services["ingestion-worker"]["depends_on"]["agent-api"][
        "condition"
    ] == "service_healthy"
    assert services["agent-api"]["healthcheck"]["test"]


def test_compose_publishes_only_loopback_ports():
    compose = _read_yaml("app/docker-compose.yml")

    for service in compose["services"].values():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:")


def test_frontend_proxies_api_requests_through_vite():
    frontend = ROOT / "frontend"
    compose = _read_yaml("app/docker-compose.yml")
    package = (frontend / "package.json").read_text(encoding="utf-8")
    vite_config = (frontend / "vite.config.ts").read_text(encoding="utf-8")
    app_component = (frontend / "src" / "App.svelte").read_text(
        encoding="utf-8"
    )

    assert '"svelte"' in package
    assert '"/api"' in vite_config
    assert "API_PROXY_TARGET" in vite_config
    assert (
        compose["services"]["frontend"]["environment"]["API_PROXY_TARGET"]
        == "http://agent-api:8000"
    )
    assert '"/api/documents/upload"' in app_component
    assert '"/api/chat"' in app_component


def test_frontend_production_image_builds_and_serves_the_spa():
    frontend = ROOT / "frontend"
    dockerfile = (frontend / "Dockerfile").read_text(encoding="utf-8")
    nginx_template = (frontend / "nginx.conf.template").read_text(
        encoding="utf-8"
    )
    compose = _read_yaml("app/docker-compose.yml")

    assert "FROM node:22-alpine AS development" in dockerfile
    assert "FROM development AS build" in dockerfile
    assert "RUN npm run build" in dockerfile
    assert "FROM nginx:alpine AS production" in dockerfile
    assert "COPY --from=build /app/dist /usr/share/nginx/html" in dockerfile
    assert (
        compose["services"]["frontend"]["build"]["target"] == "development"
    )
    assert "proxy_pass ${API_PROXY_TARGET}/;" in nginx_template
    assert "try_files $uri $uri/ /index.html;" in nginx_template


def test_e2e_compose_services_are_loopback_bound():
    compose = _read_yaml("tests/e2e/compose.yml")

    assert {"postgres", "redis"} == compose["services"].keys()
    for service in compose["services"].values():
        assert all(
            str(port).startswith("127.0.0.1:")
            for port in service["ports"]
        )
        assert service["healthcheck"]


def test_render_blueprint_has_api_worker_dispatcher_and_redis():
    blueprint = _read_yaml("render.yaml")
    services = {service["name"]: service for service in blueprint["services"]}

    assert {
        "prod-rag-api",
        "prod-rag-ingestion",
        "prod-rag-job-dispatcher",
        "prod-rag-redis",
    } <= services.keys()
    assert services["prod-rag-api"]["healthCheckPath"] == "/health"
    assert "worker" in services["prod-rag-ingestion"]["type"]
    assert "worker" in services["prod-rag-job-dispatcher"]["type"]
    assert "beat" in services["prod-rag-job-dispatcher"]["dockerCommand"]
    for service_name in (
        "prod-rag-api",
        "prod-rag-ingestion",
        "prod-rag-job-dispatcher",
    ):
        env_keys = {
            entry["key"]
            for entry in services[service_name]["envVars"]
        }
        assert "TENANT_ID" in env_keys


def test_local_compose_uses_development_defaults_and_plain_http_ollama():
    compose = _read_yaml("app/docker-compose.yml")

    for service_name in ("agent-api", "ingestion-worker"):
        environment = compose["services"][service_name]["environment"]
        assert environment["APP_ENV"] == "development"
        assert environment["LLM_BASE_URL"] == "http://ollama:11434/v1"


def test_production_image_uses_pinned_uv_and_non_root_runtime():
    dockerfile = (ROOT / "app" / "Dockerfile").read_text(encoding="utf-8")

    assert "python:3.14-slim" in dockerfile
    assert "ghcr.io/astral-sh/uv:0.13.6" in dockerfile
    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "USER appuser" in dockerfile
    assert "alembic upgrade head" in dockerfile
