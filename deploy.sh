#!/usr/bin/env sh
set -eu

ACTION="${1:-up}"
SEED="${2:-}"
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env.docker"
ENV_TEMPLATE="$PROJECT_ROOT/.env.docker.example"

case "$ACTION" in
  up|down|restart|logs|status) ;;
  *) echo "Usage: ./deploy.sh [up|down|restart|logs|status] [--seed]" >&2; exit 2 ;;
esac

get_env() {
  awk -F= -v key="$1" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$ENV_FILE"
}

set_env() {
  key=$1
  value=$2
  temporary="$ENV_FILE.tmp"
  awk -v key="$key" -v value="$value" '
    BEGIN { found = 0 }
    index($0, key "=") == 1 { print key "=" value; found = 1; next }
    { print }
    END { if (!found) print key "=" value }
  ' "$ENV_FILE" > "$temporary"
  mv "$temporary" "$ENV_FILE"
}

new_secret() {
  od -An -N32 -tx1 /dev/urandom | tr -d ' \n'
}

initialize_env() {
  if [ ! -f "$ENV_FILE" ]; then
    cp "$ENV_TEMPLATE" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "Created .env.docker from the committed template."
  fi

  for name in MYSQL_APP_PASSWORD MYSQL_ROOT_PASSWORD PII_ENCRYPTION_KEY LANGFUSE_HASH_SALT; do
    if [ "$(get_env "$name")" = "GENERATE_ON_FIRST_RUN" ]; then
      set_env "$name" "$(new_secret)"
    fi
  done

  if [ "$ACTION" = "up" ] || [ "$ACTION" = "restart" ]; then
    api_key=$(get_env QWEN_API_KEY)
    if [ -z "$api_key" ] && [ -n "${QWEN_API_KEY:-}" ]; then
      api_key=$QWEN_API_KEY
      set_env QWEN_API_KEY "$api_key"
    fi
    if [ -z "$api_key" ] && [ -t 0 ]; then
      printf "Enter QWEN_API_KEY (input is hidden): "
      stty -echo
      IFS= read -r api_key
      stty echo
      printf '\n'
      [ -n "$api_key" ] && set_env QWEN_API_KEY "$api_key"
    fi
    if [ -z "$api_key" ]; then
      echo "QWEN_API_KEY is required. Set it in .env.docker or export it first." >&2
      exit 1
    fi
  fi
}

assert_docker() {
  command -v docker >/dev/null 2>&1 || {
    echo "Docker is not installed or is not available in PATH." >&2
    exit 1
  }
  docker info >/dev/null 2>&1 || {
    echo "Docker Engine is not running. Start Docker and retry." >&2
    exit 1
  }
  docker compose version >/dev/null 2>&1 || {
    echo "Docker Compose v2 is required." >&2
    exit 1
  }
}

compose() {
  docker compose --env-file "$ENV_FILE" --file "$COMPOSE_FILE" "$@"
}

initialize_env
assert_docker

case "$ACTION" in
  up)
    compose config --quiet
    compose up --detach --build --remove-orphans --wait --wait-timeout 240
    if [ "$SEED" = "--seed" ]; then
      compose exec --no-TTY backend python scripts/seed_demo_business_data.py --apply
    fi
    port=$(get_env HTTP_PORT)
    if [ "${port:-80}" = "80" ]; then
      echo "Smart CS Agent is ready: http://localhost"
    else
      echo "Smart CS Agent is ready: http://localhost:$port"
    fi
    ;;
  restart)
    compose up --detach --build --remove-orphans --wait --wait-timeout 240
    echo "Smart CS Agent has been rebuilt and restarted."
    ;;
  down)
    compose down --remove-orphans
    echo "Containers stopped. Persistent volumes were kept."
    ;;
  logs) compose logs --follow --tail 200 ;;
  status) compose ps ;;
esac

