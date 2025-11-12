set shell := ["bash", "-euo", "pipefail", "-c"]
set dotenv-load := true

# Configuration variables
compose_file := "src/services/simulation_service/core/docker-compose.yml"
# ==============================

# Default recipe - shows available commands
default:
    @just --list --unsorted
    @just check-runner

# Core installation recipes
[group('setup')]
[doc('Install basic tools like git and curl needed by other installations')]
prerequisites:
    @echo -e "\n==== Installing prerequisites (git, curl)... ====\n\n"
    sudo apt update
    sudo apt install -y git curl
    @echo -e "\n\033[0;32m==== Prerequisites installed successfully. ====\033[0m\n\n"

[group('setup')]
[doc('Install Docker Engine with latest best practices')]
install-docker: prerequisites
    @echo -e "\n==== Removing old Docker packages... ====\n\n"
    for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd.io runc; do
    sudo apt-get remove -y "$pkg" || true
    done

    @echo -e "\n==== Setting up Docker repository... ====\n\n"
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}}) stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    @echo -e "\n==== Installing Docker Engine... ====\n\n"
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    @echo -e "\n\033[0;32m==== Docker installed successfully. ====\033[0m\n\n"

[group('setup')]
[doc('Verify Docker installation works correctly')]
verify-docker:
    @echo -e "\n==== Verifying Docker installation... ====\n\n"
    sudo docker run --rm hello-world
    @echo -e "\n\033[0;32m==== Docker verification complete. ====\033[0m\n\n"

[group('setup')]
[doc('Install xterm for external terminal logging')]
install-xterm:
    @echo -e "\n==== Installing xterm... ====\n\n"
    sudo apt-get install -y xterm
    @echo -e "\n\033[0;32m==== xterm installed successfully. ====\033[0m\n\n"
    @echo -e "\n==== Installing xfonts-base... ====\n\n"
    sudo apt-get install -y xfonts-base
    @echo -e "\n\033[0;32m==== xfonts-base installed successfully. ====\033[0m\n\n"

[group('setup')]
[doc('Install base development dependencies (Ubuntu): Python, xterm, sync dev deps, pre-commit')]
dev: install-xterm
    @echo -e "\n\033[0;32m==== Base development setup complete. ====\033[0m\n\n"

#=============================

[group('install')]
[doc('Install all dependencies for release')]
install-docker-release: install-docker verify-docker install-xterm
    @echo -e "\n\033[0;32m==== Release setup complete. ====\033[0m\n\n"

[group('install')]
[doc('install every dependency for every toolbox approach')]
install-dev: dev install-docker verify-docker
    @echo -e "\n\033[0;32m==== Full development setup complete. ====\033[0m\n\n"

[group('install')]
[doc('Install development environment for Threading runner (base dev only, no Docker)')]
install-dev-thread: dev
    @echo -e "\n\033[0;32m==== Threading development setup complete. ====\033[0m\n\n"
#=============================

# Development workflow recipes
[group('dev')]
[doc('Run repository pre-commit hooks and health checks')]
run-check:
    @echo -e "\n==== Running repository health checks with pre-commit hooks... ====\n\n"
    pixi run pre-commit run -a
    @echo -e "\n\033[0;32m==== Pre-commit checks complete. ====\033[0m\n\n"
#=============================

# Docker workflow recipes
[group('docker')]
[doc('Start the Docker-based simulation cluster ')]
docker-up:
    @echo -e "\n==== Starting Docker simulation cluster... ====\n\n"
    docker compose -f {{compose_file}} up --build -d
    @echo -e "\n\033[0;32m==== Docker cluster is up. ====\033[0m\n\n"

[group('docker')]
[doc('Stop and remove the Docker simulation cluster')]
docker-down:
    @echo -e "\n==== Stopping Docker simulation cluster... ====\n\n"
    docker compose -f {{compose_file}} down -v
    @echo -e "\n\033[0;32m==== Docker cluster stopped. ====\033[0m\n\n"

[group('docker')]
[doc('Tail logs from the Docker simulation cluster')]
docker-logs:
    docker compose -f {{compose_file}} logs -f

[group('docker')]
[doc('Prune unused Docker data (dangling images/containers)')]
docker-prune:
    docker system prune -f
    docker image prune -f

[group('docker')]
[doc('Run optimization using Docker runner')]
run-docker CONFIG_FILE MODEL_FILE:
    @just check-runner
    pixi run src/main.py --config-file {{CONFIG_FILE}} --model-file {{MODEL_FILE}} --use-docker

[group('docker')]
[doc('Run an interactive shell in a new container for a service')]
shell NAME:
    docker compose -f {{compose_file}} run --rm --service-ports {{NAME}} /bin/bash
#=============================

# Threading workflow recipes
[group('threading')]
[doc('Run optimization using Thread runner')]
run-thread CONFIG_FILE MODEL_FILE:
    pixi run python src/main.py --config-file {{CONFIG_FILE}} --model-file {{MODEL_FILE}}

#=============================

[group('utils')]
[doc('Show current environment information')]
info:
    @echo -e "\033[1;33m=== Environment Information ===\033[0m"
    @echo -e "\033[0;32mCurrent Python:\033[0m      $(which python3 2>/dev/null || echo 'Not found')"
    @echo -e "\033[0;32mPixi location:\033[0m       $(which pixi 2>/dev/null || echo 'Not found')"
    @echo -e "\033[0;32mDocker status:\033[0m       $(docker --version 2>/dev/null || echo 'Not installed')"
    @echo -e "\033[0;32mCurrent directory:\033[0m   $(pwd)"

[group('utils')]
[doc('Check that the selected runner is installed')]
check-runner:
    #!/usr/bin/env bash
    if ! command -v pixi >/dev/null 2>&1; then \
        echo -e "\033[0;31mError: pixi runner selected but not found in PATH.\033[0m\n"; \
        exit 1; \
    fi

#=============================
