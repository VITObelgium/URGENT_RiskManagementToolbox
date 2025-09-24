set shell := ["bash", "-euo", "pipefail", "-c"]
set dotenv-load := true

# Configuration variables
python_version := "3.12"
venv_dir := ".venv"
compose_file := "src/services/simulation_service/core/docker-compose.yml"

# Default recipe - shows available commands
default:
    @just --list --unsorted

# Core installation recipes
[group('setup')]
[doc('Install basic tools like git and curl needed by other installations')]
prerequisites:
    printf "\n==== Installing prerequisites (git, curl)... ====\n\n"
    sudo apt update
    sudo apt install -y git curl
    printf "\n\033[0;32m==== Prerequisites installed successfully. ====\033[0m\n\n"

[group('setup')]
[doc('Install LaTeX and TeXstudio')]
install-latex:
    printf "\n==== Installing LaTeX and TeXstudio... ====\n\n"
    sudo apt install -y texlive-latex-extra texstudio
    printf "\n\033[0;32m==== LaTeX and TeXstudio installed successfully. ====\033[0m\n\n"

[group('setup')]
[doc('Install Docker Engine with latest best practices')]
install-docker: prerequisites
    printf "\n==== Removing old Docker packages... ====\n\n"
    for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd.io runc; do
    sudo apt-get remove -y "$pkg" || true
    done

    printf "\n==== Setting up Docker repository... ====\n\n"
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}}) stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    printf "\n==== Installing Docker Engine... ====\n\n"
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    printf "\n\033[0;32m==== Docker installed successfully. ====\033[0m\n\n"

[group('setup')]
[doc('Verify Docker installation works correctly')]
verify-docker:
    printf "\n==== Verifying Docker installation... ====\n\n"
    sudo docker run --rm hello-world
    printf "\n\033[0;32m==== Docker verification complete. ====\033[0m\n\n"

[group('setup')]
[doc('Check and install Python and uv package manager')]
install-python-and-uv: prerequisites
    #!/usr/bin/env bash
    printf "\n==== Checking and installing Python {{python_version}} and uv if needed... ====\n\n"
    sudo apt update
    sudo apt upgrade -y

    if ! command -v python{{python_version}} >/dev/null; then
        echo "\033[0;33mPython {{python_version}} not found: Installing...\033[0m"
        if ! apt-cache policy python{{python_version}} | grep -q 'Candidate:'; then
            echo "\033[0;33mPython {{python_version}} not found in current repos: Adding Deadsnakes PPA...\033[0m"
            sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update
        fi
        sudo apt install -y python{{python_version}} python{{python_version}}-venv python3-pip
    else
        echo "\033[0;33mPython {{python_version}} is already installed. Skipping installation.\033[0m"
    fi

    # Install uv if not present
    if ! command -v uv >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

    # Create virtual environment
    ~/.local/bin/uv venv --python {{python_version}} {{venv_dir}}

[group('setup')]
[doc('Install xterm for external terminal logging')]
install-xterm:
    printf "\n==== Installing xterm... ====\n\n"
    sudo apt-get install -y xterm
    printf "\n\033[0;32m==== xterm installed successfully. ====\033[0m\n\n"
    printf "\n==== Installing xfonts-base... ====\n\n"
    sudo apt-get install -y xfonts-base
    printf "\n\033[0;32m==== xfonts-base installed successfully. ====\033[0m\n\n"

# High-level installation targets
[group('install')]
[doc('Install all dependencies for release (Ubuntu only)')]
install-release: install-python-and-uv install-docker verify-docker install-xterm
    printf "\n==== Installing release packages and pre-commit hooks... ====\n\n"
    uv sync --no-dev
    uv run pre-commit install
    printf "\n\033[0;32m==== Release setup complete. ====\033[0m\n\n"

[group('install')]
[doc('Install base development dependencies (Ubuntu): Python, uv, LaTeX, xterm, sync dev deps, pre-commit')]
dev: install-python-and-uv install-latex install-xterm
    printf "\n==== Installing base development packages and pre-commit hooks... ====\n\n"
    uv sync --all-groups
    uv run pre-commit install
    printf "\n\033[0;32m==== Base development setup complete. ====\033[0m\n\n"

[group('install')]
[doc('Install development environment for Docker runner (includes base dev + Docker)')]
dev-docker: dev install-docker verify-docker
    printf "\n\033[0;32m==== Docker development setup complete. ====\033[0m\n\n"

[group('install')]
[doc('Install development environment for Threading runner (base dev only, no Docker)')]
dev-thread: dev
    printf "\n\033[0;32m==== Threading development setup complete. ====\033[0m\n\n"

# Development workflow recipes
[group('dev')]
[doc('Update and lock all dependencies')]
lock-dev:
    @printf "\n==== Updating dependencies... ====\n\n"
    uv sync --all-groups

[group('dev')]
[doc('Run repository pre-commit hooks and health checks')]
run-check: lock-dev
    @printf "\n==== Running repository health checks with pre-commit hooks... ====\n\n"
    uv run pre-commit run -a
    @printf "\n\033[0;32m==== Pre-commit checks complete. ====\033[0m\n\n"

[group('dev')]
[doc('Run an interactive shell in a new container for a service')]
shell NAME:
    docker compose -f {{compose_file}} run --rm --service-ports {{NAME}} /bin/bash

# Docker workflow recipes
[group('docker')]
[doc('Start the Docker-based simulation cluster ')]
docker-up:
    printf "\n==== Starting Docker simulation cluster... ====\n\n"
    docker compose -f {{compose_file}} up --build -d
    printf "\n\033[0;32m==== Docker cluster is up. ====\033[0m\n\n"

[group('docker')]
[doc('Stop and remove the Docker simulation cluster')]
docker-down:
    printf "\n==== Stopping Docker simulation cluster... ====\n\n"
    docker compose -f {{compose_file}} down -v
    printf "\n\033[0;32m==== Docker cluster stopped. ====\033[0m\n\n"

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
    uv run src/main.py --config-file {{CONFIG_FILE}} --model-file {{MODEL_FILE}} --use-docker


# Threading workflow recipes
[group('threading')]
[doc('Run optimization using Thread runner')]
run-thread CONFIG_FILE MODEL_FILE:
    uv run src/main.py --config-file {{CONFIG_FILE}} --model-file {{MODEL_FILE}}


# Documentation recipes
[group('docs')]
[doc('Open the documentation in TeXstudio')]
edit-docs:
    @if command -v texstudio >/dev/null 2>&1; then \
        printf "\n==== Opening documentation in TeXstudio... ====\n\n"; \
        texstudio docs/Urgent_Risk_management_toolbox.tex; \
    else \
        echo "TeXstudio is not installed."; \
        read -p "Do you want to install it now? [y/N] " response; \
        if [ "$response" = "y" ]; then \
            just install-latex; \
            texstudio docs/Urgent_Risk_management_toolbox.tex; \
        else \
            echo "TeXstudio installation skipped. Please install TeXstudio to edit the documentation."; \
        fi; \
    fi

# Utility recipes
[group('utils')]
[doc('Show current environment information')]
info:
    @echo -e "\033[1;33m=== Environment Information ===\033[0m"
    @echo -e "\033[0;32mPython version:\033[0m      {{python_version}}"
    @echo -e "\033[0;32mVirtual environment:\033[0m {{venv_dir}}"
    @echo -e "\033[0;32mCurrent Python:\033[0m      $(which python3 2>/dev/null || echo 'Not found')"
    @echo -e "\033[0;32mUV location:\033[0m         $(which uv 2>/dev/null || echo 'Not found')"
    @echo -e "\033[0;32mDocker status:\033[0m       $(docker --version 2>/dev/null || echo 'Not installed')"
    @echo -e "\033[0;32mCurrent directory:\033[0m   $(pwd)"
