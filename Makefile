PYTHON_VERSION = 3.12

.PHONY: help prerequisites install-latex install-docker install-python-and-uv install install-dev install-release run-check edit-docs verify-docker

VENV_DIR ?= .venv
UV_IN_VENV := $(VENV_DIR)/bin/uv
VENV_ACTIVATE = $(VENV_DIR)/bin/activate
log = @printf "\n==== %s ====\n\n"

help:
	@echo "Available targets:"
	@echo "	prerequisites: Install basic tools like git and curl needed by other installations."
	@echo "	edit-docs: Open the documentation in TeXstudio."
	@echo "	install-dev: Install all dependencies for development (Ubuntu only)."
	@echo "	install-release: Install all dependencies for release (Ubuntu only)."
	@echo "	run-check: Run repository pre-commit hooks."

prerequisites:
	$(log) "Installing prerequisites (git, curl)..."
	sudo apt update
	sudo apt install -y git curl
	$(log) "Prerequisites installed successfully."

install-latex:
	$(log) "Installing LaTeX and TeXstudio..."
	sudo apt install -y texlive-latex-extra texstudio
	$(log) "LaTeX and TeXstudio installed successfully."

install-docker: prerequisites
	$(log) "Removing old Docker packages..."
	for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do \
		sudo apt-get remove -y "$$pkg"; \
	done
	$(log) "Setting up Docker repository..."
	sudo apt-get update
	sudo apt-get install -y ca-certificates curl gnupg lsb-release
	sudo install -m 0755 -d /etc/apt/keyrings
	sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
	sudo chmod a+r /etc/apt/keyrings/docker.asc

	# Add the Docker repository
	echo "deb [arch=$$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $$(. /etc/os-release && echo $${UBUNTU_CODENAME:-$${VERSION_CODENAME}}) stable" | \
		sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

	sudo apt-get update
	$(log) "Installing Docker Engine..."
	sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
	$(log) "Docker installed successfully."

verify-docker:
	$(log) "Verifying Docker installation..."
	sudo docker run --rm hello-world
	$(log) "Docker verification complete."

install-python-and-uv: prerequisites
	$(log) "Checking and installing Python $(PYTHON_VERSION) and uv if needed..."
	sudo apt update
	sudo apt upgrade -y
	if ! command -v python$(PYTHON_VERSION) >/dev/null; then \
		echo "Python $(PYTHON_VERSION) not found: Installing..."; \
		if ! apt-cache policy python$(PYTHON_VERSION) | grep -q 'Candidate:'; then \
			echo "Python $(PYTHON_VERSION) not found in current repos: Adding Deadsnakes PPA..."; \
			sudo apt install -y software-properties-common; \
			sudo add-apt-repository -y ppa:deadsnakes/ppa; \
			sudo apt update; \
		fi; \
		sudo apt install -y python$(PYTHON_VERSION) python$(PYTHON_VERSION)-venv python3-pip; \
	else \
		echo "Python $(PYTHON_VERSION) is already installed. Skipping installation."; \
	fi
	curl -LsSf https://astral.sh/uv/install.sh | sh
	. "$$HOME/.local/bin/env" && uv venv --python $(PYTHON_VERSION)

install-xterm: # needed for external terminal for logging
	$(log) "Installing xterm..."
	sudo apt-get install -y xterm
	$(log) "xterm installed successfully."
	$(log) "Installing xfonts-base..."
	sudo apt-get install -y xfonts-base
	$(log) "xfonts-base installed successfully."

install-release: install-python-and-uv install-docker verify-docker install-xterm
    $(log) "Installing release packages and pre-commit hooks..."
    uv pip sync
    uv run pre-commit install
    $(log) "Release setup complete."

install-dev: install-python-and-uv install-latex install-docker verify-docker install-xterm
    $(log) "Installing development packages and pre-commit hooks..."
    uv pip sync --group dev
    uv run pre-commit install
    $(log) "Development setup complete."

lock-dev:
    $(log) "Updating dependencies..."
    uv pip compile --group dev

run-check: lock-dev
    $(log) "Running repository health checks with pre-commit hooks..."
    uv run pre-commit run -a
    $(log) "Pre-commit checks complete."

edit-docs:
	@if command -v texstudio >/dev/null 2>&1; then \
		$(log) "Opening documentation in TeXstudio..."; \
		texstudio docs/Urgent_Risk_management_toolbox.tex; \
	else \
		echo "TeXstudio is not installed."; \
		read -p "Do you want to install it now? [y/N] " response; \
		if [ "$$response" = "y" ]; then \
			$(MAKE) install-latex; \
			texstudio docs/Urgent_Risk_management_toolbox.tex; \
		else \
			echo "TeXstudio installation skipped. Please install TeXstudio to edit the documentation."; \
		fi; \
	fi
