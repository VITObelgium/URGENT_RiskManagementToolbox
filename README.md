# Introduction
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project.

---

## Development Requirements

- **Operating System**: Ubuntu 22.04 (recommended)
- **Python Version**: 3.12 (configured in the Makefile)
- **Make and common Unix tools**: git, curl

---

## Getting Started

### 1. Setup Prerequisites

To install basic dependencies (`git`, `curl`) essential for setup, run:

```shell
make prerequisites
```

---

### 2. Environment Installation

You can install either a **development environment** (recommended for developers) or a streamlined **release environment**.

#### Development Environment:

This installs all the necessary tools, including development dependencies, pre-commit hooks, Docker, LaTeX, and Python/uv environments:

```shell
make install-dev
```

This command will specifically:

- Install Python 3.12 and set up a virtual environment using `uv`.
- Install LaTeX and TeXstudio for documentation purposes.
- Install Docker, verifying its functionality.
- Install all Python development dependencies and pre-commit hooks.

> **Note:** Currently, Windows OS is not supported for this setup.

#### Release Environment:

Installs only the dependencies necessary to run the application in a production setting. Run:

```shell
make install-release
```

---

### 3. Documentation Editing

To edit the project documentation using TeXstudio, execute:

```shell
make edit-docs
```

- This will open the documentation in TeXstudio if installed.
- If TeXstudio is not detected, please install it along with LaTeX by running:
```shell
make install-latex
```

---

### 4. Repository Health Checks

Maintain codebase quality by executing pre-commit hooks:

```shell
make run-check
```

---

### 5. Docker Installation and Verification

Docker is necessary for building and containerizing deployments. Install it using:

```shell
make install-docker
```

After installation, verify Docker functionality with:

```shell
make verify-docker
```

---

### Manual Virtual Environment Handling

- If for some reason you did not initially run `make install-dev` or `make install-release`, activate the Python virtual environment manually and install git hooks by running:

```shell
source .venv/bin/activate
pre-commit install
```

---

## Build and Test

### Running Tests

Use `pytest` to run the project's test suite:

```shell
pytest
```

### Pre-commit Checks Individually

To perform project compliance checks manually, run:

```shell
make run-check
```

---

## Notes

- Familiarity with Unix commands and the `make` utility is assumed.
- If you encounter issues during installation or operational errors, verify system compatibility and reread the instructions carefully.

---
