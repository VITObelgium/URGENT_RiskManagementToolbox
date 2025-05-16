
<p align="center">
  <img src="https://github.com/user-attachments/assets/d25f9fd7-7610-4725-9a8e-bead501ce568" width="250">
</p>

![In Development](https://img.shields.io/badge/status-In%20Development-yellow?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.12-blue)  ![License](https://img.shields.io/badge/License-MIT-green)

# URGENT Risk Management Toolbox

## Introduction

This Python-based toolbox is designed to optimize geothermal reservoir development by combining advanced Thermo-Hydro-Mechanical (THM) numerical modeling, machine learning (ML) optimization routines, and automated feedback loops. The goal is to maximize total heat energy production while minimizing the risk of induced seismicity on known faults.

## Core Components:

1. **THM Reservoir Models**

    Simulate the coupled thermal, hydraulic, and mechanical behavior of the subsurface, based on geological models derived from seismic data.

2. **Machine Learning Optimization**

    Algorithms adjust well locations and operational parameters (e.g., flow rate, injection temperature) to balance maximum heat recovery with minimal seismic risk.

3. **Linking & Automation Scripts**

    Scripts facilitate communication between the THM simulations and ML routines, enabling iterative simulation cycles to determine optimal well placement and operation.

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

The `make install-dev` or `make install-release` targets will set up the Python virtual environment using `uv` and install git hooks.

If you need to run commands within this environment manually (e.g., a Python script or `pytest`), you can use `uv run`:

```shell
uv run python your_script.py
uv run pytest
```

To activate a shell session within the virtual environment, you can use:

```shell
uv shell
```

Additionally, if you need to manually install git hooks (e.g., if you skipped the `make install-*` targets initially or cloned to a new location without running them):

```shell
uv run pre-commit install
```

---

## Build and Test

### Running Tests

Use `pytest` to run the project's test suite. The `make install-dev` target installs `pytest`. You can run tests using:

```shell
make run-check
```
This will also run tests as part of the pre-commit hooks. To run tests directly:
```shell
uv run pytest
```
Or, if you have activated the environment using `uv shell`:
```shell
pytest
```

---

## Notes

- Familiarity with Unix commands and the `make` utility is assumed.
- If you encounter issues during installation or operational errors, verify system compatibility and reread the instructions carefully.

---
