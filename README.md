<p align="center">
  <img src="https://github.com/user-attachments/assets/d25f9fd7-7610-4725-9a8e-bead501ce568" width="250">
</p>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/Version-0.1.0-orange">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Code Quality" src="https://img.shields.io/badge/Code%20Quality-80%25%2B-yellow">
  <img alt="OS" src="https://img.shields.io/badge/OS-Ubuntu%2022.04-blueviolet">
  <img alt="Status" src="https://img.shields.io/badge/Status-Development-pink">
</p>

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
- **Just and common Unix tools**: git, curl

---

## Setup

### 1. Environment Installation

You can install either a **development environment** (recommended for developers) or a streamlined **release environment**.

> **Note:** Currently, Windows OS is not supported for this setup.

#### Development Environment:

This installs all the necessary tools, including development dependencies, pre-commit hooks, Docker, LaTeX, and Python/uv environments:

```shell
just install-dev
```

This command will specifically:

- Install Python 3.12 and set up a virtual environment using `uv`.
- Install LaTeX and TeXstudio for documentation purposes.
- Install Docker, verifying its functionality.
- Install all Python development dependencies and pre-commit hooks.

For development aimed at only threading solution:

```shell
just install-dev-thread
```

This command omits Docker installation

#### Release Environment:

Installs only the dependencies necessary to run the application in a production setting. Run:

```shell
just install-release
```

This command will specifically:

- Install Python 3.12 and set up a virtual environment using `uv`.
- Install Docker, verifying its functionality.
- Install all Python development dependencies and pre-commit hooks.

---

### 2. Documentation Editing

To edit the project documentation using TeXstudio, execute:

```shell
just edit-docs
```

- This will open the documentation in TeXstudio if installed.
- If TeXstudio is not detected, please install it along with LaTeX by running:
```shell
just install-latex
```

---

### 3. Repository Health Checks

Maintain codebase quality by executing pre-commit hooks, which will run set of the tools including pytest and coverage:

```shell
just run-check
```
---

## Getting started

### 1. Supported Reservoir Simulators:
- OpenDarts (1.1.3) [open_darts-1.1.3-cp310-cp310-linux_x86_64] (default)

### 2. Reservoir simulation interoperability
Interoperability between reservoir simulator and toolbox is established by the proper `Connector` [src/services/simulation_service/core/connectors].
Connector allows exchange information (simulation configuration and simulation results) between toolbox and simulator.


#### OpenDarts Connector
1. Reservoir simulation must be run from the **main.py** file (user must preserve the file name)
2. User should add the following dependencies in entry point for a simulation model:

   `from connectors.open_darts import OpenDartsConnector`

   `from connectors.open_darts import open_darts_input_configuration_injector`

   > **Note:** The `connectors` package will be automatically transfer from Toolbox to simulation model folder, no action
   > needed from User

3. Entry point for reservoir simulation must be decorated by `open_darts_input_configuration_injector`

   ``` python
   @open_darts_input_configuration_injector
   def run_darts(injected_configuration) -> None:
       ...
   ```

   The `injected_configuration` contain the control vector for optimization proces. It's strongly recommended to pass
   `injected_configuration` to model initialization:

   ```python
   @open_darts_input_configuration_injector
   def run_darts(injected_configuration, ...) -> None:
       m = Model(configuration=injected_configuration)
   ```
   and then

   ```python

   class Model(DartsModel):
       def __init__(self, configuration, ...):

           self._configuration = configuration
           super().__init__()
           ...

   ```

   This allows having access to injected configuration in a whole DartsModel object.

4. Well(s) connections will be taken from `configuration` using `OpenDartsConnector.get_well_connection_cells(...)`, thus user must import
`from connectors.open_darts import OpenDartsConnector`. Wells should be introduced in a simulation model in `def set_wells(self)` section:

   ```python
       def set_wells(self):

           wells = OpenDartsConnector.get_well_connection_cells(
               self._configuration, self.reservoir
           )

           for well_name, cells in wells.items():
               self.reservoir.add_well(well_name)
               for cell in cells:
                   i, j, k = cell
                   self.reservoir.add_perforation(
                       well_name, cell_index=(i, j, k), multi_segment=False
                   )

   ```

5. Whenever user want to return objective function to toolbox, the `broadcast_result` method from `OpenDartsConnector` should be used:

   ```python
   from connectors.common import SimulationResultType
   from connectors.open_darts import OpenDartsConnector
   ...

   OpenDartsConnector.broadcast_result(SimulationResultType.Heat, HeatValue)
   ```

   Then the heat value will be broadcasted back to toolbox with a proper flag from `SimulationResultType`.

   > **Note:** Recently only "Heat" is supported in `SimulationResultType`

6. Simulation model implementation is read to run an optimization process using RiskManagementToolbox.
7. All simulation model files must be archived in `.zip` format. User should ensure that after unpacking all simulation model files are accessible in folder without any additional subfolders.

### 3. Toolbox configuration file
RiskManagementToolbox is designed to use JSON configuration file, where the user defines the optimization problem(s),
initial state, and variable constraints.

Recently the following optimization problem(s) are supported:

1. **Well placement** - toolbox will try to find the optimal configuration of well(s): trajectory and perforation.

   Well placement problem definition if following by individual well(s) used in simulation.

   ```json
   {
      "well_placement": [
         {
            "well_name": str,
            "initial_state": {

            },
            "optimization_constraints": {

            }
         },
      ]
   }
   ```

   The initial state of the well defines the proper well type and all the mandatory well parameters neccessery to build well trajectory and define the completion intervals
   > **Note:** Recently only vertical well type `"well_type": "IWell"` is supported

   The example vertical well configuration:

   ```json
   {
     "initial_state": {
       "well_type": "IWell",
       "md": 2500,
       "md_step": 10,
       "wellhead": {
         "x": 400,
         "y": 400,
         "z": 0
       }
     }
   }
   ```
   > **Note:** If a user does not define a perforation interval, then by default the whole well md will be treated as perforated

   If the user wants to define the perforation interval:

   ```json
   {
     "initial_state": {
       "well_type": "IWell",
       "md": 2500,
       "md_step": 10,
       "wellhead": {
         "x": 400,
         "y": 400,
         "z": 0
       },
        "perforations": [
           {
            "start_md": 1000,
              "end_md": 1200
           }
        ]
     }
   }
   ```
   > **Note:** Recently only one perforation range is supported

   Each well initial state is followed by the optimization constraints part, where user can pick which parameters from
   well template will be used in an optimization process:

   As an example, if a user decides to optimize the x and y position of wellhead as well as well md, the optimization constraints will take the following form:
   ```json
   {
         "optimization_constraints": {
           "wellhead": {
             "x": {
               "lb": 10,
               "ub": 3190
             },
             "y": {
               "lb": 10,
               "ub": 3190
             }
           },
            "md": {
               "lb": 2000,
               "ub": 2700
            }
         }
   }
   ```

   where each parameter is bounded in lower (lb) and upper (ub) limit.

   > **Note:** Parameters which are not listed in optimization constraints will not take part of an optimization process and remains the same as defined in the initial state

2. **Optimization Strategy** - Depending on the problem, user can define whether to `maximize` or `minimize` the objective function. The optimization strategy is defined in the configuration file as follows:

```json
  "optimization_parameters": {
    "optimization_strategy": "maximize"
  }
```

If the user does not define the optimization strategy, the default value is `maximize`.

#### 3.1. Optimization parameters: linear inequalities

You can optionally express additional relationships between variables using sparse linear (in)equalities.

These have the following norm form:

```json
"optimization_parameters": {
  "optimization_strategy": "maximize",
  "linear_inequalities": {
    "A": [ {"INJ.md": 1.0, "PRO.md": 1.0}, {"INJ.md": 1.0, "PRO.md": 1.0} ],
    "b": [30.0, 3000.0],
    "sense": [">=", "<="]
  }
}
```

Each row (same index across `A`, `b`, `sense`) defines a single inequality over selected well attributes:

```
sum_j  (coeff_j * VAR_j)   (sense)   b_i
```

In the example above this yields the following constraints:

```
INJ.md + PRO.md >= 30.0   (lower bound)
INJ.md + PRO.md <= 3000.0  (upper bound)
```

##### Variable naming

Keys inside every `A` row use the pattern `<WELL_NAME>.<attribute>` (e.g. `INJ.md`, `PRO.md`).

All variables across all rows must reference the *same* attribute suffix (validator rule). For instance you may combine many wells on `md`, or many wells on `wellhead.x`, but you cannot mix `md` and `wellhead.x` in the same linear inequalities block.

Allowed senses per row:

* `"<="`  (less‑or‑equal)
* `">="`  (greater‑or‑equal)
* `"<"`   (treated as `<=` – strictness ignored for continuous search)
* `">"`   (treated as `>=`)

If `sense` is omitted, all rows default to `"<="`.

##### Multiple independent constraints

You may supply any number of rows. Some common patterns:

Lower & upper bounds on sum of MDs:

```json
"linear_inequalities": {
  "A": [ {"INJ.md": 1, "PRO.md": 1}, {"INJ.md": 1, "PRO.md": 1} ],
  "b": [1000, 3000],
  "sense": [">=", "<="]
}
```

Individual per‑well minimums expressed sparsely (can be redundant with simple bounds if you already set them in `boundaries`):

```json
"linear_inequalities": {
  "A": [ {"INJ.md": 1}, {"PRO.md": 1} ],
  "b": [800, 800],
  "sense": [">=", ">="]
}
```

During validation the upper bound is automatically reduced if it exceeds a physically plausible maximum derived from each well's positional search box.


##### Full example combining all optimization parameters

```jsonc
{
  "well_placement": [ /* ... wells definition ... */ ],
  "optimization_parameters": {
    "optimization_strategy": "minimize",
    "linear_inequalities": {
      "A": [
        {"INJ.md": 1, "PRO.md": 1},   // corridor lower
        {"INJ.md": 1, "PRO.md": 1},   // corridor upper
        {"INJ.md": 1},                  // individual min
        {"PRO.md": 1}                   // individual min
      ],
      "b": [1200, 5000, 800, 800],
      "sense": [">=", "<=", ">=", ">="]
    }
  }
}
```

#### 3.2. Example of the configuration file

Well placement problem for two wells with maximization optimization strategy:
   ```json
   {
     "well_placement": [
       {
         "well_name": "INJ",
         "initial_state": {
           "well_type": "IWell",
           "md": 2500,
           "md_step": 10,
           "wellhead": {
             "x": 400,
             "y": 400,
             "z": 0
           }
         },
         "optimization_constraints": {
           "wellhead": {
             "x": {
               "lb": 10,
               "ub": 3190
             },
             "y": {
               "lb": 10,
               "ub": 3190
             }
           }
         }
       },
       {
         "well_name": "PRO",
         "initial_state": {
           "well_type": "IWell",
           "md": 2500,
           "md_step": 10,
           "wellhead": {
             "x": 700,
             "y": 700,
             "z": 0
           }
         },
         "optimization_constraints": {
           "wellhead": {
             "x": {
               "lb": 10,
               "ub": 3190
             },
             "y": {
               "lb": 10,
               "ub": 3190
             }
           }
         }
       }
     ],
      "optimization_parameters": {
        "optimization_strategy": "maximize",
        "linear_inequalities": {
          "A": [ {"INJ.md": 1.0, "PRO.md": 1.0}, {"INJ.md": 1.0, "PRO.md": 1.0} ],
          "b": [30.0, 3000.0],
          "sense": [">=", "<="]
        }
      }
   }
   ```

---

### 4. Toolbox running

To run the RiskManagementToolbox the virtual environment corresponded with the repository must be
activated

```URGENT_RiskManagementToolbox$ source .venv/bin/activate```

User must set:
 1. path to config file (--config-file)
 2. path to model archive(--model-file)

optionally:
1. population size (--population-size)
2. iteration count without progress (--patience)
3. max generations count (-max-generations)

```
uv run src/main.py
usage: main.py [-h] --config-file CONFIG_FILE --model-file MODEL_FILE [--population-size POPULATION_SIZE] [--patience PATIENCE] [--max-generations MAX_GENERATIONS]
```

## Contact
For issues or contributions, please open a GitHub issue or contact the maintainers.
