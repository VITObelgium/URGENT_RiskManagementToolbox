<p align="center">
  <img src="https://github.com/user-attachments/assets/d25f9fd7-7610-4725-9a8e-bead501ce568" width="250">
</p>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/Version-0.2.0-orange">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Code Quality" src="https://img.shields.io/badge/Code%20Quality-80%25%2B-yellow">
  <img alt="OS" src="https://img.shields.io/badge/OS-Ubuntu%2022.04-blueviolet">
  <img alt="Pixi" src=https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square>
  <img alt="Status" src="https://img.shields.io/badge/Status-Development-pink">
</p>

# URGENT Risk Management Toolbox

## Table of Contents
- [Introduction](#introduction)
- [Core components](#core-components)
- [Development Requirements](#development-requirements)
- [Environment Installation](#environment-installation)
- [Getting started](#getting-started)
  - [Supported Reservoir Simulators](#supported-reservoir-simulators)
  - [Execution modes](#execution-modes)
  - [Reservoir simulation interoperability](#reservoir-simulation-interoperability)
    - [OpenDarts Connector](#opendarts-connector)
  - [Global Configuration](#global-configuration)
  - [Run configuration file](#run-configuration-file)
    - [Input file schemas](#input-file-schemas)
- [Implemented services](#implemented-services)
   - [Well design service](#well-design-service)
- [Configuration example](#configuration-example)
- [Known issues](#known-issues)
- [Contact](#contact)

## Introduction

This Python-based toolbox is designed to optimize geothermal reservoir development by combining advanced Thermo-Hydro-Mechanical (THM) numerical modeling, machine learning (ML) optimization routines, and automated feedback loops. The goal is to maximize total heat energy production while minimizing the risk of induced seismicity on known faults.

## Core Components

1. **THM Reservoir Models**

    Simulate the coupled thermal, hydraulic, and mechanical behavior of the subsurface, based on geological models derived from seismic data.

2. **Machine Learning Optimization**

    Algorithms adjust well locations and operational parameters (e.g., flow rate, injection temperature) to balance maximum heat recovery with minimal seismic risk.

3. **Linking & Automation Scripts**

    Scripts facilitate communication between the THM simulations and ML routines, enabling iterative simulation cycles to determine optimal well placement and operation.

---

## Development Requirements

- **Operating System**: Ubuntu 22.04, Ubuntu 24.04
- **Python Version**: 3.12 (managed via pixi; configured in `pyproject.toml`)
- **Common Unix tools**: git, curl

---

## Environment Installation


You can install either a **development environment** (recommended for developers) or a streamlined **release environment**.


### Development Environment

Installs the tools needed for development (Python via pixi, dev dependencies, pre-commit):


```shell
pixi install -e dev
```

#### Repository Health Checks

Maintain codebase quality by executing pre-commit hooks, which will run set of the tools including pytest and coverage:

```shell
pixi run -e dev pre-commit run -a
```

#### Logs and artifacts pruning
To clean toolbox logs and produced artifacts run following pixi task:
``` shell
pixi run -e dev clean-all
```

### Release Environment
Installs the runtime dependencies:
```shell
pixi install
```

---

## Getting started

### 1. Supported Reservoir Simulators
#### OPEN-DARTS
- OpenDarts (1.1.3) [open_darts-1.1.3-cp310-cp310-linux_x86_64] (default)

### 2. Execution modes

The toolbox supports two execution modes for running simulations:

- Threaded runner (default): local execution without containers.

```shell
pixi run src/main.py --config-file <config_filepath> --model-file <model_filepath>
```

- Docker runner: containerized workers (required Docker installation).

```shell
pixi run src/main.py --config-file <config_filepath> --model-file <model_filepath> --use-docker
```

>**Note**: If you encounter Error launching 'src/main.py': Permission denied (os error 13), invoke the command using Python explicitly:
``` shell
pixi run python --config-file ...
```


### 3. Reservoir simulation interoperability

Interoperability between the reservoir simulator and the Toolbox is achieved through a dedicated `Connector`
(`src/services/simulation_service/core/connectors`).

The Connector enables bidirectional data exchange between the Toolbox and the simulator, including:
- simulation configuration (control vectors),
- simulation results (objective function values).

---

#### 3.1 OpenDarts Connector

1. **Simulation entry point**

   The reservoir simulation **must** be launched from a file named **`main.py`**.
   The file name must be preserved.

2. **Required imports**

   Add the following dependencies to the simulation entry point:

   ```python
   from connectors.open_darts import OpenDartsConnector
   from connectors.open_darts import open_darts_input_configuration_injector
   ```

   > **Note:**
   > The `connectors` package is automatically transferred from the Toolbox to the simulation model directory.
   > No user action is required.

3. **Configuration injection**

   The simulation entry-point function must be decorated with
   `open_darts_input_configuration_injector`:

   ```python
   @open_darts_input_configuration_injector
   def run_darts(injected_configuration) -> None:
       ...
   ```

   The `injected_configuration` contains the control vector for the optimization process.
   It is **strongly recommended** to pass this configuration to the model during initialization:

   ```python
   @open_darts_input_configuration_injector
   def run_darts(injected_configuration, ...) -> None:
       model = Model(configuration=injected_configuration)
   ```

   ```python
   class Model(DartsModel):
       def __init__(self, configuration, ...):
           self._configuration = configuration
           super().__init__()
           ...
   ```

   This ensures that the injected configuration is accessible throughout the entire `DartsModel` instance.

4. **Well connections**

   Well connections are extracted from the injected configuration using
   `OpenDartsConnector.get_well_connection_cells(...)`.

   Ensure the following import is present:

   ```python
   from connectors.open_darts import OpenDartsConnector
   ```

   Wells must be defined in the `set_wells` method of the simulation model:

   ```python
   def set_wells(self):

       wells = OpenDartsConnector.get_well_connection_cells(
           self._configuration, self.reservoir
       )

       for well_name, cells in wells.items():
           self.reservoir.add_well(well_name)
           for i, j, k in cells:
               self.reservoir.add_perforation(
                   well_name,
                   cell_index=(i, j, k),
                   multi_segment=False
               )
   ```

5. **Returning objective function values**

   To return an objective function value to the Toolbox, use
   `OpenDartsConnector.broadcast_result(...)`:

   ```python
   from connectors.common import SimulationResultType
   from connectors.open_darts import OpenDartsConnector

   OpenDartsConnector.broadcast_result(
       "Heat",
       heat_value
   )
   ```

   The result is transmitted back to the Toolbox with the corresponding name as ex.: "HEAT".
> Note: The parameters name must be the same as the one defined in the run configuration file.



6. **Optimization readiness**

   Once implemented as described above, the simulation model is ready to be used in an optimization workflow with **RiskManagementToolbox**.

7. **Packaging requirements**

   All simulation model files must be archived in a single `.zip` file.
   After extraction, all files must be located directly in the root directory (no nested subfolders).

### 4. Global Configuration

Global toolbox settings are defined in `pyproject.toml` under the `[toolbox-config]` table.

| Parameter                    | Type | Description|
|-----|----|----|
| `simulation_timeout_seconds` | int  | Maximum allowed time (in seconds) for a single simulation run before it is terminated. |


### 5. Run configuration file
RiskManagementToolbox is designed to use JSON configuration file, where the user defines the optimization problem(s), initial state, and variable constraints.

Configuration file define services to be used for simulation and optimization as well as the global optimization parameters as objectives or linear inequality constraints.

The toolbox expects **one JSON file** that defines:

1. Services name and parameters for optimization (with their bounds)
3. How the optimization algorithm is configured

### Input file schemas

Input configuration file is a JSON file with the structures presented in `schemas/x.y.z.json`

### Top-level structure:

```json
{
  "=== SERVICE NAME ===": service item(s),
  "optimization_parameters": { ... }
}
```

## Implemented services

| Service name  | Description                                                            |
|---------------|------------------------------------------------------------------------|
| `well_design` | Service sesponsible for well(s) placement, trajectory and completion.  |




### Well design service

`well_design` expecting is an array of objects (service items):

```json
{
  "well_name": "INJ",
  "initial_state": { ... },
  "parameter_bounds": { ... }
}
```

#### Mandatory fields

| Field | Required | Description |
|----|----|----|
| `well_name` | ✅ | Unique identifier used across the configuration |
| `initial_state` | ✅ | Defines well initial (user defined) geometry and completion |
| `parameter_bounds` | ✅ | Selects which parameters (from initial state) are optimized, with the lower and upper range |

### Initial state
The `initial_state` defines the **baseline geometry** of a well.

#### Common fields
| Field | Required | Description |
|----|----|----|
| `wellhead` | ✅ | XYZ coordinates of wellhead ex. {"x": 400,"y": 400, "z": 0}  |
|`perforations`| ❌ | Optional (but well without perforation may be skipped in simulator - check the worker log(s) file):: dictionary of name and perforation interval of well in measure depth ex. {"perforation_1": {"start_md":  1000.00, "end_md": 1200.00}, "perforation_2":{"start_md": 1500, "end_md": 1550}}|
| `md_step` | ❌ | Optional:  well trajectory discretization step, default: `0.5 m`, `≥ 0.1m` |

Data Validation Rules
- **Perforation Alignment**: Any perforation defined beyond the well's total `md` is automatically truncated. Intervals starting after the total `md` are discarded.
- **Overlap Detection**: The system ensures no two perforation intervals overlap.
- **Automatic Sorting**: Perforations are automatically ordered by their start depth.


The well type is selected using the `well_type` discriminator:

| well_type | Model | Description |
|---------|------|------------|
| `IWell` | IWellModel | Vertical well |
| `JWell` | JWellModel | Build-and-hold well (J shape) |
| `SWell` | SWellModel | Multi-curvature well (S shape) |
| `HWell` | HWellModel | Horizontal well |

#### Vertical well

The `IWell` represents a straight, inclined well trajectory. It is defined by its surface location, total measured depth, and calculation resolution.

| Field | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `well_type` | Literal | Fixed identifier for the trajectory type. | Must be `IWell` |
| `md` | Float | **Measured Depth**: Total length of the wellbore. | `> 0.0` |


Example:
``` json
{
  "well_type": "IWell",
  "md": 2500.0,
  "wellhead": {
    "x": 1450.0,
    "y": 2200.0,
    "z": 0.0
  },
  "md_step": 0.5,
  "perforations": {
   "p1":
    {
      "start_md": 1800.0,
      "end_md": 1950.0
    }
  }
}
```



#### J shape well

The `JWell` represents a directional well trajectory consisting of an initial vertical/linear section, a curved build section, and a final tangential linear section.

| Field | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `well_type` | Literal | Fixed identifier for the trajectory type. | Must be `JWell` |
| `md_linear1` | Float | **Initial Linear Section**: Length of the first straight section. | `> 0.0` |
| `md_curved` | Float | **Curved Section**: Length of the build/curve section. | `> 0.0` |
| `dls` | Float | **Dogleg Severity**: Curvature rate of the build section in °/30m. The positive value define anticlockwise build direction | `-45.0` to `45.0` |
| `md_linear2` | Float | **Final Linear Section**: Length of the final tangential section. | `> 0.0` |
| `azimuth` | Float | Azimuth of the well in degrees. | `0.0` to `< 360.0` |


Example:
```JSON
{
  "well_type": "JWell",
  "md_linear1": 500.0,
  "md_curved": 300.0,
  "dls": 5.0,
  "md_linear2": 700.0,
  "wellhead": {
    "x": 1000.0,
    "y": 1000.0,
    "z": 0.0
  },
  "azimuth": 45.0,
  "md_step": 0.5,
  "perforations": {
   "p1":
    {
      "start_md": 1200.0,
      "end_md": 1450.0
    }
  }
}
```

#### S shape well

The `SWell` represents a complex directional well trajectory with two curved sections, often used to offset the lateral position of the wellbore while maintaining a final vertical or tangential orientation.

| Field | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `well_type` | Literal | Fixed identifier for the trajectory type. | Must be `SWell` |
| `md_linear1` | Float | **First Linear Section**: Initial straight section. | `> 0.0` |
| `md_curved1` | Float | **First Curve**: Length of the first build/drop section. | `> 0.0` |
| `dls1` | Float | **First DLS**: Dogleg Severity for the first curve in °/30m. The positive value define anticlockwise build direction  | `-45.0` to `45.0` |
| `md_linear2` | Float | **Second Linear Section**: Intermediate straight section. | `> 0.0` |
| `md_curved2` | Float | **Second Curve**: Length of the second build/drop section. | `> 0.0` |
| `dls2` | Float | **Second DLS**: Dogleg Severity for the second curve in °/30m. The positive value define anticlockwise build direction  | `-45.0` to `45.0` |
| `md_linear3` | Float | **Third Linear Section**: Final straight section. | `> 0.0` |
| `azimuth` | Float | The horizontal direction of the well in degrees. | `0.0` to `< 360.0` |


Example:
``` JSON
{
  "well_type": "SWell",
  "md_linear1": 400.0,
  "md_curved1": 200.0,
  "dls1": 5.0,
  "md_linear2": 500.0,
  "md_curved2": 300.0,
  "dls2": -3.0,
  "md_linear3": 600.0,
  "wellhead": {
    "x": 500.0,
    "y": 500.0,
    "z": 0.0
  },
  "azimuth": 180.0,
  "md_step": 0.5,
  "perforations": {
   "p1":
    {
      "start_md": 1600.0,
      "end_md": 1900.0
    }
  }
}
```

#### Horizontal well


The `HWell` represents a horizontal well trajectory defined by a specific True Vertical Depth (TVD) and a lateral extension (width). The system automatically calculates the necessary build curve to transition from the wellhead to the horizontal section using dls of 4.0° /30m

| Field | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `well_type` | Literal | Fixed identifier for the trajectory type. | Must be `HWell` |
| `TVD` | Float | **True Vertical Depth**: The vertical depth of the horizontal lateral. | `> 0.0` |
| `md_lateral` | Float | **Lateral Length**: The length of the horizontal section. | `> 0.0` |
| `azimuth` | Float | The horizontal direction of the lateral in degrees. | `0.0` to `< 360.0` |


Example:
```json
{
  "well_type": "HWell",
  "TVD": 1000.0,
  "md_lateral": 1500.0,
  "wellhead": {
    "x": 2000.0,
    "y": 2000.0,
    "z": 0.0
  },
  "azimuth": 90.0,
  "md_step": 1.0,
  "perforations": {
   "p1":
    {
      "start_md": 1200.0,
      "end_md": 2500.0
    }
  }
}
```

#### Data Validation Rules
- **Geometry Check**: The `TVD` must be sufficient to accommodate the calculated curvature radius of the build section.
- **Automatic MD Calculation**: The total Measured Depth is automatically derived from the vertical transition and lateral width for perforation clipping.


### Optimization constraints


Optimization constraints (`parameter_bounds`) define the boundaries for individual well parameters

####  Parameter Boundaries
Boundaries define the search space (Lower Bound and Upper Bound) for specific well attributes.

 **The optimizing well attribute has to present in initial state.**

| Field | Type | Description |
| :--- | :--- | :--- |
| `parameter_bounds` | Dictionary | Maps a variable name to a {"lb": float, "ub": float} |

> Important!
>> For nested parameters like wellhead coordinates or perforations, the following naming convention is used:
>
>  "main_parameter":{ "sub_parameter": {"sub_sub_parameter" :{ "lb": xxx, "ub": yyy }}} example:

Example:
```JSON
  "parameter_bounds": {
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
    },
    "perforations": {
      "p1": {
        "start_md": {
          "lb": 2000,
          "ub": 2200
        }
      }
    }
  }
```

### Optimization Parameters Section

The toolbox uses the `optimization_parameters` block to define how the optimization engine (e.g., PSO) behaves and to set global constraints across multiple wells.

####
These settings control the execution and termination of the optimization process.

| Parameter | Type           | Default  | Description                                                                                                                                                                                                                                                                                                              |
| :--- |:---------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `objectives` | dict[str, str] | REQUIRED | Dictionary of objective to optimize with optimization strategy ex. {"Heat": "maximize", "WellLength": "minimize"}. The objective names must match the values broadcasted from connector, otherwise the optimization run will be aborded. If multiple objectives are present the RMT will run in pareto optimization mode. |
| `max_generations` | Integer        | `10`     | Maximum number of iterations for the algorithm.                                                                                                                                                                                                                                                                          |
| `population_size` | Integer        | `10`     | Number of solution candidates to evaluate per generation.                                                                                                                                                                                                                                                                |
| `max_stall_generations` | Integer        | `10`     | Generations to wait for improvement before early stopping.                                                                                                                                                                                                                                                               |
| `worker_count` | Integer        | `4`      | Number of parallel simulation workers (limited by physical CPU cores).                                                                                                                                                                                                                                                   |

#### Linear inequalities allow you to define relationships between variables across different wells, such as a combined "drilling budget" for total measured depth.

- **A**: List of coefficient maps. Variables must be named as `service_name.attribute.subattribute` (e.g., `well_design.PRO.md` or `well_design.INJ.perforations.p1.start_md`).
- **b**: List of constant values (right-hand side of the inequality).
- **sense**: List of operators (`<=`, `>=`, `<`, `>`). Defaults to `<=` if omitted.

> Important!
>> The number of rows in `A` and `b` must match the number of variables in the optimization space.

>> For perforation optimization make sure that end_md of perforation is greater than start_md

#### Example: Combined Depth Constraint
To ensure the total length of two wells (`INJ` and `PRO`) is between 1200m and 5000m:

```json
"optimization_parameters": {
  "objectives": {"HEAT": "maximize"},
  "population_size": 20,
  "linear_inequalities": {
    "A": [
      { "well_design.INJ.md": 1.0, "well_design.PRO.md": 1.0 },
      { "well_design.INJ.md": 1.0, "well_design.PRO.md": 1.0 }
    ],
    "b": [1200.0, 5000.0],
    "sense": [">=", "<="]
  }
}
```
## Configuration example
### **Case Summary:**

The Well design service will be use to determine the optimal wells placement and trajectory for maximizing the heat production.

1. **Search Space**:
    - The injector (`INJ`) is confined to a 900m x 900m square in the bottom-left area.
    - The producer (`PRO`) is confined to a 1500m x 1500m square in the top-right area.
    - Both wells can vary in depth between **1500m** and **3000m**.

2. **Linear Constraint**: The total combined depth of both wells is restricted to **5000 meters** maximum (enforced via ). `linear_inequalities`
3. **Strategy**: The engine will attempt to **maximize** the objective function (e.g., heat production) over **50 generations** using **4 parallel workers**.
4. **Completions**:
    - `INJ` has a fixed 500m perforation at the toe.
    - `PRO` has no perforations defined in , so the system will default to perforating its entire length. `initial_state`
5. **Optimization strategy**:
	User defined parameter "HEAT" should be "maximized"

```json
{
  "well_design": [
    {
      "well_name": "INJ",
      "initial_state": {
        "well_type": "IWell",
        "md": 2500.0,
        "md_step": 1.0,
        "wellhead": {
          "x": 500.0,
          "y": 500.0,
          "z": 0.0
        },
        "perforations": {
          "p1": {
            "start_md": 2000.0,
            "end_md": 2500.0
          }
        }
      },
      "parameter_bounds": {
        "wellhead": {
          "x": {
            "lb": 100.0,
            "ub": 1000.0
          },
          "y": {
            "lb": 100.0,
            "ub": 1000.0
          }
        },
        "md": {
          "lb": 1500.0,
          "ub": 3000.0
        }
      }
    },
    {
      "well_name": "PRO",
      "initial_state": {
        "well_type": "IWell",
        "md": 2500.0,
        "md_step": 1.0,
        "wellhead": {
          "x": 1500.0,
          "y": 1500.0,
          "z": 0.0
        },
        "perforations": {
          "p1": {
            "start_md": 2100.0,
            "end_md": 2200.0
          }
        }
      },
      "parameter_bounds": {
        "wellhead": {
          "x": {
            "lb": 1000.0,
            "ub": 2500.0
          },
          "y": {
            "lb": 1000.0,
            "ub": 2500.0
          }
        },
        "md": {
          "lb": 1500.0,
          "ub": 3000.0
        }
      }
    }
  ],
  "optimization_parameters": {
  "objectives": {"HEAT": "maximize"},
    "max_generations": 50,
    "population_size": 20,
    "max_stall_generations": 5,
    "worker_count": 4,
    "linear_inequalities": {
      "A": [
        {
          "well_design.INJ.md": 1.0,
          "well_design.PRO.md": 1.0
        }
      ],
      "b": [
        5000.0
      ],
      "sense": [
        "<="
      ]
    }
  }
}


```

## Known Issues
 - DRMT with docker backend cannot start simulation server

## Contact
For issues or contributions, please open a GitHub issue or contact the maintainers.
