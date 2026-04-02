from enum import Enum


class OptimizationStrategy(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    def __str__(self) -> str:
        return self.value
