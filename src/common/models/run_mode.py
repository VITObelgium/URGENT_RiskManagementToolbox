from enum import Enum


class RunMode(str, Enum):
    Evaluation = "evaluation"
    Optimization = "optimization"

    def __str__(self) -> str:
        return self.value
