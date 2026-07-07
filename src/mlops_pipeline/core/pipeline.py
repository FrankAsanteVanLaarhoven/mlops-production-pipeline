"""Abstract interfaces for MLOps pipeline and steps."""

from abc import ABC, abstractmethod
from typing import Any


class BaseStep(ABC):
    """Abstract interface for a pipeline step."""

    @abstractmethod
    def run(self, config: dict, *args: Any, **kwargs: Any) -> Any:
        """Execute the step logic."""
        pass


class BasePipeline(ABC):
    """Abstract interface for a pipeline orchestrator."""

    @abstractmethod
    def run(self, config_path: str) -> None:
        """Run the full gated training/selection lifecycle."""
        pass
