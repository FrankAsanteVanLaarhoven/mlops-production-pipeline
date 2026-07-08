"""Abstract interface for model serving deployment."""

from abc import ABC, abstractmethod


class BaseServingService(ABC):
    """Abstract interface for a model serving deployer."""

    @abstractmethod
    def start(
        self,
        host: str,
        port: int,
        registry_root: str,
        max_abs_feature_value: float,
        production_data_path: str | None = None,
    ) -> None:
        """Start the serving deployment server."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Shut down the serving engine."""
        pass
