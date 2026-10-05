"""Public reconstruction API."""

from dataclasses import dataclass
from pathlib import Path

from .family import default_family
from .io import read_json, write_json
from .model import PeriodicModel


@dataclass(frozen=True)
class Reconstruction:
    record: dict

    @property
    def recovered(self) -> bool:
        return self.record["status"] == "RECOVERED"

    @property
    def model(self) -> PeriodicModel | None:
        value = self.record.get("model")
        return None if value is None else PeriodicModel.from_record(value)

    @property
    def seconds(self) -> float:
        return self.record["seconds"]

    def save(self, path: str | Path) -> None:
        write_json(path, self.record)

    @classmethod
    def load(cls, path: str | Path) -> "Reconstruction":
        return cls(read_json(path))

    def verify(self, observations: str | Path) -> dict:
        from .verification import verify_trace

        return verify_trace(observations, self.record)


def reconstruct(
    observations: str | Path,
    *,
    family: str | Path | None = None,
    screen: bool = True,
    relations: str = "full",
    acceptance_mse: float = 1e-8,
) -> Reconstruction:
    """Infer a waveform and hidden weights from saved, precise observations.

    A successful result includes a replayable consistency trace, which does not
    establish uniqueness. The decoder assumes high-precision observations.
    """
    from .decoder import recover

    return Reconstruction(
        recover(
            observations, family or default_family(), relations, acceptance_mse, range_screen=screen
        )
    )
