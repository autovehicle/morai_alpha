"""Transport-neutral sensor receiver contracts.

Actual Camera/LiDAR transports are intentionally not selected in this pass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Union


SourceFrameId = Union[int, str]


@dataclass(frozen=True)
class SensorFrame:
    sensor_id: str
    modality: str
    timestamp_ns: int
    payload: Any
    source_frame_id: Optional[SourceFrameId] = None
    clock_domain: str = 'morai_simulation'
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SensorReceiver(ABC):
    """Interface implemented once the MORAI sensor transport is confirmed."""

    @abstractmethod
    def start(self, callback: Callable[[SensorFrame], None]) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
