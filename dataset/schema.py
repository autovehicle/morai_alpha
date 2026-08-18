"""Versioned, JSON-serializable Dataset records."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union


SCHEMA_VERSION = '0.1.0-draft'
TARGET_COORDINATE_FRAME = 'ego_forward_left_up'
SourceFrameId = Union[int, str]


def validate_timestamp_ns(value: int) -> int:
    """Validate that a timestamp fits a signed int64 representation."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('timestamp_ns must be an int')
    if value < -(2 ** 63) or value > (2 ** 63 - 1):
        raise OverflowError('timestamp_ns does not fit int64')
    return value


@dataclass(frozen=True)
class RawEgoState:
    timestamp_ns: int
    world_position_m: Tuple[float, float, float]
    roll_pitch_yaw_deg: Tuple[float, float, float]
    velocity_kmh: Tuple[float, float, float]
    signed_velocity_kmh: float
    size_m: Tuple[float, float, float]


@dataclass(frozen=True)
class RawObjectState:
    timestamp_ns: int
    obj_id: int
    obj_type: int
    world_position_m: Tuple[float, float, float]
    heading_deg: float
    size_xyz_m: Tuple[float, float, float]
    velocity_kmh: Tuple[float, float, float]
    acceleration_mps2: Tuple[float, float, float]


@dataclass(frozen=True)
class ObjectLabel:
    obj_id: int
    class_id: int
    class_name: str
    position_ego_m: Tuple[float, float, float]
    size_lwh_m: Tuple[float, float, float]
    yaw_ego_rad: float


@dataclass(frozen=True)
class SyncSource:
    timestamp_ns: int
    offset_ns: int
    source_frame_id: Optional[SourceFrameId] = None
    clock_domain: str = 'morai_simulation'


@dataclass(frozen=True)
class FrameMetadata:
    sequence_id: str
    frame_id: int
    sample_timestamp_ns: int
    anchor_stream: str
    coordinate_convention: str
    ego_raw: RawEgoState
    objects_raw: Tuple[RawObjectState, ...]
    synchronization: Mapping[str, SyncSource]


def to_dict(record: Any) -> Dict[str, Any]:
    """Convert a schema dataclass into a JSON-compatible dictionary."""
    return asdict(record)
