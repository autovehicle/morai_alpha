"""Transport-neutral calibration records.

No calibration value is invented here. Adapters must populate these records
from the selected MORAI sensor configuration and store a version per sequence.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


Matrix3x3 = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]
Matrix4x4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


@dataclass(frozen=True)
class CameraCalibration:
    sensor_id: str
    image_width: int
    image_height: int
    intrinsic: Matrix3x3
    sensor_to_ego: Matrix4x4
    distortion_model: Optional[str] = None
    distortion_coefficients: Tuple[float, ...] = ()


@dataclass(frozen=True)
class LidarCalibration:
    sensor_id: str
    sensor_to_ego: Matrix4x4
    point_fields: Tuple[str, ...]
