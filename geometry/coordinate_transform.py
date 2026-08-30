"""Pure coordinate transforms, independent from OpenCV/BEV rendering.

The default values represent the assumptions found in the old renderer. They
are guarded by an explicit verification flag; see ``docs/DATASET_SCHEMA.md``.
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CoordinateConvention:
    """Explicit source convention required by world-to-Ego conversion."""

    name: str = 'morai_world_x_forward_ccw__PROVISIONAL'
    yaw_zero_axis: str = '+x'
    yaw_positive: str = 'counter_clockwise'
    object_heading_matches_ego_yaw: bool = True
    size_x_is_length: bool = True
    size_y_is_width: bool = True
    verified: bool = False

    def validate_supported(self) -> None:
        if self.yaw_zero_axis != '+x':
            raise NotImplementedError(
                'Only a +X yaw-zero source frame is implemented; verify and '
                'add an explicit transform for another MORAI convention.'
            )
        if self.yaw_positive not in ('counter_clockwise', 'clockwise'):
            raise ValueError('yaw_positive must be counter_clockwise or clockwise')
        if not self.object_heading_matches_ego_yaw:
            raise NotImplementedError(
                'Object heading requires its own verified reference transform.'
            )
        if not self.size_x_is_length or not self.size_y_is_width:
            raise NotImplementedError(
                'Object size axes must be verified before label conversion.'
            )


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _mathematical_yaw(yaw_deg: float,
                      convention: CoordinateConvention) -> float:
    convention.validate_supported()
    sign = 1.0 if convention.yaw_positive == 'counter_clockwise' else -1.0
    return sign * math.radians(float(yaw_deg))


def world_to_ego_xyz(
    world_xyz_m: Tuple[float, float, float],
    ego_world_xyz_m: Tuple[float, float, float],
    ego_yaw_deg: float,
    convention: CoordinateConvention,
) -> Tuple[float, float, float]:
    """Transform world xyz into target Ego forward/left/up coordinates.

    Roll and pitch are intentionally not applied in this first planar object
    detection contract. If 3D labels require a fully level-independent frame,
    a verified SE(3) transform must replace this helper.
    """
    yaw = _mathematical_yaw(ego_yaw_deg, convention)
    dx = float(world_xyz_m[0]) - float(ego_world_xyz_m[0])
    dy = float(world_xyz_m[1]) - float(ego_world_xyz_m[1])
    dz = float(world_xyz_m[2]) - float(ego_world_xyz_m[2])

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    # R_world_from_ego.T @ delta_world
    x_forward = cos_yaw * dx + sin_yaw * dy
    y_left = -sin_yaw * dx + cos_yaw * dy
    return x_forward, y_left, dz


def heading_to_ego_yaw(
    object_heading_deg: float,
    ego_yaw_deg: float,
    convention: CoordinateConvention,
) -> float:
    """Return object heading relative to Ego in target CCW radians."""
    object_yaw = _mathematical_yaw(object_heading_deg, convention)
    ego_yaw = _mathematical_yaw(ego_yaw_deg, convention)
    return wrap_to_pi(object_yaw - ego_yaw)


def quaternion_to_roll_pitch_yaw_deg(
    x: float, y: float, z: float, w: float
) -> Tuple[float, float, float]:
    """Standard ZYX (aerospace) quaternion-to-Euler conversion, in degrees.

    Used to recover Ego roll/pitch from a ``/tf`` orientation quaternion,
    since MORAI's ``morai_msgs/EgoVehicleStatus`` only reports yaw (as
    ``heading``). Verified against a live sample: the yaw this function
    derives from the ``map -> base_link`` quaternion matched
    ``EgoVehicleStatus.heading`` exactly (86.64 deg both), confirming both
    describe the same pose in degrees.
    """
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
