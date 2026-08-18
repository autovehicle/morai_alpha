"""Convert MORAI privileged GT into raw metadata and Ego-frame labels."""

from typing import Dict, Iterable, Mapping, Tuple

from geometry.coordinate_transform import (
    CoordinateConvention,
    heading_to_ego_yaw,
    world_to_ego_xyz,
)
from network.UDP.protocol import EgoState, ObjectData

from .schema import ObjectLabel, RawEgoState, RawObjectState


DEFAULT_CLASS_MAPPING = {
    0: (0, 'pedestrian'),
    1: (1, 'vehicle'),
    2: (2, 'obstacle'),
}


class GTConverter:
    """World-to-Ego converter with an explicit coordinate verification gate."""

    def __init__(
        self,
        convention: CoordinateConvention,
        class_mapping: Mapping[int, Tuple[int, str]] = None,
        strict_convention: bool = True,
    ):
        convention.validate_supported()
        if strict_convention and not convention.verified:
            raise RuntimeError(
                'Coordinate convention is not verified. Complete the MORAI '
                'validation checklist before producing training labels.'
            )
        self.convention = convention
        self.class_mapping = dict(class_mapping or DEFAULT_CLASS_MAPPING)

    @staticmethod
    def raw_ego(ego: EgoState) -> RawEgoState:
        return RawEgoState(
            timestamp_ns=int(ego.timestamp_ns),
            world_position_m=(ego.pos_x, ego.pos_y, ego.pos_z),
            roll_pitch_yaw_deg=(ego.roll, ego.pitch, ego.yaw),
            velocity_kmh=(ego.vel_x, ego.vel_y, ego.vel_z),
            signed_velocity_kmh=ego.signed_vel,
            size_m=(ego.size_x, ego.size_y, ego.size_z),
        )

    @staticmethod
    def raw_object(obj: ObjectData) -> RawObjectState:
        return RawObjectState(
            timestamp_ns=int(obj.timestamp_ns),
            obj_id=int(obj.obj_id),
            obj_type=int(obj.obj_type),
            world_position_m=(obj.pos_x, obj.pos_y, obj.pos_z),
            heading_deg=obj.heading,
            size_xyz_m=(obj.size_x, obj.size_y, obj.size_z),
            velocity_kmh=(obj.vel_x, obj.vel_y, obj.vel_z),
            acceleration_mps2=(obj.acc_x, obj.acc_y, obj.acc_z),
        )

    def convert_objects(
        self,
        ego: EgoState,
        objects: Iterable[ObjectData],
    ) -> Tuple[ObjectLabel, ...]:
        labels = []
        ego_position = (ego.pos_x, ego.pos_y, ego.pos_z)
        for obj in objects:
            mapped = self.class_mapping.get(int(obj.obj_type))
            if mapped is None:
                continue
            class_id, class_name = mapped
            position_ego = world_to_ego_xyz(
                (obj.pos_x, obj.pos_y, obj.pos_z),
                ego_position,
                ego.yaw,
                self.convention,
            )
            labels.append(ObjectLabel(
                obj_id=int(obj.obj_id),
                class_id=int(class_id),
                class_name=str(class_name),
                position_ego_m=position_ego,
                size_lwh_m=(obj.size_x, obj.size_y, obj.size_z),
                yaw_ego_rad=heading_to_ego_yaw(
                    obj.heading, ego.yaw, self.convention
                ),
            ))
        return tuple(labels)

    def convert_frame(
        self,
        ego: EgoState,
        objects: Iterable[ObjectData],
    ) -> Tuple[RawEgoState, Tuple[RawObjectState, ...], Tuple[ObjectLabel, ...]]:
        object_tuple = tuple(objects)
        return (
            self.raw_ego(ego),
            tuple(self.raw_object(obj) for obj in object_tuple),
            self.convert_objects(ego, object_tuple),
        )
