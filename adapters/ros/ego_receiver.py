"""MORAI Ego vehicle state ROS receiver.

Subscribes to ``/Ego_topic`` (``morai_msgs/EgoVehicleStatus``, ~44Hz
observed) and ``/tf`` (``tf2_msgs/TFMessage``) together: EgoVehicleStatus
only carries yaw (as ``heading``), with no roll/pitch, while the
``map -> base_link`` transform in ``/tf`` carries the full orientation
quaternion. Cross-checked empirically against a live sample: the yaw
derived from the ``/tf`` quaternion matched ``EgoVehicleStatus.heading``
exactly (86.64 deg both), confirming both describe the same pose in
degrees.

Produces ``network.UDP.protocol.EgoState`` instances so
``dataset/gt_converter.py`` needs zero changes regardless of whether Ego
data came from UDP or ROS.

Known gaps versus the UDP-sourced EgoState (not available on any ROS topic
seen so far): ``size_x/y/z``, ``overhang``, ``wheelbase``,
``rear_overhang``, ``ctrl_mode``, ``gear``, ``signed_vel``. These default
to 0.

Known unit uncertainty: EgoState.vel_*/signed_vel are documented as km/h
on the UDP side; MORAI's ROS EgoVehicleStatus.velocity units have not been
empirically confirmed yet (Ego was stationary during inspection, so
magnitude alone couldn't disambiguate km/h vs m/s). Treated as km/h here
to match EgoState's contract — reverify once the vehicle is actually
moving.

Runs as a native ROS node inside the same machine as roscore (WSL) — no
rosbridge/websocket involved on this side.
"""

import threading
from typing import Callable, Mapping, Optional

import rospy
from morai_msgs.msg import EgoVehicleStatus
from tf2_msgs.msg import TFMessage

from geometry.coordinate_transform import quaternion_to_roll_pitch_yaw_deg
from network.UDP.protocol import EgoState, timestamp_to_ns

from ._rosnode import ensure_node_initialized

DEFAULT_TOPIC = '/Ego_topic'
DEFAULT_TF_TOPIC = '/tf'
DEFAULT_MAP_FRAME = 'map'
DEFAULT_BASE_FRAME = 'base_link'


def _strip_leading_slash(frame_id: str) -> str:
    return frame_id[1:] if frame_id.startswith('/') else frame_id


class RosEgoReceiver:
    """Receive MORAI Ego vehicle state over native ROS subscriptions."""

    def __init__(
        self,
        topic: str = DEFAULT_TOPIC,
        tf_topic: str = DEFAULT_TF_TOPIC,
        map_frame: str = DEFAULT_MAP_FRAME,
        base_frame: str = DEFAULT_BASE_FRAME,
        queue_size: int = 10,
        node_name: Optional[str] = None,
        auto_init_node: bool = True,
    ):
        if not topic:
            raise ValueError('topic must be non-empty')
        self.topic = topic
        self.tf_topic = tf_topic
        self.map_frame = map_frame
        self.base_frame = base_frame
        self.queue_size = int(queue_size)
        self.node_name = node_name or 'morai_alpha_ego'
        self.auto_init_node = bool(auto_init_node)
        self._ego_subscriber = None
        self._tf_subscriber = None
        self._callback: Optional[Callable[[EgoState], None]] = None
        self._lock = threading.Lock()
        self._latest_roll_pitch_deg: Optional[tuple] = None
        self._stats = {
            'messages': 0,
            'tf_messages': 0,
            'tf_unavailable': 0,
            'callback_errors': 0,
        }

    @property
    def stats(self) -> Mapping[str, int]:
        return dict(self._stats)

    def start(self, callback: Callable[[EgoState], None]) -> None:
        if not callable(callback):
            raise TypeError('callback must be callable')
        with self._lock:
            if self._ego_subscriber is not None:
                raise RuntimeError('ego receiver is already running')
            if self.auto_init_node:
                ensure_node_initialized(self.node_name)
            self._callback = callback
            self._tf_subscriber = rospy.Subscriber(
                self.tf_topic, TFMessage, self._on_tf, queue_size=self.queue_size,
            )
            self._ego_subscriber = rospy.Subscriber(
                self.topic, EgoVehicleStatus, self._on_ego, queue_size=self.queue_size,
            )

    def stop(self) -> None:
        with self._lock:
            ego_subscriber = self._ego_subscriber
            tf_subscriber = self._tf_subscriber
            self._ego_subscriber = None
            self._tf_subscriber = None
            self._callback = None
        if ego_subscriber is not None:
            ego_subscriber.unregister()
        if tf_subscriber is not None:
            tf_subscriber.unregister()

    def _on_tf(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if (
                _strip_leading_slash(transform.child_frame_id) != self.base_frame
                or _strip_leading_slash(transform.header.frame_id) != self.map_frame
            ):
                continue
            self._stats['tf_messages'] += 1
            rotation = transform.transform.rotation
            roll, pitch, _yaw = quaternion_to_roll_pitch_yaw_deg(
                rotation.x, rotation.y, rotation.z, rotation.w
            )
            with self._lock:
                self._latest_roll_pitch_deg = (roll, pitch)

    def _on_ego(self, msg: EgoVehicleStatus) -> None:
        callback = self._callback
        if callback is None:
            return
        self._stats['messages'] += 1
        with self._lock:
            roll_pitch = self._latest_roll_pitch_deg
        if roll_pitch is None:
            self._stats['tf_unavailable'] += 1
            roll, pitch = 0.0, 0.0
        else:
            roll, pitch = roll_pitch
        ego_state = EgoState(
            timestamp_ns=timestamp_to_ns(msg.header.stamp.secs, msg.header.stamp.nsecs),
            pos_x=msg.position.x,
            pos_y=msg.position.y,
            pos_z=msg.position.z,
            roll=roll,
            pitch=pitch,
            yaw=msg.heading,
            vel_x=msg.velocity.x,
            vel_y=msg.velocity.y,
            vel_z=msg.velocity.z,
            accel=msg.accel,
            brake=msg.brake,
            front_steer=msg.wheel_angle,
        )
        try:
            callback(ego_state)
        except Exception:
            self._stats['callback_errors'] += 1
