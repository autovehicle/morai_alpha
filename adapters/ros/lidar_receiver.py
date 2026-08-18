"""MORAI LiDAR ROS receiver.

Subscribes to a single rosbridge-fed topic (default ``/lidar3D``,
``sensor_msgs/PointCloud2``) and decodes it with the standard
``sensor_msgs.point_cloud2`` helper rather than hand-parsing the packed
``uint8[] data`` field.

Runs as a native ROS node inside the same machine as roscore (WSL) — no
rosbridge/websocket involved on this side, that bridge exists only for
MORAI (Windows, no ROS) to reach the ROS graph.
"""

import threading
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

from network.UDP.protocol import timestamp_to_ns
from sensors.base import SensorFrame, SensorReceiver

from ._rosnode import ensure_node_initialized

DEFAULT_TOPIC = '/lidar3D'
PREFERRED_FIELD_ORDER = ('x', 'y', 'z', 'intensity')


def _select_field_names(available: Sequence[str]) -> Sequence[str]:
    available_set = set(available)
    selected = [name for name in PREFERRED_FIELD_ORDER if name in available_set]
    if not selected:
        raise ValueError(
            'PointCloud2 message has none of the expected fields {0}; got {1}'.format(
                PREFERRED_FIELD_ORDER, list(available)
            )
        )
    return selected


class RosLidarReceiver(SensorReceiver):
    """Receive MORAI LiDAR point clouds over a native ROS subscription."""

    def __init__(
        self,
        sensor_id: str,
        topic: str = DEFAULT_TOPIC,
        queue_size: int = 1,
        node_name: Optional[str] = None,
        auto_init_node: bool = True,
    ):
        if not sensor_id:
            raise ValueError('sensor_id must be non-empty')
        if not topic:
            raise ValueError('topic must be non-empty')
        self.sensor_id = sensor_id
        self.topic = topic
        self.queue_size = int(queue_size)
        self.node_name = node_name or 'morai_alpha_lidar_{0}'.format(sensor_id)
        self.auto_init_node = bool(auto_init_node)
        self._subscriber = None
        self._callback: Optional[Callable[[SensorFrame], None]] = None
        self._lock = threading.Lock()
        self._stats = {
            'messages': 0,
            'callback_errors': 0,
        }

    @property
    def stats(self) -> Mapping[str, int]:
        return dict(self._stats)

    def start(self, callback: Callable[[SensorFrame], None]) -> None:
        if not callable(callback):
            raise TypeError('callback must be callable')
        with self._lock:
            if self._subscriber is not None:
                raise RuntimeError('lidar receiver is already running')
            if self.auto_init_node:
                ensure_node_initialized(self.node_name)
            self._callback = callback
            self._subscriber = rospy.Subscriber(
                self.topic, PointCloud2, self._on_message,
                queue_size=self.queue_size,
            )

    def stop(self) -> None:
        with self._lock:
            subscriber = self._subscriber
            self._subscriber = None
            self._callback = None
        if subscriber is not None:
            subscriber.unregister()

    def _on_message(self, msg: PointCloud2) -> None:
        callback = self._callback
        if callback is None:
            return
        self._stats['messages'] += 1
        field_names = _select_field_names([f.name for f in msg.fields])
        points = np.fromiter(
            (value for point in point_cloud2.read_points(msg, field_names=field_names, skip_nans=False)
             for value in point),
            dtype=np.float32,
        ).reshape(-1, len(field_names))
        frame = SensorFrame(
            sensor_id=self.sensor_id,
            modality='lidar',
            timestamp_ns=timestamp_to_ns(msg.header.stamp.secs, msg.header.stamp.nsecs),
            payload=points,
            source_frame_id=msg.header.frame_id,
            metadata={
                'field_names': field_names,
                'height': msg.height,
                'width': msg.width,
                'point_step': msg.point_step,
                'is_dense': msg.is_dense,
                'seq': msg.header.seq,
                'topic': self.topic,
            },
        )
        try:
            callback(frame)
        except Exception:
            self._stats['callback_errors'] += 1
