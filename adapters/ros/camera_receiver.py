"""MORAI Camera ROS receiver.

Subscribes to a single rosbridge-fed topic (default ``/image_jpeg/compressed``,
``sensor_msgs/CompressedImage``). MORAI publishes all connected cameras onto
this one topic, distinguished only by ``header.frame_id`` (observed values:
``Camera-1``, ``Camera-2``, ``Camera-3`` — confirmed via ``rostopic echo``),
so this receiver does not attempt to split cameras into separate topics
itself; it forwards ``frame_id`` through as ``SensorFrame.source_frame_id``
and leaves demuxing to the caller.

Runs as a native ROS node inside the same machine as roscore (WSL) — no
rosbridge/websocket involved on this side, that bridge exists only for
MORAI (Windows, no ROS) to reach the ROS graph.
"""

import threading
import time
from typing import Callable, Mapping, Optional

import rospy
from sensor_msgs.msg import CompressedImage

from sensors.base import SensorFrame, SensorReceiver

from ._rosnode import ensure_node_initialized
from ._timestamp import (
    clock_domain_for,
    resolve_timestamp_ns,
    validate_timestamp_source,
)

DEFAULT_TOPIC = '/image_jpeg/compressed'


class RosCameraReceiver(SensorReceiver):
    """Receive MORAI camera frames over a native ROS subscription."""

    def __init__(
        self,
        sensor_id: str,
        topic: str = DEFAULT_TOPIC,
        queue_size: int = 10,
        timestamp_source: str = 'header',
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
        self.timestamp_source = validate_timestamp_source(timestamp_source)
        self.clock_domain = clock_domain_for(self.timestamp_source)
        self.node_name = node_name or 'morai_alpha_camera_{0}'.format(sensor_id)
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
                raise RuntimeError('camera receiver is already running')
            if self.auto_init_node:
                ensure_node_initialized(self.node_name)
            self._callback = callback
            self._subscriber = rospy.Subscriber(
                self.topic, CompressedImage, self._on_message,
                queue_size=self.queue_size,
            )

    def stop(self) -> None:
        with self._lock:
            subscriber = self._subscriber
            self._subscriber = None
            self._callback = None
        if subscriber is not None:
            subscriber.unregister()

    def _on_message(self, msg: CompressedImage) -> None:
        received_monotonic_ns = time.monotonic_ns()
        callback = self._callback
        if callback is None:
            return
        self._stats['messages'] += 1
        selected_timestamp_ns, raw_header_timestamp_ns = resolve_timestamp_ns(
            msg.header.stamp, self.timestamp_source, received_monotonic_ns
        )
        frame = SensorFrame(
            sensor_id=self.sensor_id,
            modality='camera',
            timestamp_ns=selected_timestamp_ns,
            payload=bytes(msg.data),
            source_frame_id=msg.header.frame_id,
            clock_domain=self.clock_domain,
            metadata={
                'encoding': msg.format,
                'seq': msg.header.seq,
                'topic': self.topic,
                'timestamp_source': self.timestamp_source,
                'received_monotonic_ns': received_monotonic_ns,
                'raw_header_timestamp_ns': raw_header_timestamp_ns,
            },
        )
        try:
            callback(frame)
        except Exception:
            self._stats['callback_errors'] += 1
