"""MORAI Object (privileged NPC/pedestrian/obstacle GT) ROS receiver.

Subscribes to ``/Object_topic`` (``morai_msgs/ObjectStatusList``, ~45Hz
observed), which carries three separate arrays (``npc_list``,
``pedestrian_list``, ``obstacle_list``) rather than a single flat list.
Object category is taken from *which array* an object appears in — not
from the message's own ``type`` field, whose numbering has not been
cross-checked against ``network.UDP.protocol``'s
``OBJ_TYPE_PEDESTRIAN/VEHICLE/OBSTACLE`` convention — so the three arrays
are mapped directly to those constants.

Produces ``network.UDP.protocol.ObjectFrame`` (a list of ``ObjectData``
carrying the packet-level timestamp) so ``dataset/gt_converter.py`` needs
zero changes regardless of whether Object GT came from UDP or ROS.

Known gap versus the UDP-sourced ObjectData: ``overhang``, ``wheelbase``,
``rear_overhang`` are not present on ``morai_msgs/ObjectStatus`` and
default to 0 — ``dataset/gt_converter.py`` does not read them, so this is
not a blocker for label conversion.

Runs as a native ROS node inside the same machine as roscore (WSL) — no
rosbridge/websocket involved on this side.
"""

import threading
from typing import Callable, Mapping, Optional

import rospy
from morai_msgs.msg import ObjectStatusList

from network.UDP.protocol import (
    OBJ_TYPE_OBSTACLE,
    OBJ_TYPE_PEDESTRIAN,
    OBJ_TYPE_VEHICLE,
    ObjectData,
    ObjectFrame,
    timestamp_to_ns,
)

from ._rosnode import ensure_node_initialized

DEFAULT_TOPIC = '/Object_topic'


def _object_data(obj, obj_type: int, timestamp_ns: int) -> ObjectData:
    return ObjectData(
        timestamp_ns=timestamp_ns,
        obj_id=obj.unique_id,
        obj_type=obj_type,
        pos_x=obj.position.x,
        pos_y=obj.position.y,
        pos_z=obj.position.z,
        heading=obj.heading,
        size_x=obj.size.x,
        size_y=obj.size.y,
        size_z=obj.size.z,
        vel_x=obj.velocity.x,
        vel_y=obj.velocity.y,
        vel_z=obj.velocity.z,
        acc_x=obj.acceleration.x,
        acc_y=obj.acceleration.y,
        acc_z=obj.acceleration.z,
    )


class RosGtReceiver:
    """Receive MORAI privileged Object GT over a native ROS subscription."""

    def __init__(
        self,
        topic: str = DEFAULT_TOPIC,
        queue_size: int = 1,
        node_name: Optional[str] = None,
        auto_init_node: bool = True,
    ):
        if not topic:
            raise ValueError('topic must be non-empty')
        self.topic = topic
        self.queue_size = int(queue_size)
        self.node_name = node_name or 'morai_alpha_gt'
        self.auto_init_node = bool(auto_init_node)
        self._subscriber = None
        self._callback: Optional[Callable[[ObjectFrame], None]] = None
        self._lock = threading.Lock()
        self._stats = {
            'messages': 0,
            'objects': 0,
            'callback_errors': 0,
        }

    @property
    def stats(self) -> Mapping[str, int]:
        return dict(self._stats)

    def start(self, callback: Callable[[ObjectFrame], None]) -> None:
        if not callable(callback):
            raise TypeError('callback must be callable')
        with self._lock:
            if self._subscriber is not None:
                raise RuntimeError('gt receiver is already running')
            if self.auto_init_node:
                ensure_node_initialized(self.node_name)
            self._callback = callback
            self._subscriber = rospy.Subscriber(
                self.topic, ObjectStatusList, self._on_message,
                queue_size=self.queue_size,
            )

    def stop(self) -> None:
        with self._lock:
            subscriber = self._subscriber
            self._subscriber = None
            self._callback = None
        if subscriber is not None:
            subscriber.unregister()

    def _on_message(self, msg: ObjectStatusList) -> None:
        callback = self._callback
        if callback is None:
            return
        self._stats['messages'] += 1
        timestamp_ns = timestamp_to_ns(msg.header.stamp.secs, msg.header.stamp.nsecs)
        objects = (
            [_object_data(obj, OBJ_TYPE_VEHICLE, timestamp_ns) for obj in msg.npc_list]
            + [_object_data(obj, OBJ_TYPE_PEDESTRIAN, timestamp_ns) for obj in msg.pedestrian_list]
            + [_object_data(obj, OBJ_TYPE_OBSTACLE, timestamp_ns) for obj in msg.obstacle_list]
        )
        self._stats['objects'] += len(objects)
        frame = ObjectFrame(objects, timestamp_ns=timestamp_ns)
        try:
            callback(frame)
        except Exception:
            self._stats['callback_errors'] += 1
