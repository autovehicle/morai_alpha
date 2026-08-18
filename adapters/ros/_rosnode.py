"""Shared rospy node-initialization guard for the ROS sensor receivers."""

import rospy


def ensure_node_initialized(node_name: str, anonymous: bool = True) -> None:
    """Initialize a rospy node once per process, idempotently.

    Multiple receivers (camera, lidar, ...) may be constructed within the
    same application; only the first call actually needs to register a
    node. ``disable_signals=True`` keeps rospy from installing its own
    SIGINT handler, since this runs as a component inside a larger host
    process rather than as a standalone ``rosrun`` script.
    """
    if rospy.core.is_initialized():
        return
    rospy.init_node(node_name, anonymous=anonymous, disable_signals=True)
