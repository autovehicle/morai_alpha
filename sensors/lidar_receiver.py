"""LiDAR receiver interface placeholder; no transport is assumed."""

from .base import SensorReceiver


class LidarReceiver(SensorReceiver):
    """Marker interface for future MORAI LiDAR adapters."""

    pass
