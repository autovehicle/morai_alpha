"""Shared timestamp handling for ROS receivers."""

import time

from network.UDP.protocol import timestamp_to_ns


VALID_TIMESTAMP_SOURCES = ('header', 'arrival')


def validate_timestamp_source(timestamp_source: str) -> str:
    source = str(timestamp_source)
    if source not in VALID_TIMESTAMP_SOURCES:
        raise ValueError(
            'timestamp_source must be one of {0}; got {1!r}'.format(
                VALID_TIMESTAMP_SOURCES, source
            )
        )
    return source


def resolve_timestamp_ns(stamp, timestamp_source: str, received_monotonic_ns=None):
    """Return ``(selected_ns, header_ns)`` for a ROS message."""
    header_ns = timestamp_to_ns(stamp.secs, stamp.nsecs)
    if timestamp_source == 'header':
        return header_ns, header_ns
    if received_monotonic_ns is None:
        received_monotonic_ns = time.monotonic_ns()
    return int(received_monotonic_ns), header_ns


def clock_domain_for(timestamp_source: str) -> str:
    if timestamp_source == 'arrival':
        return 'ros_receive_monotonic'
    return 'morai_simulation'
