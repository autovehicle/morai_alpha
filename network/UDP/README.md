# MORAI UDP layer

This package is the reusable communication boundary for Dataset collection.

```text
MORAI
  -> Camera UDP -> sensors.camera_receiver timestamped JPEG frames
  -> VLP16 UDP -> official Velodyne local decoder
  -> Ego/Object UDP -> receiver.py -> protocol.py objects
  -> timestamped bounded buffers
  -> dataset.frame_sync.FrameSynchronizer
```

## Receive path

- `protocol.py` defines wire-facing Ego/Object/TrafficLight fields.
- `receiver.py` parses MORAI binary packets on background threads.
- `ObjectFrame` remains list-compatible but carries a packet-level
  `timestamp_ns`, including for valid frames containing zero objects.
- `UdpManager.ego_state` and `object_list` preserve the old latest-value API for
  diagnostics.
- Dataset code must drain `ego_states`/`object_frames` and synchronize their
  timestamps; it must not record `snapshot_latest()` as if it were synchronized.
- `buffer_stats` exposes bounded-buffer drops. A collection run with drops
  should be marked invalid or repeated.

The packet timestamp fields are float32 at the source. Internally they are
combined into an int64-compatible Python `int` named `timestamp_ns`; this does
not imply true nanosecond source precision.

## Optional send path

`sender.py` and `UdpManager.send_ctrl()`/`force_green()` remain for possible
collection-driving or scenario-control use. They are not part of the Dataset
label contract and should eventually be separated from a receive-only manager
after the simulator automation strategy is confirmed.

## Current limitations

- Camera UDP is implemented in `sensors/camera_receiver.py`, because it is a
  sensor adapter rather than an Ego/Object network message.
- LiDAR remains native VLP16 UDP at the simulator boundary and is decoded
  locally by `velodyne_pointcloud`. It does not use rosbridge.
- TrafficLight packets are not part of the minimum object Dataset schema.
- MORAI packet fixtures from the competition build are still required for
  protocol regression tests and timestamp precision validation.
