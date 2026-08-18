# MORAI UDP collection setup

This is the selected acquisition path for both training-data collection and
competition runtime. It removes rosbridge/WebSocket from the data path.

## Data paths

```text
Camera  : MORAI UDP JPEG -> CameraReceiver -> timestamped SensorFrame
LiDAR   : MORAI VLP16 UDP -> velodyne_pointcloud -> /velodyne_points
Ego/Obj : MORAI UDP -> UdpManager timestamped buffers
```

The Velodyne ROS node is only a local VLP16 packet decoder. MORAI is still
configured as UDP, and `rosbridge_websocket` must not be launched.

## 1. Resolve the current WSL address

Run in WSL whenever WSL or Windows has restarted:

```bash
hostname -I
```

Use the WSL address (currently observed as `172.19.0.55`) as every MORAI
sensor's **Destination IP**. Do not use `127.0.0.1`: MORAI runs on Windows,
so that address targets Windows loopback rather than the WSL UDP receiver.

## 2. Camera

For the fixed front camera:

- Sensor Network: UDP
- Destination IP: current WSL IP
- Destination Port: `9291` (or the value configured in
  `config/dataset.yaml`)
- resolution/FOV: preserve the competition-fixed values

The MORAI Host Sensor IP/Port are sender-side values. Preserve the
competition-provided values unless MORAI refuses to connect.

Smoke test from the repository root:

```bash
python3 -m scripts.check_camera_udp --port 9291 \
  --save-first /tmp/morai_camera_front.jpg
```

A valid result prints a nonzero `timestamp_ns`, JPEG byte count, and packet
count. Any nonzero `malformed_datagrams`, `incomplete_frames`, or
`invalid_jpeg_frames` must be investigated before collection.

## 3. VLP16 LiDAR

Use the settings already verified on this machine:

- Model: VLP16
- Intensity Type: Intensity
- Rotation Rate: 10 Hz
- Sensor Network: UDP
- Host Sensor IP: Windows WSL-adapter IP (observed `172.19.0.1`)
- Host Sensor Port: `2369`
- Destination IP: current WSL IP
- Destination Port: `2368`

Start the local decoder:

```bash
source /opt/ros/noetic/setup.bash
roslaunch velodyne_pointcloud VLP16_points.launch port:=2368 rpm:=600
```

Verify in another sourced terminal:

```bash
rostopic hz /velodyne_points
rostopic echo -n 1 /velodyne_points/header
```

The prior smoke test produced roughly 8.7 Hz and a populated header. The
measured rate must be recorded again during a real collection run.

## 4. Ego Status and Object Info

In **Ego Network -> Publisher, Subscriber, Service**, select UDP and set:

- Destination IP: current WSL IP
- Competition Vehicle Status destination port: `909`
- Object Info destination port: `7505`, if Object Info is available in the
  competition build/license

`network/UDP/ipconfig.json` binds receivers to `0.0.0.0`, so packets sent
to the current WSL interface can be received. Host/source ports must match the
corresponding MORAI fields; do not assume the example defaults if the
competition configuration fixes them.

## Remaining validation gates

1. Capture one real Camera UDP packet and confirm timestamp integer encoding,
   index base, and chunk-size semantics against the parser.
2. Confirm Camera and VLP16 timestamps share the MORAI simulation clock.
3. Confirm whether privileged Object Info is exposed by the competition
   build/license during offline Dataset collection.
4. Record camera/LiDAR extrinsics and intrinsics; do not enable final GT
   conversion until coordinate overlays are verified.
