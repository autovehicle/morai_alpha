# MORAI Dataset schema (v0.1, confirmed against a real collection run)

This schema is the interface between `morai_alpha` and the separate TransFuser
repository. It does not reproduce the CARLA Dataset layout.

Confirmed end-to-end against a real collection run (`seq_20260813_231519`,
406 frames, `scripts.validate_dataset` reports `valid=True`, zero issues).
Source: MORAI, connected over ROS (rosbridge on the Windows/MORAI side only;
`morai_alpha` itself is a native ROS node — no rosbridge in its own path).

## Layout

```text
datasets/morai/                          <- collection.output_root in config/dataset.yaml
  schema.json
  sequences/
    <sequence_id>/                       <- e.g. seq_20260813_231519
      sequence.json
      camera/
        camera_1/<frame_id>.jpg          <- 000000.jpg, 000001.jpg, ... (6-digit, zero-padded)
        camera_2/<frame_id>.jpg
        camera_3/<frame_id>.jpg
      lidar/
        lidar3d/<frame_id>.npy
      labels/<frame_id>.json
      metadata/<frame_id>.json
      manifest.jsonl
```

`frame_id` is a recorder-assigned monotonic integer starting at 0 per
sequence, zero-padded to 6 digits in filenames (`collection.frame_digits` in
config).

## Sensors (confirmed)

| Stream | Source topic | Format | Notes |
|---|---|---|---|
| `camera_1`/`camera_2`/`camera_3` | MORAI `/image_jpeg/compressed` (ROS), demuxed by `header.frame_id` (`Camera-1/2/3`) | JPEG, 1280x720 | ~20Hz per camera (MORAI sensor setting), 3 cameras interleaved on one topic |
| `lidar3d` | MORAI `/lidar3D` (ROS) | `.npy`, `numpy.save` of a `(N, 4) float32` array, columns `[x, y, z, intensity]` | ~8-10Hz (VLP16). `(0,0,0,·)` rows are no-return rays, not real points at the origin — filter them out |

Camera and LiDAR point/pixel data is stored exactly as received (no
undistortion, no LiDAR deskew) — `morai_alpha` does not modify sensor payloads,
only synchronizes and labels them.

## Identity and time

- `sequence_id`: stable string for one continuous collection run (default
  auto-generated as `seq_<YYYYMMDD_HHMMSS>`).
- `frame_id`: recorder-assigned monotonic integer within a sequence.
- `timestamp_ns`: signed 64-bit int in the clock selected by
  `synchronization.timestamp_source`. The active ROS configuration uses
  monotonic callback arrival time (`clock_domain: "ros_receive_monotonic"`) because the
  live MORAI ROS bridge was observed publishing sensor-specific header clocks
  that differ by several seconds. Header mode remains available as
  `timestamp_source: "header"` when every sensor is confirmed to share the
  MORAI simulation clock. Camera and LiDAR artifacts retain the original ROS
  header value as `metadata.raw_header_timestamp_ns`.
- `sample_timestamp_ns`: timestamp of the anchor stream. **Anchor is LiDAR**
  (`synchronization.anchor_stream: "lidar"` in config/dataset.yaml) — it's the
  slowest stream (~8-10Hz), so every recorded frame is paced by a LiDAR scan.
- Every accepted frame stores **all 6 streams** (`camera_1`, `camera_2`,
  `camera_3`, `lidar`, `ego`, `objects`) matched within their configured
  tolerance of the anchor (currently 80ms for camera/ego/objects, 0 for the
  anchor itself). A frame is only ever recorded once all 6 streams have a
  matching sample — partial frames are silently dropped by the synchronizer,
  never written with a missing/stale stream.

## `labels/<frame_id>.json` — the training target

```json
{
  "schema_version": "0.1.0-draft",
  "sequence_id": "seq_20260813_231519",
  "frame_id": 200,
  "sample_timestamp_ns": 1786630537895000064,
  "coordinate_frame": "ego_forward_left_up",
  "objects": [
    {
      "obj_id": 13,
      "class_id": 2,
      "class_name": "obstacle",
      "position_ego_m": [14.02, 0.17, 0.06],
      "size_lwh_m": [4.69, 2.20, 1.60],
      "yaw_ego_rad": -0.0022
    }
  ]
}
```

- **Coordinate frame**: ego-relative, **x = forward, y = left, z = up**
  (metres). Empirically confirmed (not just assumed) against real MORAI
  motion/object placement.
- `size_lwh_m`: `[length, width, height]` — confirmed against a real vehicle
  NPC (`4.69 x 2.20 x 1.60`, matches typical sedan dimensions).
- `yaw_ego_rad`: object heading relative to Ego yaw, **radians, positive
  counter-clockwise**, 0 = same heading as Ego.
- `class_id`/`class_name`: `0`=`pedestrian`, `1`=`vehicle`, `2`=`obstacle`
  (`class_mapping` in `config/dataset.yaml`; MORAI's `npc_list`/
  `pedestrian_list`/`obstacle_list` map to `1`/`0`/`2` respectively).
- `z` reference point: **not** ground/tire-contact and **not** full-body
  centre — empirically ~0.32-0.35m above the ground plane (see
  the LiDAR-derived ground-plane check). Only matters
  if you need precise 3D box height; irrelevant for 2D/BEV.
- Frame-to-frame `obj_id` continuity is **not guaranteed** — it's MORAI's raw
  object id, not a verified persistent track id.

## `metadata/<frame_id>.json` — raw GT + sync record

Same frame as above (abbreviated):

```json
{
  "schema_version": "0.1.0-draft",
  "sequence_id": "seq_20260813_231519",
  "frame_id": 200,
  "sample_timestamp_ns": 1786630537895000064,
  "anchor_stream": "lidar",
  "coordinate_convention": "morai_world_x_forward_ccw__PROVISIONAL",
  "ego_raw": {
    "timestamp_ns": 1786630537894000128,
    "world_position_m": [-96.02, -256.37, 28.77],
    "roll_pitch_yaw_deg": [0.229, -0.235, 60.627],
    "velocity_kmh": [5.4e-06, 4.0e-05, 2.7e-05],
    "signed_velocity_kmh": 0.0,
    "size_m": [0.0, 0.0, 0.0]
  },
  "objects_raw": [
    {
      "obj_id": 13, "obj_type": 2,
      "world_position_m": [-89.29, -244.07, 28.83],
      "heading_deg": 60.50,
      "size_xyz_m": [4.69, 2.20, 1.60],
      "velocity_kmh": [0.0, 0.0, 0.0],
      "acceleration_mps2": [0.0, 0.0, 0.0],
      "timestamp_ns": 1786630537894000128
    }
  ],
  "sensor_artifacts": [
    {
      "modality": "camera", "sensor_id": "camera_1",
      "path": "sequences/.../camera/camera_1/000200.jpg",
      "timestamp_ns": 1786630537892999936, "source_frame_id": "Camera-1",
      "size_bytes": 155832, "sha256": "783ab4a9...",
      "clock_domain": "ros_receive_monotonic"
    }
  ],
  "synchronization": {
    "camera_1": {"timestamp_ns": 1786630537892999936, "offset_ns": -2000128, "source_frame_id": "Camera-1", "clock_domain": "ros_receive_monotonic"},
    "lidar":    {"timestamp_ns": 1786630537895000064, "offset_ns": 0,        "source_frame_id": "Lidar3D-4", "clock_domain": "ros_receive_monotonic"},
    "ego":      {"timestamp_ns": 1786630537894000128, "offset_ns": -999936,  "source_frame_id": null,         "clock_domain": "ros_receive_monotonic"}
  }
}
```

Raw fields keep MORAI's native units (`degree` for angles, `km/h` for
velocity, `m/s^2` for acceleration) — see `schema.json`'s unit declarations
below. Raw velocity/acceleration are **not currently used as training
targets**; they're preserved for future temporal-model work
(`t-1`/`t` fusion) so the raw data never needs re-collecting.

`ego_raw.size_m` is always `[0,0,0]` — MORAI's ROS Ego topic doesn't expose
vehicle dimensions (a real UDP-sourced Ego would have this; the ROS path
doesn't). `ego_raw.velocity_kmh` units are the field's documented contract
but have not been empirically confirmed on the ROS path yet (Ego was
stationary during verification) — treat as provisional until confirmed with
the vehicle actually moving.

`sensor_artifacts[].sha256`/`size_bytes` let you verify a camera/lidar file
wasn't corrupted/truncated without re-reading the whole dataset.

## `schema.json` (dataset root, written once)

```json
{
  "schema_version": "0.1.0-draft",
  "target_coordinate_frame": "ego_forward_left_up",
  "time_unit": "nanosecond",
  "position_unit": "metre",
  "target_angle_unit": "radian",
  "raw_angle_unit": "degree",
  "raw_velocity_unit": "kilometre_per_hour",
  "obj_id_semantics": "source_object_id_not_verified_track_id"
}
```

## `sequence.json` (per sequence, written once)

```json
{
  "schema_version": "0.1.0-draft",
  "sequence_id": "seq_20260813_231519",
  "coordinate_convention": "morai_world_x_forward_ccw__PROVISIONAL",
  "calibration": null,
  "calibration_status": "TODO_sensor_interfaces_not_confirmed"
}
```

`calibration` is `null` for now — per-camera intrinsics/extrinsics aren't
written into the dataset yet (see below). The coordinate convention name
keeps its `__PROVISIONAL` suffix for now even though it's been empirically
verified; the underlying
`config/dataset.yaml`'s `coordinate_convention.verified` flag is `true`.

## `manifest.jsonl`

One JSON object per line, one line per recorded frame, append-only:

```json
{"sequence_id": "seq_20260813_231519", "frame_id": 0, "sample_timestamp_ns": 1786630516566000128, "labels": "sequences/seq_20260813_231519/labels/000000.json", "metadata": "sequences/seq_20260813_231519/metadata/000000.json"}
```

Use this to enumerate frames without listing the `labels/`/`metadata/`
directories directly.

## Camera calibration (confirmed for Camera-1; 2/3 not yet measured)

Not yet written into `sequence.json.calibration`, but confirmed by direct
LiDAR-to-image reprojection (points landed exactly on real road
edges/poles in the actual camera frame):

- **Intrinsics** (pinhole, no distortion modeled): `width=1280, height=720,
  horizontal_fov=90deg` → `fx=fy=640, cx=640, cy=360`.
- **Extrinsic** (Camera-1 relative to Ego/`base_link`, from MORAI's sensor
  transform / ROS `/tf`): translation `(x=1.90, y=0.00, z=1.20)` m, rotation
  = pure **+2deg pitch** (quaternion `x=0, y=0.017452, z=0, w=0.999848`), in
  the same `ego forward/left/up` body frame as everything else — **not**
  the OpenCV optical convention (right/down/forward); convert before
  projecting (see `dataset/collector.py`'s reprojection helper logic if
  reimplementing this in TransFuser).
- Camera-2/Camera-3 have different mount rotations (~60deg yaw-ish offsets)
  and haven't had their resolution/FOV/extrinsic independently confirmed
  yet — don't assume they match Camera-1.
- LiDAR (`Lidar3D-4`) extrinsic: translation `(1.092, -0.019, 1.233)` m,
  **identity rotation** relative to `base_link` (LiDAR's local x/y/z already
  equal Ego's forward/left/up, no rotation needed).

## Known open items (not blocking; lower priority)

1. Camera-2/Camera-3 intrinsics/extrinsics not independently confirmed.
2. `ego_raw.velocity_kmh` / `objects_raw[].velocity_kmh` units not confirmed
   moving (see above) — not used as a training target yet, so not urgent.
3. BEV ROI/resolution convention for any BEV-rasterized representation.
4. Calibration is not yet serialized into `sequence.json.calibration` (data
   above is accurate but lives in this doc, not in the dataset files
   themselves).
