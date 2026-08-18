# Repository audit

This audit follows `MORAI_ALPHA_ROLE_AND_CLEANUP_GUIDE_FINAL.md`. The current
repository role is Dataset generation, not RL/ROACH training.

## Current entry points and data flow

The working data path represented in the repository is:

```text
MORAI Ego/Object/TrafficLight UDP
  -> network/UDP/receiver.py (one thread per stream)
  -> network/UDP/protocol.py data objects
  -> network/UDP/udp_manager.py latest-value slots
  -> run/test_bev_dynamic.py polling loop
  -> morai_gym/.../bev_render.py
  -> optional PNG/NPY BEV output
```

There is no Camera or LiDAR receiver/recorder in the repository. The existing
BEV output is not a training dataset: it has no sensor files, raw GT, source
timestamps, synchronization report, calibration, or label JSON.

Known entry points:

- `run/test_bev_dynamic.py`: UDP-to-BEV diagnostic, currently blocked by a
  missing `config/birdview.yaml` path.
- `run/test_traffic_light.py`: standalone traffic-light UDP diagnostic.
- `run/find_mapping_stopline.py`, `run/find_stopline.py`, `run/linking.py`:
  map inspection scripts that execute at module top level.
- `train_rl.py`: legacy RL entry point; currently has missing imports/configs
  and references a commented-out `server_manager`.
- `network/GRPC/simulation_network.py`: simulator-control example; its local
  generated `proto/` dependency is absent.

## KEEP

- `network/UDP/protocol.py`: Ego/Object wire-level data fields.
- `network/UDP/receiver.py`: MORAI Ego/Object packet parsing.
- `network/UDP/ipconfig.json`: current UDP endpoint configuration.
- `morai_gym/lib/core/birdiview/map/*.json`: retain for GT/map overlays and
  coordinate validation until their competition-map applicability is checked.

The timestamp edits already present in `protocol.py` and `receiver.py` before
this refactor are preserved as user work.

## REFACTOR

- `network/UDP/udp_manager.py`: preserve latest-value compatibility, but add
  packet buffers and atomic snapshots so Dataset code does not pair unrelated
  latest values.
- `morai_gym/lib/core/birdiview/bev_render.py`: split geometry from rendering.
  It assumes yaw zero along world +X and positive counter-clockwise rotation;
  those assumptions are not yet verified against the target MORAI build.
- `morai_gym/lib/core/birdiview/map_to_h5.py`: it currently contains config and
  wrapper classes, not an H5 conversion pipeline, and points at a missing config.
- `run/test_bev_dynamic.py`: keep as a visualization diagnostic after it is
  migrated to the new config/geometry interfaces.
- root/UDP README and package exports: update around the Dataset role.

## LEGACY candidates (do not delete yet)

- `train_rl.py`.
- `agents/rl_birdview/**` and `config/agent/ppo/**`.
- `config/train_rl.yaml` and `run/train_rl.bat`.
- empty `agents/cilrs/cilrs_agent.py`, `agents/roach/lbc_roaming_agent.py`, and
  `utils/server_utils.py`.
- `collected_data/bev_test/**`: 94 PNG/NPY pairs from the old renderer. They may
  be reduced to curated regression fixtures later, but are retained for now.

The RL/PPO files form a separate dependency island: Dataset/UDP/BEV code does
not import them. The legacy entry point imports them and also depends on missing
CARLA/ROACH-era modules and inconsistent Hydra paths.

## UNKNOWN / confirmation required

- `network/UDP/sender.py`: not needed to label a recorded frame, but may be
  needed to drive Ego during automated collection. `UdpManager` imports it.
- `network/GRPC/simulation_network.py`: may be useful for scenario reset and
  collection automation, but cannot run without generated proto modules.
- traffic-light parsing/control and map-to-stopline tools: not part of the
  minimum object GT contract, but may be useful for collection/validation.

## Deletion gate

No candidate should be removed until the new collection entry point works,
packet regression fixtures pass, coordinate overlays are verified, and any
required simulator-driving/reset strategy is selected. Prefer a tag/branch or
an explicit archive commit before physical deletion.
