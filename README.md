# morai_alpha

`morai_alpha` is the MORAI data acquisition repository for the camera-LiDAR
object-perception project. Its responsibility is limited to:

- receiving Camera, LiDAR, Ego Status, and Object Info data from MORAI;
- preserving source timestamps and raw privileged ground truth;
- synchronizing independently arriving streams;
- converting world-frame object ground truth into a verified ego/BEV frame;
- recording and validating a versioned dataset for the separate TransFuser repository.

This repository does **not** train TransFuser, PPO/RL, imitation-learning,
waypoint, PID, temporal BEV, motion-head, or relative-velocity models.

## Current status

MORAI-to-collector transport is UDP for Camera, LiDAR, Ego Status, and Object
Info. Camera UDP JPEG reassembly is implemented with source timestamp and
packet-loss accounting. VLP16 UDP reception has been verified through the
official Velodyne driver; this local decoder does not use rosbridge/WebSocket.
The coordinate convention remains intentionally unverified until it is checked
in the competition MORAI environment.

See `docs/DATASET_SCHEMA.md` for the repository-to-TransFuser Dataset
contract, sensor conventions, synchronization policy, and remaining
calibration items.

The existing RL/ROACH files remain in place as legacy candidates. No legacy
code or collected artifact has been deleted during the first refactoring pass.
