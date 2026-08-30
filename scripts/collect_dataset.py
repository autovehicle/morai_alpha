"""Run DatasetCollector until interrupted.

Requires a ROS-sourced WSL shell (rospy/sensor_msgs/morai_msgs on
PYTHONPATH) and a live MORAI connection publishing the camera/lidar/ego/
object topics this collector subscribes to.

Real collection (writing labels that matter) requires the MORAI
coordinate convention to be verified first — see
docs/DATASET_SCHEMA.md and config/dataset.yaml's
``coordinate_convention.verified`` flag. Until then, only
--no-strict-convention smoke tests are meaningful, and those must not
point at the real --output-root.
"""

import argparse
import time

from dataset.collector import DatasetCollector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None, help='Path to dataset.yaml (default: config/dataset.yaml)')
    parser.add_argument('--sequence-id', default=None, help='Override collection.sequence_id')
    parser.add_argument('--output-root', default=None, help='Override collection.output_root')
    parser.add_argument(
        '--no-strict-convention', action='store_true',
        help='Bypass the coordinate-verification gate. Smoke-test use only '
             '— do not point --output-root at real collection output.',
    )
    args = parser.parse_args()

    collector = DatasetCollector(
        config_path=args.config,
        sequence_id=args.sequence_id,
        output_root=args.output_root,
        strict_convention=not args.no_strict_convention,
    )

    print('sequence_id:', collector.sequence_id)
    collector.start()
    print('collecting — Ctrl-C to stop')
    try:
        while True:
            time.sleep(2.0)
            print(collector.stats)
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop()
        print('final stats:', collector.stats)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
