"""Validate a Dataset directory produced by morai_alpha."""

import argparse
from pathlib import Path

from dataset.validator import DatasetValidator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--max-sync-offset-ns', type=int, default=None)
    parser.add_argument('--skip-checksums', action='store_true')
    args = parser.parse_args()

    report = DatasetValidator(
        args.dataset_root,
        max_sync_offset_ns=args.max_sync_offset_ns,
    ).validate(verify_checksums=not args.skip_checksums)
    print('frames={0} valid={1}'.format(report.frame_count, report.is_valid))
    for issue in report.issues:
        location = ' [{0}]'.format(issue.path) if issue.path else ''
        print('{0} {1}: {2}{3}'.format(
            issue.level.upper(), issue.code, issue.message, location
        ))
    return 0 if report.is_valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
