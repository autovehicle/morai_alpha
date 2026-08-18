"""Structural and semantic checks for a recorded MORAI Dataset."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .schema import SCHEMA_VERSION


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    path: Optional[str] = None


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)
    frame_count: int = 0

    @property
    def is_valid(self) -> bool:
        return not any(issue.level == 'error' for issue in self.issues)

    def add(self, level: str, code: str, message: str,
            path: Optional[Path] = None) -> None:
        self.issues.append(ValidationIssue(
            level=level,
            code=code,
            message=message,
            path=str(path) if path is not None else None,
        ))


class DatasetValidator:
    def __init__(self, root: Path, max_sync_offset_ns: Optional[int] = None):
        self.root = Path(root)
        self.max_sync_offset_ns = max_sync_offset_ns

    @staticmethod
    def _load_json(path: Path):
        with path.open('r', encoding='utf-8') as stream:
            return json.load(stream)

    @staticmethod
    def _finite_vector(value, expected_length: int) -> bool:
        return (
            isinstance(value, list)
            and len(value) == expected_length
            and all(isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(item) for item in value)
        )

    def validate(self, verify_checksums: bool = True) -> ValidationReport:
        report = ValidationReport()
        schema_path = self.root / 'schema.json'
        if not schema_path.exists():
            report.add('error', 'missing_schema', 'schema.json is missing', schema_path)
            return report
        try:
            schema = self._load_json(schema_path)
        except (OSError, ValueError) as error:
            report.add('error', 'invalid_schema_json', str(error), schema_path)
            return report
        if schema.get('schema_version') != SCHEMA_VERSION:
            report.add('error', 'schema_version', 'unsupported schema version', schema_path)

        sequence_root = self.root / 'sequences'
        if not sequence_root.exists():
            report.add('error', 'missing_sequences', 'sequences directory is missing', sequence_root)
            return report

        for sequence_dir in sorted(path for path in sequence_root.iterdir()
                                   if path.is_dir()):
            metadata_dir = sequence_dir / 'metadata'
            if not metadata_dir.exists():
                report.add('warning', 'missing_metadata_dir',
                           'sequence has no metadata directory', metadata_dir)
                continue
            for metadata_path in sorted(metadata_dir.glob('*.json')):
                report.frame_count += 1
                self._validate_frame(sequence_dir, metadata_path, report,
                                     verify_checksums)
        return report

    def _validate_frame(self, sequence_dir: Path, metadata_path: Path,
                        report: ValidationReport, verify_checksums: bool) -> None:
        try:
            metadata = self._load_json(metadata_path)
        except (OSError, ValueError) as error:
            report.add('error', 'invalid_metadata_json', str(error), metadata_path)
            return

        frame_id = metadata.get('frame_id')
        timestamp_ns = metadata.get('sample_timestamp_ns')
        if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
            report.add('error', 'invalid_frame_id', 'frame_id must be non-negative int', metadata_path)
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
            report.add('error', 'invalid_timestamp', 'sample_timestamp_ns must be int', metadata_path)

        label_path = sequence_dir / 'labels' / (metadata_path.stem + '.json')
        if not label_path.exists():
            report.add('error', 'missing_labels', 'matching label file is missing', label_path)
        else:
            self._validate_labels(label_path, metadata, report)

        synchronization = metadata.get('synchronization', {})
        if not isinstance(synchronization, dict):
            report.add('error', 'invalid_sync', 'synchronization must be an object', metadata_path)
        elif isinstance(timestamp_ns, int):
            for stream, source in synchronization.items():
                if not isinstance(source, dict):
                    report.add('error', 'invalid_sync_source', stream, metadata_path)
                    continue
                source_timestamp = source.get('timestamp_ns')
                offset = source.get('offset_ns')
                if not isinstance(source_timestamp, int) or not isinstance(offset, int):
                    report.add('error', 'invalid_sync_time', stream, metadata_path)
                    continue
                if source_timestamp - timestamp_ns != offset:
                    report.add('error', 'sync_offset_mismatch', stream, metadata_path)
                if (self.max_sync_offset_ns is not None
                        and abs(offset) > self.max_sync_offset_ns):
                    report.add('error', 'sync_tolerance', stream, metadata_path)

        for artifact in metadata.get('sensor_artifacts', []):
            relative_path = artifact.get('path') if isinstance(artifact, dict) else None
            if not isinstance(relative_path, str):
                report.add('error', 'invalid_artifact', 'artifact path is missing', metadata_path)
                continue
            artifact_path = self.root / relative_path
            if not artifact_path.is_file():
                report.add('error', 'missing_artifact', 'sensor artifact is missing', artifact_path)
                continue
            if artifact_path.stat().st_size != artifact.get('size_bytes'):
                report.add('error', 'artifact_size', 'sensor artifact size mismatch', artifact_path)
            if verify_checksums:
                checksum = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                if checksum != artifact.get('sha256'):
                    report.add('error', 'artifact_checksum',
                               'sensor artifact checksum mismatch', artifact_path)

    def _validate_labels(self, label_path: Path, metadata,
                         report: ValidationReport) -> None:
        try:
            labels = self._load_json(label_path)
        except (OSError, ValueError) as error:
            report.add('error', 'invalid_label_json', str(error), label_path)
            return
        if labels.get('frame_id') != metadata.get('frame_id'):
            report.add('error', 'label_frame_id', 'label/metadata frame mismatch', label_path)
        if labels.get('sample_timestamp_ns') != metadata.get('sample_timestamp_ns'):
            report.add('error', 'label_timestamp', 'label/metadata timestamp mismatch', label_path)

        seen_obj_ids = set()
        for label in labels.get('objects', []):
            if not isinstance(label, dict):
                report.add('error', 'invalid_label', 'object label must be an object', label_path)
                continue
            obj_id = label.get('obj_id')
            if not isinstance(obj_id, int) or isinstance(obj_id, bool):
                report.add('error', 'invalid_obj_id', 'obj_id must be int', label_path)
            elif obj_id in seen_obj_ids:
                report.add('error', 'duplicate_obj_id',
                           'duplicate obj_id in one frame: {0}'.format(obj_id), label_path)
            seen_obj_ids.add(obj_id)
            if not self._finite_vector(label.get('position_ego_m'), 3):
                report.add('error', 'invalid_position', 'position_ego_m must be finite xyz', label_path)
            if not self._finite_vector(label.get('size_lwh_m'), 3):
                report.add('error', 'invalid_size', 'size_lwh_m must be finite lwh', label_path)
            yaw = label.get('yaw_ego_rad')
            if not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
                report.add('error', 'invalid_yaw', 'yaw_ego_rad must be finite', label_path)

    def audit_obj_id_continuity(self) -> Dict[str, Dict[int, Dict[str, object]]]:
        """Report observations only; never promote obj_id to track semantics."""
        observations = {}
        sequence_root = self.root / 'sequences'
        if not sequence_root.exists():
            return observations
        for sequence_dir in sorted(path for path in sequence_root.iterdir()
                                   if path.is_dir()):
            sequence_observations = observations.setdefault(sequence_dir.name, {})
            metadata_dir = sequence_dir / 'metadata'
            for path in sorted(metadata_dir.glob('*.json')) if metadata_dir.exists() else ():
                metadata = self._load_json(path)
                frame_id = metadata.get('frame_id')
                for obj in metadata.get('objects_raw', []):
                    obj_id = obj.get('obj_id')
                    if not isinstance(obj_id, int):
                        continue
                    record = sequence_observations.setdefault(obj_id, {
                        'frames': [], 'obj_types': set(),
                    })
                    record['frames'].append(frame_id)
                    record['obj_types'].add(obj.get('obj_type'))
        for sequence_observations in observations.values():
            for record in sequence_observations.values():
                frames = sorted(
                    item for item in record['frames'] if isinstance(item, int)
                )
                record['frames'] = frames
                record['gap_count'] = sum(
                    1 for previous, current in zip(frames, frames[1:])
                    if current != previous + 1
                )
                record['obj_types'] = sorted(record['obj_types'], key=str)
                record['type_changed'] = len(record['obj_types']) > 1
        return observations
