"""Atomic recorder for the draft MORAI Dataset layout."""

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .schema import (
    FrameMetadata,
    ObjectLabel,
    SCHEMA_VERSION,
    TARGET_COORDINATE_FRAME,
    SourceFrameId,
    validate_timestamp_ns,
)


_SAFE_COMPONENT = re.compile(r'^[A-Za-z0-9._-]+$')


@dataclass(frozen=True)
class SensorArtifact:
    """An encoded sensor payload supplied by a future transport adapter."""

    modality: str
    sensor_id: str
    extension: str
    content: bytes
    timestamp_ns: int
    source_frame_id: Optional[SourceFrameId] = None
    clock_domain: str = 'morai_simulation'


class DatasetRecorder:
    """Write complete synchronized samples without assuming sensor codecs."""

    def __init__(self, root: Path, frame_digits: int = 6):
        self.root = Path(root)
        if frame_digits <= 0:
            raise ValueError('frame_digits must be positive')
        self.frame_digits = int(frame_digits)
        self._manifest_lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_schema_once()

    def _write_schema_once(self) -> None:
        path = self.root / 'schema.json'
        if path.exists():
            with path.open('r', encoding='utf-8') as stream:
                existing = json.load(stream)
            if existing.get('schema_version') != SCHEMA_VERSION:
                raise ValueError(
                    'existing Dataset schema version does not match recorder'
                )
            return
        self._atomic_json(path, {
            'schema_version': SCHEMA_VERSION,
            'target_coordinate_frame': TARGET_COORDINATE_FRAME,
            'time_unit': 'nanosecond',
            'position_unit': 'metre',
            'target_angle_unit': 'radian',
            'raw_angle_unit': 'degree',
            'raw_velocity_unit': 'kilometre_per_hour',
            'obj_id_semantics': 'source_object_id_not_verified_track_id',
        })

    @staticmethod
    def _safe_component(value: str, field_name: str) -> str:
        value = str(value)
        if not value or not _SAFE_COMPONENT.fullmatch(value):
            raise ValueError('{0} contains unsafe path characters'.format(field_name))
        return value

    @staticmethod
    def _atomic_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb', dir=str(path.parent), delete=False
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temporary_path), str(path))
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def _atomic_json(cls, path: Path, value) -> None:
        content = (json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False,
        ) + '\n').encode('utf-8')
        cls._atomic_bytes(path, content)

    def record_frame(
        self,
        metadata: FrameMetadata,
        labels: Iterable[ObjectLabel],
        artifacts: Iterable[SensorArtifact],
    ) -> Tuple[Path, Path]:
        """Record one already synchronized frame and append its manifest row."""
        sequence_id = self._safe_component(metadata.sequence_id, 'sequence_id')
        if metadata.frame_id < 0:
            raise ValueError('frame_id must be non-negative')
        validate_timestamp_ns(metadata.sample_timestamp_ns)

        sequence_dir = self.root / 'sequences' / sequence_id
        sequence_path = sequence_dir / 'sequence.json'
        if not sequence_path.exists():
            self._atomic_json(sequence_path, {
                'schema_version': SCHEMA_VERSION,
                'sequence_id': sequence_id,
                'coordinate_convention': metadata.coordinate_convention,
                'calibration': None,
                'calibration_status': 'TODO_sensor_interfaces_not_confirmed',
            })
        frame_stem = ('{0:0' + str(self.frame_digits) + 'd}').format(
            metadata.frame_id
        )
        label_path = sequence_dir / 'labels' / (frame_stem + '.json')
        metadata_path = sequence_dir / 'metadata' / (frame_stem + '.json')
        if label_path.exists() or metadata_path.exists():
            raise FileExistsError(
                'frame {0}/{1} was already recorded'.format(
                    sequence_id, metadata.frame_id
                )
            )

        artifact_records = []
        for artifact in tuple(artifacts):
            modality = self._safe_component(artifact.modality, 'modality')
            sensor_id = self._safe_component(artifact.sensor_id, 'sensor_id')
            extension = artifact.extension.lstrip('.').lower()
            self._safe_component(extension, 'extension')
            validate_timestamp_ns(artifact.timestamp_ns)
            relative_path = Path('sequences') / sequence_id / modality / sensor_id
            relative_path /= frame_stem + '.' + extension
            absolute_path = self.root / relative_path
            if absolute_path.exists():
                raise FileExistsError(str(absolute_path))
            self._atomic_bytes(absolute_path, bytes(artifact.content))
            artifact_records.append({
                'modality': modality,
                'sensor_id': sensor_id,
                'path': relative_path.as_posix(),
                'timestamp_ns': artifact.timestamp_ns,
                'source_frame_id': artifact.source_frame_id,
                'clock_domain': artifact.clock_domain,
                'size_bytes': len(artifact.content),
                'sha256': hashlib.sha256(artifact.content).hexdigest(),
            })

        label_values = tuple(labels)
        label_document = {
            'schema_version': SCHEMA_VERSION,
            'sequence_id': sequence_id,
            'frame_id': metadata.frame_id,
            'sample_timestamp_ns': metadata.sample_timestamp_ns,
            'coordinate_frame': TARGET_COORDINATE_FRAME,
            'objects': [asdict(label) for label in label_values],
        }
        metadata_document = asdict(metadata)
        metadata_document.update({
            'schema_version': SCHEMA_VERSION,
            'sensor_artifacts': artifact_records,
        })

        # Metadata/labels become visible only after all sensor artifacts exist.
        self._atomic_json(label_path, label_document)
        self._atomic_json(metadata_path, metadata_document)

        manifest_record = {
            'sequence_id': sequence_id,
            'frame_id': metadata.frame_id,
            'sample_timestamp_ns': metadata.sample_timestamp_ns,
            'labels': label_path.relative_to(self.root).as_posix(),
            'metadata': metadata_path.relative_to(self.root).as_posix(),
        }
        manifest_path = sequence_dir / 'manifest.jsonl'
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_line = json.dumps(
            manifest_record, ensure_ascii=False, sort_keys=True,
            allow_nan=False,
        ) + '\n'
        with self._manifest_lock:
            with manifest_path.open('a', encoding='utf-8') as stream:
                stream.write(manifest_line)
                stream.flush()
                os.fsync(stream.fileno())

        return label_path, metadata_path
