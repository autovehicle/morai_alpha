"""Wire the ROS receivers into FrameSynchronizer -> GTConverter -> DatasetRecorder.

Every piece this module uses already exists and is independently verified:
``adapters/ros/*_receiver.py`` (live-tested against real MORAI streams),
``dataset/frame_sync.py`` (unit-tested), ``dataset/gt_converter.py``
(unit-tested), ``dataset/recorder.py`` (unit-tested). This module is only
the glue that connects them.

Camera note: a single ``RosCameraReceiver`` subscribes to one topic that
carries all three MORAI cameras, distinguished by ``source_frame_id``
(``Camera-1``/``Camera-2``/``Camera-3``, confirmed via ``rostopic echo``).
This collector demuxes those into three separate synchronizer streams
(``camera_1``/``camera_2``/``camera_3``) since ``FrameSynchronizer``
matches one sample per named stream per bundle.

Coordinate safety gate: ``GTConverter`` refuses to run (raises
``RuntimeError`` at construction) while
``config/dataset.yaml``'s ``coordinate_convention.verified`` is ``false``.
This collector defaults to ``strict_convention=True``, i.e. it inherits
that refusal — real collection cannot start until the MORAI coordinate
convention has been verified (see ``docs/DATASET_SCHEMA.md``) and the config
flipped to ``verified: true``. Passing
``strict_convention=False`` explicitly bypasses the gate for wiring/smoke
tests only; frames recorded that way are not valid training labels and
must not be written into the real ``output_root``.
"""

import io
import queue
import threading
import time
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import yaml

from adapters.ros import RosCameraReceiver, RosEgoReceiver, RosGtReceiver, RosLidarReceiver
from geometry.coordinate_transform import CoordinateConvention
from sensors.base import SensorFrame

from .frame_sync import FrameSynchronizer, SynchronizedBundle, TimedSample
from .gt_converter import GTConverter
from .recorder import DatasetRecorder, SensorArtifact
from .schema import FrameMetadata

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'dataset.yaml'

CAMERA_STREAM_BY_FRAME_ID = {
    'Camera-1': 'camera_1',
    'Camera-2': 'camera_2',
    'Camera-3': 'camera_3',
}


def _load_config(config_path: Optional[str]) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


class DatasetCollector:
    """Own the four ROS receivers plus the synchronizer/converter/recorder."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        sequence_id: Optional[str] = None,
        output_root: Optional[str] = None,
        strict_convention: bool = True,
    ):
        config = _load_config(config_path)

        collection_cfg = config['collection']
        self.sequence_id = sequence_id or collection_cfg.get('sequence_id') or (
            'seq_' + __import__('time').strftime('%Y%m%d_%H%M%S')
        )
        root = output_root or collection_cfg['output_root']

        if not strict_convention:
            print(
                'WARNING: DatasetCollector(strict_convention=False) — the '
                'coordinate convention is unverified, so any labels produced '
                'now are not valid training data. Wiring/smoke-test use only.'
            )

        self.recorder = DatasetRecorder(
            root=Path(root), frame_digits=collection_cfg.get('frame_digits', 6)
        )
        convention = CoordinateConvention(**config['coordinate_convention'])
        class_mapping = {
            int(key): (int(key), value)
            for key, value in config['class_mapping'].items()
        }
        self.converter = GTConverter(
            convention=convention,
            class_mapping=class_mapping,
            strict_convention=strict_convention,
        )

        sync_cfg = config['synchronization']
        timestamp_source = sync_cfg.get('timestamp_source', 'header')
        required_streams = ('lidar', 'camera_1', 'camera_2', 'camera_3', 'ego', 'objects')
        self._sync = FrameSynchronizer(
            anchor_stream=sync_cfg['anchor_stream'],
            required_streams=required_streams,
            tolerance_ns=sync_cfg['tolerance_ns'],
            max_buffer_size=sync_cfg.get('max_buffer_size', 256),
        )
        self._sync_lock = threading.Lock()

        sensors_cfg = config['sensors']
        self._camera = RosCameraReceiver(
            sensor_id='camera',
            topic=sensors_cfg['camera']['topic'],
            timestamp_source=timestamp_source,
        )
        self._lidar = RosLidarReceiver(
            sensor_id='lidar3d',
            topic=sensors_cfg['lidar']['topic'],
            timestamp_source=timestamp_source,
        )
        self._ego = RosEgoReceiver(
            topic=sensors_cfg['ego']['topic'],
            tf_topic=sensors_cfg['ego'].get('tf_topic', '/tf'),
            timestamp_source=timestamp_source,
        )
        self._gt = RosGtReceiver(
            topic=sensors_cfg['objects']['topic'],
            timestamp_source=timestamp_source,
        )

        self._frame_lock = threading.Lock()
        self._next_frame_id = 0
        self._stats = {
            'frames_recorded': 0,
            'record_errors': 0,
            'unmapped_camera_frames': 0,
        }
        self._record_queue = queue.Queue()
        self._record_thread = threading.Thread(
            target=self._record_worker,
            daemon=True,
        )

    @property
    def stats(self) -> Mapping[str, int]:
        result = dict(self._stats)
        result['synchronizer'] = dict(self._sync.stats)
        result['pending'] = self._sync.pending()
        return result

    def start(self) -> None:
        self._record_thread.start()

        self._camera.start(self._on_camera)
        self._lidar.start(self._on_lidar)
        self._ego.start(self._on_ego)
        self._gt.start(self._on_gt)

    def stop(self) -> None:
        self._camera.stop()
        self._lidar.stop()
        self._ego.stop()
        self._gt.stop()
        with self._sync_lock:
            self._drain_locked(force=True)
        self._record_queue.join()
        self._record_queue.put(None)
        self._record_thread.join()

    # ── receiver callbacks ──────────────────────────────────────────

    def _on_camera(self, frame: SensorFrame) -> None:
        stream = CAMERA_STREAM_BY_FRAME_ID.get(frame.source_frame_id)
        if stream is None:
            self._stats['unmapped_camera_frames'] += 1
            return
        self._add(stream, TimedSample(
            timestamp_ns=frame.metadata['received_monotonic_ns'],
            payload=frame,
            source_frame_id=frame.source_frame_id,
            clock_domain='ros_receive_monotonic',
        ))

    def _on_lidar(self, frame: SensorFrame) -> None:
        self._add('lidar', TimedSample(
            timestamp_ns=frame.metadata['received_monotonic_ns'],
            payload=frame,
            source_frame_id=frame.source_frame_id,
            clock_domain='ros_receive_monotonic',
        ))

    def _on_ego(self, ego_state) -> None:
        self._add('ego', TimedSample(
            timestamp_ns=time.monotonic_ns(),
            payload=ego_state,
            clock_domain='ros_receive_monotonic',
        ))

    def _on_gt(self, object_frame) -> None:
        self._add('objects', TimedSample(
            timestamp_ns=time.monotonic_ns(),
            payload=object_frame,
            clock_domain='ros_receive_monotonic',
        ))

    def _add(self, stream: str, sample: TimedSample) -> None:
        with self._sync_lock:
            self._sync.add(stream, sample)
            self._drain_locked()

    # ── synchronization draining ────────────────────────────────────

    def _drain_locked(self, force: bool = False) -> None:
        while True:
            bundle = self._sync.pop_next(force=force)
            if bundle is None:
                return
            self._record_queue.put(bundle)

    def _record_worker(self) -> None:
        while True:
            bundle = self._record_queue.get()
            try:
                if bundle is None:
                    return
                try:
                    self._record_bundle(bundle)
                except Exception:
                    self._stats['record_errors'] += 1
            finally:
                self._record_queue.task_done()

    def _record_bundle(self, bundle: SynchronizedBundle) -> None:
        ego_state = bundle.samples['ego'].payload
        object_frame = bundle.samples['objects'].payload

        raw_ego, raw_objects, labels = self.converter.convert_frame(ego_state, object_frame)

        artifacts = []
        for stream in ('camera_1', 'camera_2', 'camera_3'):
            cam_sample = bundle.samples[stream]
            cam_frame: SensorFrame = cam_sample.payload
            artifacts.append(SensorArtifact(
                modality='camera',
                sensor_id=stream,
                extension='jpg',
                content=cam_frame.payload,
                timestamp_ns=cam_sample.timestamp_ns,
                source_frame_id=cam_frame.source_frame_id,
                clock_domain=cam_sample.clock_domain,
                metadata=cam_frame.metadata,
            ))

        lidar_sample = bundle.samples['lidar']
        lidar_frame: SensorFrame = lidar_sample.payload
        buffer = io.BytesIO()
        np.save(buffer, lidar_frame.payload)
        artifacts.append(SensorArtifact(
            modality='lidar',
            sensor_id='lidar3d',
            extension='npy',
            content=buffer.getvalue(),
            timestamp_ns=lidar_sample.timestamp_ns,
            source_frame_id=lidar_frame.source_frame_id,
            clock_domain=lidar_sample.clock_domain,
            metadata=lidar_frame.metadata,
        ))

        with self._frame_lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1

        metadata = FrameMetadata(
            sequence_id=self.sequence_id,
            frame_id=frame_id,
            sample_timestamp_ns=bundle.timestamp_ns,
            anchor_stream=bundle.anchor_stream,
            coordinate_convention=self.converter.convention.name,
            ego_raw=raw_ego,
            objects_raw=raw_objects,
            synchronization=bundle.sync_sources(),
        )
        try:
            self.recorder.record_frame(metadata, labels, artifacts)
            self._stats['frames_recorded'] += 1
        except Exception:
            self._stats['record_errors'] += 1
