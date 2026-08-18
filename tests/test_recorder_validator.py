import tempfile
import unittest
from pathlib import Path

from dataset.recorder import DatasetRecorder, SensorArtifact
from dataset.schema import (
    FrameMetadata,
    ObjectLabel,
    RawEgoState,
    RawObjectState,
    SyncSource,
)
from dataset.validator import DatasetValidator


class RecorderValidatorTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recorder = DatasetRecorder(root)
            ego = RawEgoState(
                timestamp_ns=101,
                world_position_m=(1.0, 2.0, 3.0),
                roll_pitch_yaw_deg=(0.0, 0.0, 90.0),
                velocity_kmh=(10.0, 0.0, 0.0),
                signed_velocity_kmh=10.0,
                size_m=(4.5, 1.8, 1.5),
            )
            raw_object = RawObjectState(
                timestamp_ns=102, obj_id=35, obj_type=1,
                world_position_m=(11.0, 2.0, 3.0),
                heading_deg=90.0,
                size_xyz_m=(4.6, 1.8, 1.5),
                velocity_kmh=(5.0, 0.0, 0.0),
                acceleration_mps2=(0.0, 0.0, 0.0),
            )
            metadata = FrameMetadata(
                sequence_id='sequence_0001',
                frame_id=1,
                sample_timestamp_ns=100,
                anchor_stream='lidar',
                coordinate_convention='verified_test_convention',
                ego_raw=ego,
                objects_raw=(raw_object,),
                synchronization={
                    'lidar': SyncSource(100, 0, source_frame_id=10),
                    'camera_front': SyncSource(99, -1, source_frame_id=20),
                    'ego': SyncSource(101, 1),
                    'objects': SyncSource(102, 2),
                },
            )
            label = ObjectLabel(
                obj_id=35, class_id=1, class_name='vehicle',
                position_ego_m=(10.0, 0.0, 0.0),
                size_lwh_m=(4.6, 1.8, 1.5),
                yaw_ego_rad=0.0,
            )
            artifacts = (
                SensorArtifact('camera', 'front', 'png', b'camera', 99, 20),
                SensorArtifact('lidar', 'top', 'bin', b'lidar', 100, 10),
            )

            label_path, metadata_path = recorder.record_frame(
                metadata, (label,), artifacts
            )

            self.assertTrue(label_path.is_file())
            self.assertTrue(metadata_path.is_file())
            report = DatasetValidator(root, max_sync_offset_ns=10).validate()
            self.assertTrue(report.is_valid, report.issues)
            self.assertEqual(report.frame_count, 1)
            continuity = DatasetValidator(root).audit_obj_id_continuity()
            self.assertEqual(continuity['sequence_0001'][35]['frames'], [1])
            self.assertFalse(continuity['sequence_0001'][35]['type_changed'])


if __name__ == '__main__':
    unittest.main()
