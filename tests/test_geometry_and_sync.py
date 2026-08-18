import math
import unittest

from dataset.frame_sync import FrameSynchronizer, TimedSample
from dataset.gt_converter import GTConverter
from geometry.coordinate_transform import (
    CoordinateConvention,
    heading_to_ego_yaw,
    world_to_ego_xyz,
)
from network.UDP.protocol import EgoState, ObjectData


class CoordinateTransformTest(unittest.TestCase):
    def setUp(self):
        self.convention = CoordinateConvention(verified=True)

    def test_world_to_forward_left_at_zero_yaw(self):
        result = world_to_ego_xyz(
            (12.0, 3.0, 2.0), (2.0, 1.0, 0.5), 0.0,
            self.convention,
        )
        self.assertEqual(result, (10.0, 2.0, 1.5))

    def test_world_to_forward_left_at_ninety_degrees(self):
        result = world_to_ego_xyz(
            (-2.0, 10.0, 0.0), (0.0, 0.0, 0.0), 90.0,
            self.convention,
        )
        self.assertAlmostEqual(result[0], 10.0, places=6)
        self.assertAlmostEqual(result[1], 2.0, places=6)

    def test_relative_heading_wraps(self):
        result = heading_to_ego_yaw(-170.0, 170.0, self.convention)
        self.assertAlmostEqual(result, math.radians(20.0), places=6)

    def test_converter_requires_verified_convention(self):
        with self.assertRaises(RuntimeError):
            GTConverter(CoordinateConvention(verified=False))

    def test_converter_preserves_raw_and_builds_label(self):
        converter = GTConverter(self.convention)
        ego = EgoState(
            timestamp_ns=100, pos_x=10.0, pos_y=20.0, pos_z=1.0,
            yaw=90.0,
        )
        obj = ObjectData(
            timestamp_ns=101, obj_id=35, obj_type=1,
            pos_x=8.0, pos_y=30.0, pos_z=1.5,
            heading=90.0, size_x=4.6, size_y=1.8, size_z=1.5,
        )

        raw_ego, raw_objects, labels = converter.convert_frame(ego, [obj])

        self.assertEqual(raw_ego.timestamp_ns, 100)
        self.assertEqual(raw_objects[0].obj_id, 35)
        self.assertEqual(labels[0].class_name, 'vehicle')
        self.assertAlmostEqual(labels[0].position_ego_m[0], 10.0, places=6)
        self.assertAlmostEqual(labels[0].position_ego_m[1], 2.0, places=6)
        self.assertAlmostEqual(labels[0].yaw_ego_rad, 0.0, places=6)


class FrameSynchronizerTest(unittest.TestCase):
    def test_selects_nearest_after_stream_watermark(self):
        sync = FrameSynchronizer(
            anchor_stream='lidar',
            required_streams=('camera_front', 'lidar', 'ego', 'objects'),
            tolerance_ns={
                'camera_front': 20,
                'ego': 20,
                'objects': 20,
            },
        )
        sync.add('lidar', TimedSample(100, 'lidar'))
        sync.add('camera_front', TimedSample(90, 'camera-old'))
        self.assertIsNone(sync.pop_next())
        sync.add('camera_front', TimedSample(105, 'camera-new'))
        sync.add('ego', TimedSample(101, 'ego'))
        sync.add('objects', TimedSample(102, 'objects'))

        bundle = sync.pop_next()

        self.assertEqual(bundle.timestamp_ns, 100)
        self.assertEqual(bundle.samples['camera_front'].payload, 'camera-new')
        self.assertEqual(bundle.offset_ns, {
            'lidar': 0,
            'camera_front': 5,
            'ego': 1,
            'objects': 2,
        })
        self.assertEqual(bundle.sync_sources()['camera_front'].offset_ns, 5)

    def test_rejects_duplicate_stream_timestamp(self):
        sync = FrameSynchronizer(
            'lidar', ('lidar', 'ego'), {'ego': 10}
        )
        self.assertTrue(sync.add('ego', TimedSample(1, 'first')))
        self.assertFalse(sync.add('ego', TimedSample(1, 'duplicate')))
        self.assertEqual(sync.stats['duplicates'], 1)

    def test_rejects_mixed_clock_domains(self):
        sync = FrameSynchronizer(
            'lidar', ('lidar', 'ego'), {'ego': 10}
        )
        sync.add('lidar', TimedSample(1, 'lidar', clock_domain='morai_sim'))
        with self.assertRaises(ValueError):
            sync.add('ego', TimedSample(1, 'ego', clock_domain='wall_clock'))


if __name__ == '__main__':
    unittest.main()
