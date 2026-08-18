import struct
import unittest

from network.UDP.protocol import EgoState, ObjectFrame
from network.UDP.receiver import EgoReceiver, ObjectReceiver
from network.UDP.udp_manager import UdpManager


class UdpTimestampTest(unittest.TestCase):
    def test_ego_packet_exposes_timestamp_ns(self):
        raw = bytearray(141)
        raw[:11] = b'#MoraiInfo$'
        struct.pack_into('<i', raw, 11, 216)
        struct.pack_into('<f', raw, 27, 12.0)
        struct.pack_into('<f', raw, 31, 345.0)

        receiver = EgoReceiver.__new__(EgoReceiver)
        ego = receiver._parse(bytes(raw))

        self.assertEqual(ego.timestamp_ns, 12_000_000_345)

    def test_empty_object_packet_keeps_packet_timestamp(self):
        raw = bytearray(38)
        raw[:14] = b'#MoraiObjInfo$'
        struct.pack_into('<i', raw, 14, 8)
        struct.pack_into('<f', raw, 30, 7.0)
        struct.pack_into('<f', raw, 34, 21.0)

        receiver = ObjectReceiver.__new__(ObjectReceiver)
        frame = receiver._parse(bytes(raw))

        self.assertIsInstance(frame, ObjectFrame)
        self.assertEqual(len(frame), 0)
        self.assertEqual(frame.timestamp_ns, 7_000_000_021)

    def test_udp_manager_buffers_without_opening_sockets(self):
        manager = UdpManager(buffer_size=2)
        ego = EgoState(timestamp_ns=100)
        objects = ObjectFrame(timestamp_ns=101)

        manager._on_ego(ego)
        manager._on_objects(objects)

        latest_ego, latest_objects, _ = manager.snapshot_latest()
        self.assertIs(latest_ego, ego)
        self.assertEqual(latest_objects.timestamp_ns, 101)
        self.assertEqual(manager.drain_ego_states(), [ego])
        self.assertEqual(manager.drain_object_frames()[0].timestamp_ns, 101)
        self.assertEqual(manager.drain_ego_states(), [])

    def test_udp_manager_reports_bounded_buffer_loss(self):
        manager = UdpManager(buffer_size=1)
        manager._on_ego(EgoState(timestamp_ns=1))
        manager._on_ego(EgoState(timestamp_ns=2))
        self.assertEqual(manager.buffer_stats['ego_pending'], 1)
        self.assertEqual(manager.buffer_stats['ego_dropped'], 1)


if __name__ == '__main__':
    unittest.main()
