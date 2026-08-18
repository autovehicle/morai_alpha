import struct
import unittest

from sensors.camera_receiver import (
    CameraFrameAssembler,
    CameraPacketError,
    parse_camera_packet,
)


def camera_datagram(timestamp_sec, timestamp_nsec, index, chunk, final=False):
    tail = b'EI' if final else b'AI'
    return (
        b'MOR'
        + struct.pack('<IIii', timestamp_sec, timestamp_nsec, index, len(chunk))
        + chunk
        + tail
    )


class CameraPacketTest(unittest.TestCase):
    def test_parses_timestamp_index_size_and_final_marker(self):
        packet = parse_camera_packet(
            camera_datagram(12, 345, 0, b'jpeg-part', final=True)
        )

        self.assertEqual(packet.timestamp_ns, 12_000_000_345)
        self.assertEqual(packet.index, 0)
        self.assertEqual(packet.declared_size, 9)
        self.assertEqual(packet.jpeg_chunk, b'jpeg-part')
        self.assertTrue(packet.is_final)

    def test_rejects_declared_size_mismatch(self):
        datagram = bytearray(camera_datagram(1, 2, 0, b'abc', final=True))
        struct.pack_into('<i', datagram, 15, 4)

        with self.assertRaises(CameraPacketError):
            parse_camera_packet(bytes(datagram))

    def test_rejects_out_of_range_nanoseconds(self):
        with self.assertRaises(CameraPacketError):
            parse_camera_packet(
                camera_datagram(1, 1_000_000_000, 0, b'abc', final=True)
            )


class CameraAssemblerTest(unittest.TestCase):
    def test_reassembles_out_of_order_packets_by_timestamp_and_index(self):
        jpeg = b'\xff\xd8first-middle-last\xff\xd9'
        chunks = (jpeg[:8], jpeg[8:15], jpeg[15:])
        packets = [
            parse_camera_packet(camera_datagram(10, 20, 2, chunks[2], True)),
            parse_camera_packet(camera_datagram(10, 20, 0, chunks[0])),
            parse_camera_packet(camera_datagram(10, 20, 1, chunks[1])),
        ]
        assembler = CameraFrameAssembler()

        self.assertIsNone(assembler.push(packets[0], monotonic_ns=1))
        self.assertIsNone(assembler.push(packets[1], monotonic_ns=2))
        frame = assembler.push(packets[2], monotonic_ns=3)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.timestamp_ns, 10_000_000_020)
        self.assertEqual(frame.jpeg, jpeg)
        self.assertEqual(frame.packet_count, 3)
        self.assertEqual(assembler.stats['completed_frames'], 1)

    def test_missing_packet_never_emits_a_frame(self):
        assembler = CameraFrameAssembler()
        packet = parse_camera_packet(
            camera_datagram(10, 20, 2, b'last\xff\xd9', True)
        )

        self.assertIsNone(assembler.push(packet, monotonic_ns=1))
        self.assertEqual(assembler.pending_frames, 1)

    def test_expired_incomplete_frame_is_counted(self):
        assembler = CameraFrameAssembler(assembly_timeout_ns=10)
        first = parse_camera_packet(
            camera_datagram(1, 0, 0, b'\xff\xd8unfinished')
        )
        next_frame = parse_camera_packet(
            camera_datagram(2, 0, 0, b'\xff\xd8next')
        )

        assembler.push(first, monotonic_ns=1)
        assembler.push(next_frame, monotonic_ns=12)

        self.assertEqual(assembler.stats['incomplete_frames'], 1)

    def test_identical_duplicate_is_ignored_and_counted(self):
        assembler = CameraFrameAssembler()
        packet = parse_camera_packet(
            camera_datagram(1, 0, 0, b'\xff\xd8first')
        )

        assembler.push(packet, monotonic_ns=1)
        self.assertIsNone(assembler.push(packet, monotonic_ns=2))
        self.assertEqual(assembler.stats['duplicates'], 1)


if __name__ == '__main__':
    unittest.main()
