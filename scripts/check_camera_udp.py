"""Smoke-test one MORAI Camera UDP endpoint without rosbridge."""

import argparse
import threading
import time
from pathlib import Path

from sensors.camera_receiver import CameraReceiver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind-ip', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9291)
    parser.add_argument('--sensor-id', default='camera_front')
    parser.add_argument('--timeout', type=float, default=15.0)
    parser.add_argument('--save-first', type=Path)
    args = parser.parse_args()

    first_frame = threading.Event()
    result = {}

    def on_frame(frame):
        if first_frame.is_set():
            return
        result['frame'] = frame
        if args.save_first is not None:
            args.save_first.parent.mkdir(parents=True, exist_ok=True)
            args.save_first.write_bytes(frame.payload)
        first_frame.set()

    receiver = CameraReceiver(
        sensor_id=args.sensor_id,
        bind_ip=args.bind_ip,
        bind_port=args.port,
    )
    print(
        'Waiting for MORAI Camera UDP on {0}:{1} ...'.format(
            args.bind_ip, args.port
        )
    )
    receiver.start(on_frame)
    try:
        deadline = time.monotonic() + args.timeout
        while not first_frame.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        receiver.stop()

    frame = result.get('frame')
    print('receiver_stats={0}'.format(dict(receiver.stats)))
    if frame is None:
        print(
            'No complete frame. Check MORAI Destination IP/Port, Windows '
            'firewall, and packet-loss counters above.'
        )
        return 1

    print(
        'OK sensor_id={0} timestamp_ns={1} jpeg_bytes={2} packets={3}'.format(
            frame.sensor_id,
            frame.timestamp_ns,
            len(frame.payload),
            frame.metadata['packet_count'],
        )
    )
    if args.save_first is not None:
        print('saved={0}'.format(args.save_first))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
