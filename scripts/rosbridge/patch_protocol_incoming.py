#!/usr/bin/env python3
"""Fix rosbridge_library's incoming() mangling binary WebSocket frames.

Root cause (confirmed via the diagnostic logging added by
patch_autobahn_websocket.py): when a client sends a message over a
*binary* WebSocket frame (rosbridge_server's onMessage() only
UTF-8-decodes text frames, leaving binary frames as raw `bytes`),
RosbridgeProtocol.incoming() appends it to the text buffer with
`str(message_string)`. In Python 3, str(bytes_obj) does NOT decode it —
it produces the repr, e.g. b'{"topic":...}' becoms the literal 8-character
string "b'{\"top" — so otherwise perfectly valid JSON gets mangled into
garbage before it's even parsed. This is why the buffer-recovery fallback
parser (a known-fragile piece of code) kept throwing on payloads that were
valid JSON all along, e.g. MORAI's LiDAR PointCloud2 publishes sent as
binary frames.

Fix: decode bytes as UTF-8 before appending to the (str) buffer, instead
of taking their repr.

Idempotent and self-upgrading, same scheme as patch_autobahn_websocket.py.
"""
import argparse
import os
import shutil
import sys

MARKER_PREFIX = "# rosbridge-binary-frame-patch"
VERSION = 1
MARKER = f"{MARKER_PREFIX} v{VERSION}"

OLD = "            self.buffer = self.buffer + str(message_string)\n"

NEW = (
    "            if isinstance(message_string, bytes):  " + MARKER + "\n"
    "                message_string = message_string.decode(\"utf-8\")\n"
    "            self.buffer = self.buffer + message_string\n"
)


def default_target_path():
    import rosbridge_library.protocol as m
    return m.__file__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to rosbridge_library/protocol.py (default: auto-detected "
        "from the installed rosbridge_library package)",
    )
    args = parser.parse_args()

    target = args.path or default_target_path()

    if not os.path.isfile(target):
        print(f"error: target file not found: {target}", file=sys.stderr)
        return 1

    backup = target + ".bak"

    with open(target, "r") as f:
        content = f.read()

    if MARKER in content:
        print(f"already patched (current version): {target}")
        return 0

    if MARKER_PREFIX in content:
        if not os.path.exists(backup):
            print(
                f"error: found a stale patch marker in {target} but no backup "
                f"at {backup} to restore from. Refusing to guess — restore "
                "the file manually (e.g. from git/apt) before re-running.",
                file=sys.stderr,
            )
            return 1
        shutil.copy2(backup, target)
        with open(target, "r") as f:
            content = f.read()
        print(f"restored pristine original from {backup} to upgrade patch")

    if content.count(OLD) != 1:
        print(
            f"error: expected exactly one match for the target line in {target}, "
            f"found {content.count(OLD)} — the installed rosbridge_library version "
            "may differ from what this patch targets. Inspect incoming() manually "
            "instead of patching.",
            file=sys.stderr,
        )
        return 1

    if not os.path.exists(backup):
        shutil.copy2(target, backup)
        print(f"backed up original to {backup}")

    with open(target, "w") as f:
        f.write(content.replace(OLD, NEW))

    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
