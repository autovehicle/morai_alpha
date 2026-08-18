#!/usr/bin/env python3
"""Patch rosbridge_server's IncomingQueue.run() so one malformed message
can't silently kill a client's message-processing thread.

Without this patch, IncomingQueue.run() calls self.protocol.incoming(msg)
with no exception handling. When rosbridge's JSON buffer-recovery parser
(rosbridge_library/protocol.py) throws on a malformed/fragmented message,
the whole processing thread dies silently: the websocket connection stays
open but stops handling any further messages for that client, which looks
like a dropped connection.

Also logs a bounded sample of the exact `msg` that triggered the failure
(type, length, first 300 chars/bytes) — the buffer-recovery bug's real
trigger (malformed JSON? a binary WebSocket frame reaching the text-mode
buffer path? something else?) has never been directly observed, so this is
the evidence needed to fix it properly instead of just containing it.

Idempotent and self-upgrading: guarded by a versioned marker comment. Safe
to re-run (e.g. after `apt upgrade` reinstalls the package, or after this
script itself changes) — a stale patch from an older version of this
script is detected, restored from backup, and re-patched with the current
version.
"""
import argparse
import os
import shutil
import sys

MARKER_PREFIX = "# rosbridge-resilience-patch"
VERSION = 2
MARKER = f"{MARKER_PREFIX} v{VERSION}"

OLD = "            self.protocol.incoming(msg)\n"

NEW = (
    "            try:  " + MARKER + "\n"
    "                self.protocol.incoming(msg)\n"
    "            except Exception:\n"
    "                _log_exception()\n"
    "                try:\n"
    "                    sample = msg[:300] if hasattr(msg, \"__getitem__\") else msg\n"
    "                    rospy.logerr(\n"
    "                        \"rosbridge-resilience-patch: msg type=%s len=%s sample=%r\"\n"
    "                        % (type(msg).__name__, len(msg) if hasattr(msg, \"__len__\") else \"?\", sample)\n"
    "                    )\n"
    "                except Exception:\n"
    "                    pass\n"
    "                if getattr(self.protocol, \"bson_only_mode\", False):\n"
    "                    self.protocol.buffer = bytearray()\n"
    "                else:\n"
    "                    self.protocol.buffer = \"\"\n"
)


def default_target_path():
    # Import the submodule itself (not just the rosbridge_server package) so
    # this resolves correctly even under a catkin devel-space overlay, where
    # the package's __init__.py lives in devel/ but individual modules are
    # loaded straight from the workspace's src/ tree via __path__ tricks —
    # joining the package __init__.py's dirname with the filename (the naive
    # approach) silently produces a path that doesn't exist in that case.
    import rosbridge_server.autobahn_websocket as m
    return m.__file__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to autobahn_websocket.py (default: auto-detected from the "
        "installed rosbridge_server package)",
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
        # A stale version of this patch is present. Restore the pristine
        # original and re-patch with the current NEW template below.
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
            f"found {content.count(OLD)} — the installed rosbridge_server version "
            "may differ from what this patch targets. Inspect IncomingQueue.run() "
            "manually instead of patching.",
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
