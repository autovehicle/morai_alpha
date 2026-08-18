#!/usr/bin/env bash
# Applies patch_autobahn_websocket.py to whichever rosbridge_server package
# is actually live (respects catkin workspace overlays — e.g. a workspace
# vendoring rosbridge_suite under src/ will shadow the system /opt/ros
# copy). Only escalates to sudo if the resolved file isn't writable by the
# current user, so files inside your own workspace don't end up root-owned.
# Safe to re-run after an apt upgrade or a workspace rebuild.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the target path as the invoking (non-root) user: sudo resets
# PYTHONPATH, so doing this resolution inside a sudo'd process would fail to
# import the ROS-sourced rosbridge_server package.
# Note: importing rosbridge_library prints "registered capabilities" noise
# to stdout as a side effect, so `tail -n 1` isolates the actual path (our
# print(m.__file__) always runs last, after all import side effects).
if [ $# -ge 1 ]; then
    TARGET="$1"
    shift
else
    TARGET="$(python3 -c 'import rosbridge_server.autobahn_websocket as m; print(m.__file__)' | tail -n 1)"
fi

echo "target: $TARGET"

if [ -w "$TARGET" ]; then
    python3 "$SCRIPT_DIR/patch_autobahn_websocket.py" "$TARGET" "$@"
else
    sudo python3 "$SCRIPT_DIR/patch_autobahn_websocket.py" "$TARGET" "$@"
fi
