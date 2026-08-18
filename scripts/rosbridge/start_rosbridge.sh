#!/usr/bin/env bash
# Launches rosbridge_websocket and auto-restarts it with a short backoff if
# the roslaunch process ever exits unexpectedly. This is a safety net for
# crash paths other than the one fixed by apply_resilience_patch.sh — run
# that first so a single bad message doesn't take the connection down at
# all.
set -uo pipefail

PORT="${ROSBRIDGE_PORT:-9090}"
ADDRESS="${ROSBRIDGE_ADDRESS:-0.0.0.0}"
BACKOFF="${ROSBRIDGE_RESTART_BACKOFF:-3}"
# MORAI's client was seen sending RSV1-flagged frames while the server had
# use_compression:=false (default), which autobahn correctly rejects as a
# protocol violation ("RSV = 1 and no extension negotiated"), dropping that
# client. Enabling compression here lets the server actually negotiate
# permessage-deflate so those frames become legitimate instead of rejected.
# Set ROSBRIDGE_USE_COMPRESSION=false to roll back if this causes new issues.
USE_COMPRESSION="${ROSBRIDGE_USE_COMPRESSION:-true}"

while true; do
    echo "[start_rosbridge] launching rosbridge_websocket (port=${PORT} address=${ADDRESS} use_compression=${USE_COMPRESSION})"
    roslaunch rosbridge_server rosbridge_websocket.launch \
        port:="${PORT}" address:="${ADDRESS}" bson_only_mode:=false \
        use_compression:="${USE_COMPRESSION}" output:=screen
    status=$?
    echo "[start_rosbridge] roslaunch exited with status ${status}, restarting in ${BACKOFF}s (Ctrl-C to stop)"
    sleep "${BACKOFF}"
done
