#!/usr/bin/env python3
"""Make autobahn's hl() (colored debug-log text helper) crash-proof.

Observed in practice: under sustained multi-client load (4 rosbridge
clients streaming camera+lidar), autobahn.util.hl()'s call to
click.style() intermittently raises "TypeError: 'bool' object is not
callable" — click.style is not actually reassigned anywhere in this
codebase (checked: no monkeypatching, single click install), so this
looks like a concurrency hazard inside click 7.0 itself when hit from
multiple threads at once (autobahn.twisted.websocket.dataReceived() calls
hl()/hlval() unconditionally on every socket read, regardless of whether
debug logging is even enabled, since Python evaluates log call arguments
eagerly).

hl() only produces cosmetic colored text for debug log lines — nothing
about actual WebSocket/rosbridge data handling depends on its result being
correct. So instead of chasing the exact click threading bug, this patch
makes the styling fail open: any exception from click.style() just
falls back to the plain unstyled text, same as when click isn't installed
at all.

Idempotent and self-upgrading, same scheme as the other rosbridge patches
in this directory.
"""
import argparse
import os
import shutil
import sys

MARKER_PREFIX = "# rosbridge-autobahn-hl-patch"
VERSION = 1
MARKER = f"{MARKER_PREFIX} v{VERSION}"

OLD = (
    "    if _HAS_CLICK:\n"
    "        return click.style(text, fg=color, bold=bold)\n"
    "    else:\n"
    "        return text\n"
)

NEW = (
    "    if _HAS_CLICK:  " + MARKER + "\n"
    "        try:\n"
    "            return click.style(text, fg=color, bold=bold)\n"
    "        except Exception:\n"
    "            return text\n"
    "    else:\n"
    "        return text\n"
)


def default_target_path():
    import autobahn.util as m
    return m.__file__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to autobahn/util.py (default: auto-detected from the "
        "installed autobahn package)",
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
                "the file manually before re-running.",
                file=sys.stderr,
            )
            return 1
        shutil.copy2(backup, target)
        with open(target, "r") as f:
            content = f.read()
        print(f"restored pristine original from {backup} to upgrade patch")

    if content.count(OLD) != 1:
        print(
            f"error: expected exactly one match for the target block in {target}, "
            f"found {content.count(OLD)} — the installed autobahn version may "
            "differ from what this patch targets. Inspect hl() manually instead "
            "of patching.",
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
