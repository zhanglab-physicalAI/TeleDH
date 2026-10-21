"""Argument parsing and command dispatch for ``run.py``.

Each subcommand lives in ``telehand.commands.<name>`` and is imported only
when invoked. Commands that drive the hand or the camera run inside the
one-instance session lock (see ``session.py``); the rest do not need it.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from . import paths, session
from .experimental import cli as _experimental
from .hand import DEFAULT_BAUD, DEFAULT_PORT

# Commands that touch neither the hand nor a camera exclusively.
UNLOCKED_COMMANDS = {"stop", "handview", "cameras", "latency", *_experimental.NO_HARDWARE}


def build_parser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they can be given after the
    # subcommand, which is where people naturally reach for them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--port", default=DEFAULT_PORT, help="RS485 serial port")
    common.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    common.add_argument(
        "--camera",
        default="auto",
        help="camera index, /dev node, by-path link, or name fragment; "
        "'auto' uses the one saved by `run.py cameras --use`",
    )
    common.add_argument(
        "--hand", default="Right", choices=["Right", "Left"], help="which of your hands to track"
    )
    common.add_argument("--model", default=str(paths.HAND_LANDMARKER_MODEL))
    common.add_argument(
        "--poses", default=str(paths.POSES_FILE), help="pose library: robot joint angles per named pose"
    )
    common.add_argument(
        "--mirror", action="store_true", default=True, help="mirror the camera image (default: on)"
    )
    common.add_argument("--no-mirror", dest="mirror", action="store_false")
    common.add_argument("--no-display", action="store_true", help="run without an OpenCV window")

    p = argparse.ArgumentParser(
        prog="telehand",
        description="Teleoperate the LHandPro dexterous hand with camera hand tracking.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", parents=[common], help="verify camera, tracking and hand link; no motion")
    cal = sub.add_parser(
        "calibrate", parents=[common], help="record the tracker signals for the poses in the library"
    )
    cal.add_argument(
        "--all", action="store_true", help="re-record every pose, not just the ones missing signals"
    )

    t = sub.add_parser(
        "teleop",
        parents=[common],
        help="run live teleoperation",
        description="Live teleoperation. Recommended:\n"
        "  python3 run.py teleop --mode fk --smoothing 0.7 --deadband 0.4 "
        "--max-speed 300 --no-serial-flush\n"
        "Add --show-3d-pose for the phantom hand view. Options marked [experimental] "
        "are not part of the stable path.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = t.add_argument_group("retargeting")
    g.add_argument(
        "--mode",
        choices=["fk", "poses", "direct", "hybrid"],
        default="fk",
        help="fk: kinematic solver on fingertip geometry (default, recommended); "
        "poses: pose-library blend; direct: vendor-style per-joint linear map; "
        "hybrid: direct fingers + pose-blend thumb",
    )
    g.add_argument(
        "--smoothing",
        type=float,
        default=0.35,
        help="EMA factor on commanded angles; lower is smoother but laggier (0.7 recommended)",
    )
    g.add_argument(
        "--deadband",
        type=float,
        default=0.8,
        help="ignore commanded changes smaller than this, in degrees (0.4 recommended)",
    )
    g.add_argument(
        "--lost-timeout-frames",
        type=int,
        default=15,
        help="frames without a hand before the filters reset (the last pose is held)",
    )
    g.add_argument(
        "--profile", default=str(paths.PROFILE_FILE), help="fk: operator hand profile from `run.py profile`"
    )
    g.add_argument(
        "--no-tactile",
        dest="tactile",
        action="store_false",
        default=True,
        help="fk: disable closing pinches by feel (thumb-tip sensor / finger stall)",
    )
    g.add_argument(
        "--allow-estimates",
        action="store_true",
        help="poses: also blend poses whose robot joints are estimates, not jog-measured",
    )
    g.add_argument(
        "--blend-power",
        type=float,
        default=2.5,
        help="poses: blending sharpness; higher snaps harder to the nearest recorded pose",
    )
    g.add_argument(
        "--direct-calibration",
        default=str(paths.DIRECT_CALIBRATION_FILE),
        help="direct: per-joint ranges captured with keys 1 / 2",
    )
    g.add_argument("--lm-alpha", type=float, default=0.3, help="direct: landmark EMA factor")
    g.add_argument(
        "--lm-deadband", type=float, default=0.02, help="direct: landmark deadband in normalized units"
    )
    g.add_argument("--no-kalman", action="store_true", help="direct: disable the per-landmark Kalman stage")

    g = t.add_argument_group("hand and safety")
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="track and display but never command the hand (no homing, motors stay disabled)",
    )
    g.add_argument(
        "--no-home",
        dest="home",
        action="store_false",
        default=True,
        help="skip homing (hand must already be homed)",
    )
    g.add_argument(
        "--max-speed", type=float, default=180.0, help="host-side joint rate limit, deg/s (300 recommended)"
    )
    g.add_argument(
        "--max-current", type=int, default=500, help="per-motor current limit in per-mille (1000 = full)"
    )
    g.add_argument(
        "--no-stall-kick",
        dest="stall_kick",
        action="store_false",
        default=True,
        help="disable the brief overshoot that unsticks a worm-drive joint parked short of target",
    )
    g.add_argument(
        "--keep-enabled",
        action="store_true",
        help="leave the motors energized and holding on exit (default: de-energize)",
    )
    g.add_argument(
        "--no-serial-flush",
        action="store_true",
        help="skip the vendor serial layer's tcdrain after each write (recommended: "
        "-27%% command latency, ordering preserved; patched at runtime)",
    )

    g = t.add_argument_group("visualisation (never touches control)")
    g.add_argument(
        "--show-3d-pose",
        action="store_true",
        help="phantom 3-D hand beside the camera view; key p toggles, [ ] rotate",
    )
    g.add_argument(
        "--show-joint-frames",
        action="store_true",
        help="3-D hand: also draw every articulated joint's local XYZ frame; key f toggles",
    )
    g.add_argument(
        "--pose-hz",
        type=float,
        default=30.0,
        metavar="HZ",
        help="3-D hand refresh rate; below the camera rate the last render is reused",
    )
    g.add_argument(
        "--show-measured",
        action="store_true",
        help="outline the pose the hand actually reached over the commanded one",
    )
    g.add_argument("--diag", action="store_true", help="overlay retargeter diagnostics on the camera view")

    g = t.add_argument_group("profiling and recording")
    g.add_argument(
        "--latency-csv",
        default=None,
        metavar="PATH",
        help="write per-frame stage timings to this CSV (the p50/p95 summary prints either way)",
    )
    g.add_argument(
        "--record",
        default=None,
        metavar="CSV",
        help="log per-frame signals and commands to CSV, and the 3-D/2-D landmarks to a "
        ".npz beside it (replayable by `run.py frames`)",
    )

    g = t.add_argument_group("[experimental] serial write reduction")
    g.add_argument(
        "--skip-unchanged",
        action="store_true",
        help="only send set_target_angle for joints whose command moved since last sent "
        "(measured: ~3%% latency gain, kept optional)",
    )
    g.add_argument(
        "--skip-epsilon",
        type=float,
        default=0.1,
        metavar="DEG",
        help="how far a joint must move to be resent",
    )
    g.add_argument(
        "--resync-frames",
        type=int,
        default=30,
        metavar="N",
        help="resend every joint every N writes so a dropped RS485 frame cannot leave a "
        "stale target (0 disables)",
    )

    g = t.add_argument_group("[experimental] frame-based control -- not recommended, kept for comparison")
    g.add_argument(
        "--pose-mode",
        choices=("scalar", "frames"),
        default="scalar",
        help="scalar: the FK solver for all joints (default). frames: thumb abduction/"
        "flexion and index flexion from the wrist-local frame representation, "
        "everything else and every fallback from FK (fk mode only)",
    )
    g.add_argument(
        "--thumb-abd",
        choices=("swing", "lift"),
        default="swing",
        help="frames: thumb-metacarpal quantity driving abduction",
    )
    g.add_argument(
        "--frames-calibration",
        default=str(paths.FRAMES_CALIBRATION_FILE),
        metavar="JSON",
        help="frames: open/closed ranges captured with keys 1 / 2",
    )
    g.add_argument(
        "--frames-min-conf",
        type=float,
        default=0.05,
        metavar="C",
        help="frames: fall back to FK below this bone-length confidence",
    )
    g.add_argument(
        "--frames-history",
        type=int,
        default=300,
        metavar="N",
        help="frames: bone-length history behind the confidence estimate",
    )
    g.add_argument(
        "--pose-log",
        default=None,
        metavar="CSV",
        help="log scalar signals, frame signals, FK vs frame commands and the final "
        "command every frame (builds the frame mapper even in scalar mode)",
    )

    j = sub.add_parser("jog", parents=[common], help="drive joints by hand to see what each one does")
    j.add_argument("--dry-run", action="store_true")
    j.add_argument("--no-home", dest="home", action="store_false", default=True)
    j.add_argument("--max-speed", type=float, default=180.0)
    j.add_argument("--max-current", type=int, default=500)
    j.add_argument(
        "--no-stall-kick",
        dest="stall_kick",
        action="store_false",
        default=True,
        help="disable the brief overshoot that unsticks a worm-drive joint parked short of target",
    )
    j.add_argument(
        "--keep-enabled",
        action="store_true",
        help="leave the motors energized and holding on exit (default: de-energize so the hand goes limp)",
    )

    ts = sub.add_parser("touchscan", parents=[common], help="map contact geometry with the tactile sensors")
    ts.add_argument("--fingers", default="index,middle,ring,pinky")
    ts.add_argument("--abd", default="0,10,20,30,40,50,60", help="thumb abduction grid, degrees")
    ts.add_argument("--flex", default="0,10,20,30", help="thumb flexion grid, degrees")
    ts.add_argument("--threshold", type=float, default=0.05, help="thumb-tip pressure that counts as contact")
    ts.add_argument("--step", type=float, default=2.0, help="finger step, degrees")
    ts.add_argument("--dwell", type=float, default=0.12, help="seconds per step")
    ts.add_argument("--speed", type=float, default=90.0, help="deg/s")
    ts.add_argument("--max-current", type=int, default=300)
    ts.add_argument("--out", default=str(paths.TOUCHSCAN_FILE))

    ph = sub.add_parser(
        "photostep", parents=[common], help="park joints at preset angles for calibration photos"
    )
    ph.add_argument(
        "--joints",
        default="2,1,0",
        help="joint indices to step, 0-based (2=index, 1=thumb flex, 0=thumb abd)",
    )
    ph.add_argument("--max-current", type=int, default=500)

    cams = sub.add_parser("cameras", help="list capture devices and their stable paths")
    cams.add_argument(
        "--settle", type=float, default=1.5, help="seconds to let auto-exposure settle before measuring"
    )
    cams.add_argument("--use", default=None, help="remember this camera (index, node or name) as the default")

    hv = sub.add_parser("handview", help="render the hand from its URDF; no camera or motors")
    hv.add_argument("--pose", default=None, help="six firmware angles, comma separated")
    hv.add_argument("--poses", default=None, help="browse every anchor in a pose library")
    hv.add_argument("--direction", default="R", choices=["R", "L"])
    hv.add_argument("--width", type=int, default=560)
    hv.add_argument("--height", type=int, default=420)
    hv.add_argument("--out", default=None, help="write one image and exit")

    pf = sub.add_parser(
        "profile", parents=[common], help="measure this operator's hand for fk mode (no motors)"
    )
    pf.add_argument("--out", default=str(paths.PROFILE_FILE))
    pf.add_argument("--name", default="")

    la = sub.add_parser("latency", help="summarize a latency CSV from a previous teleop run")
    la.add_argument("csv", help="CSV written by `teleop --latency-csv`")

    _experimental.add_parsers(sub)

    st = sub.add_parser("stop", help="stop the running telehand session cleanly")
    st.add_argument(
        "--timeout", type=float, default=15.0, help="seconds to wait for a clean shutdown before SIGTERM"
    )
    return p


def run_command(name: str, args: argparse.Namespace) -> int:
    if name == "stop":
        return session.stop(timeout=args.timeout)
    if name in _experimental.HANDLERS:
        return _experimental.HANDLERS[name](args)
    module = importlib.import_module(f"telehand.commands.{name}")
    return module.run(args)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in UNLOCKED_COMMANDS:
            return run_command(args.command, args)
        with session.hold(
            args.command, port=getattr(args, "port", None), camera=getattr(args, "camera", None)
        ):
            return run_command(args.command, args)
    except session.SessionBusy as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
