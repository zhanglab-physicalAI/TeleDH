"""Regression guard for the stable control path.

The FK retargeter's output on the canonical recording is frozen in
tests/data/control_reference.npz (default and recommended settings, plus the
six scalar tracker signals). Any change to tracker.py, fkretarget.py or
kinematics.py that alters a commanded angle fails here, before it can reach
the hand. Rendering the phantom view between frames must not change anything
either.

    python3 tests/test_control_replay.py
"""

import sys

from support import DATA_DIR, check, run_standalone

import numpy as np

from telehand import phantom as ph
from telehand.fkretarget import FKRetargeter
from telehand.hand import JointLimit
from telehand.kinematics import FIRMWARE_MAX_DEG
from telehand.tracker import HandReading, extract_signals

REC = np.load(DATA_DIR / "fk_session.npz", allow_pickle=True)
REF = np.load(DATA_DIR / "control_reference.npz")
P, LAB = REC["world_pts"].astype(np.float64), REC["handedness"]
LIMITS = [JointLimit(0.0, m) for m in FIRMWARE_MAX_DEG]


def test_scalar_fk_commands_match_frozen_reference():
    for name, (sm, db) in {"default": (0.35, 0.8), "recommended": (0.7, 0.4)}.items():
        fk = FKRetargeter(LIMITS, smoothing=sm, deadband_deg=db)
        out = np.array([fk(p, "Left") for p in P])
        check(
            np.array_equal(out, REF[name]),
            f"{name} settings (smoothing {sm}, deadband {db}): identical on {len(P)} frames "
            f"(max |diff| {np.abs(out - REF[name]).max():.1e})",
        )
    sig = np.array([extract_signals(p) for p in P])
    check(np.array_equal(sig, REF["signals"]), "tracker scalar signals identical")


def test_phantom_view_between_frames_leaves_commands_untouched():
    fk = FKRetargeter(LIMITS, smoothing=0.7, deadband_deg=0.4)
    view = ph.PhantomHand(hz=0, joint_frames=True)
    out = []
    for p, label in zip(P, LAB):
        out.append(fk(p, "Left"))
        view.render(HandReading(extract_signals(p), str(label), None, p), 480)
    check(
        np.array_equal(np.array(out), REF["recommended"]),
        f"identical with the phantom rendering ({view.failures} failures)",
    )


if __name__ == "__main__":
    sys.exit(run_standalone(globals()))
