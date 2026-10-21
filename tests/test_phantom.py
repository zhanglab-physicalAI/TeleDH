"""Phantom hand verification without hardware.

1. commanded angles identical with the view on vs off on the same replay;
2. it tracks open hand, fist, pinch and index curl (montage);
3. no chirality flips / 180-degree turns over the recording;
4. a rendering failure cannot reach the loop;
5. render cost, and the refresh throttle;
6. joint frames are attached rigidly and cost little.

    python3 tests/test_phantom.py [montage.png]
"""

import sys
import tempfile
import time
from pathlib import Path

from support import DATA_DIR, check, run_standalone

import cv2
import numpy as np

from telehand import phantom as ph
from telehand.fkretarget import FKRetargeter
from telehand.hand import JointLimit
from telehand.kinematics import FIRMWARE_MAX_DEG
from telehand.tracker import FINGER_CHAINS, HandReading, _chain_bend, extract_signals

MONTAGE_OUT = Path(tempfile.gettempdir()) / "phantom_montage.png"
if __name__ == "__main__" and len(sys.argv) > 1:
    MONTAGE_OUT = Path(sys.argv[1])

d = np.load(str(DATA_DIR / "fk_session.npz"), allow_pickle=True)
P, LAB = d["world_pts"].astype(np.float64), d["handedness"]
READINGS = [HandReading(extract_signals(p), str(label), None, p) for p, label in zip(P, LAB)]
LIMITS = [JointLimit(0.0, m) for m in FIRMWARE_MAX_DEG]
H = 480

_curl = np.array([_chain_bend(p, FINGER_CHAINS["index"]) for p in P])
_allcurl = np.array(
    [sum(_chain_bend(p, FINGER_CHAINS[f]) for f in ("index", "middle", "ring", "pinky")) for p in P]
)
_pinch = np.array([np.linalg.norm(p[4] - p[8]) for p in P])
# Skip the first second: the hand is still entering the view there.
_S = 30
I_OPEN, I_FIST = _S + int(np.argmin(_allcurl[_S:700])), _S + int(np.argmax(_allcurl[_S:700]))
I_PINCH = _S + int(np.argmin(_pinch[_S:700]))
I_CURL = _S + int(np.argmax(_curl[_S:700] - 0.3 * (_allcurl[_S:700] - _curl[_S:700])))


def _fk():
    return FKRetargeter(LIMITS, smoothing=0.35, deadband_deg=0.8)


def _render_cost_ms(view) -> float:
    t0 = time.perf_counter()
    for r in READINGS:
        view.render(r, H)
    return (time.perf_counter() - t0) / len(READINGS) * 1000


def test_depth_convention():
    fist = P[I_FIST]
    tips_z = fist[list(ph.TIPS)][:, 2].mean()
    mcp_z = fist[[5, 9, 13, 17]][:, 2].mean()
    print(
        f"       fist frame {I_FIST}: mean world z of tips {tips_z:+.4f}, of knuckles {mcp_z:+.4f} "
        f"(MediaPipe: smaller z = closer to the camera)"
    )
    view = ph.PhantomHand()
    scene = view.scene_points(READINGS[I_FIST])
    check(
        scene[list(ph.TIPS)][:, 2].mean() > scene[[5, 9, 13, 17]][:, 2].mean(),
        "curled fingertips come out nearer the viewer than the knuckles (scene z toward viewer)",
    )


def test_control_path_invariance():
    fk_off, fk_on = _fk(), _fk()
    view = ph.PhantomHand(hz=0)  # hz=0: render every call (the throttle is tested separately)
    a_off, a_on = [], []
    for r in READINGS:
        a_off.append(fk_off(r.world_pts, "Left"))
        a = fk_on(r.world_pts, "Left")
        view.render(r, H)  # as in the display block, after the command
        a_on.append(a)
    a_off, a_on = np.array(a_off), np.array(a_on)
    check(
        np.array_equal(a_off, a_on),
        f"commanded angles identical on all {len(P)} frames (max |diff| {np.abs(a_off - a_on).max():.1e})",
    )
    check(
        view.failures == 0 and view.rendered == len(P),
        f"rendered every frame without failure ({view.rendered}/{len(P)})",
    )


def test_chirality_over_recording():
    view = ph.PhantomHand()
    prev_n = prev_pts = None
    flips, jumps = [], []
    for i, r in enumerate(READINGS):
        s = view.scene_points(r)
        y0 = s[5] - s[17]
        z0 = s[[5, 9, 13]].mean(0) - s[0]
        n = np.cross(y0, z0)
        n /= np.linalg.norm(n)
        if prev_n is not None:
            if np.dot(n, prev_n) < 0:
                flips.append(i)
            jumps.append(float(np.linalg.norm(s - prev_pts, axis=1).max() * 1000))
        prev_n, prev_pts = n, s
    jumps = np.array(jumps)
    check(
        len([f for f in flips if f < 728]) == 0,
        f"palm-normal sign never flips on frames 1..727 (flips at {flips or 'none'}; the recording's own "
        f"landmark glitch is at 728-729)",
    )
    check(
        view.canon.counts["switches"] == 1,
        f"the one genuine geometry flip (frame 728) was absorbed by the canonicalizer "
        f"({view.canon.counts['switches']} switch)",
    )
    print(
        f"       largest per-frame joint move: p95 {np.percentile(jumps, 95):.1f} mm, "
        f"max {jumps.max():.1f} mm at frame {int(jumps.argmax()) + 1}; "
        f"frames above 30 mm: {[int(j) + 1 for j in np.where(jumps > 30)[0]]}"
    )


def test_rendering_failure_cannot_reach_the_loop():
    bad = ph.PhantomHand(hz=0)

    def fail(*a, **k):
        raise RuntimeError("synthetic render failure")

    bad._draw_scene = fail
    raised = False
    imgs = []
    try:
        imgs = [bad.render(r, H) for r in READINGS[:40]]
    except Exception:
        raised = True
    check(
        not raised and bad.failures == 40 and all(im.shape == (H, bad.width, 3) for im in imgs),
        f"40 failing frames swallowed, counted ({bad.failures}), placeholder panels returned",
    )
    blank = ph.PhantomHand(hz=0)
    im = blank.render(None, H)
    check(
        im.shape == (H, blank.width, 3) and blank.failures == 0, "no reading -> 'no hand' panel, no failure"
    )


def test_render_cost_and_throttle():
    view = ph.PhantomHand(hz=0)
    ms = _render_cost_ms(view)
    check(ms < 4.0, f"{ms:.2f} ms/frame at {view.width}x{H} (all 21 joints, 23 bones, grid, shadow, fog)")
    slow = ph.PhantomHand(hz=15.0)
    t = time.perf_counter()
    for r in READINGS[:60]:
        slow.render(r, H)
        while time.perf_counter() - t < 1 / 30.0:
            pass
        t = time.perf_counter()
    check(
        20 <= slow.rendered <= 40,
        f"hz=15 at a 30 Hz feed re-renders about every other frame ({slow.rendered}/60)",
    )


def test_joint_frames():
    v = ph.PhantomHand(hz=0, joint_frames=True)
    worst = 0.0
    n_axes = 0
    for r in READINGS[30:700:7]:
        v.render(r, H)
        pose = v.last_pose
        assert pose is not None
        pts = ph.PhantomHand.to_scene(v._smooth_canon)
        for name, j in pose.joints.items():
            if name == "palm" or name.endswith("_tip"):
                continue
            child = next(c for c in pose.joints.values() if c.parent == name)
            bone = pts[child.landmark] - pts[j.landmark]
            bone /= np.linalg.norm(bone)
            z_scene = ph.DIR_TO_SCENE * (pose.wrist_rotation @ j.rotation)[:, 2]
            worst = max(worst, 1.0 - float(np.dot(bone, z_scene)))
            n_axes += 1
    check(
        worst < 1e-9,
        f"every joint's drawn z axis lies along its child bone in scene space "
        f"({n_axes} joints, worst 1-dot {worst:.1e})",
    )
    check(
        v.frame_failures == 0 and v.failures == 0, f"joint frames built on every frame ({v.rendered} renders)"
    )
    t_off = _render_cost_ms(ph.PhantomHand(hz=0))
    t_on = _render_cost_ms(ph.PhantomHand(hz=0, joint_frames=True))
    check(t_on - t_off < 3.0, f"joint frames add {t_on - t_off:.2f} ms/render ({t_off:.2f} -> {t_on:.2f} ms)")
    tg = ph.PhantomHand(hz=0)
    tg.render(READINGS[100], H)
    a = tg.last_pose
    tg.toggle_frames()
    tg.render(READINGS[101], H)
    b = tg.last_pose
    tg.toggle_frames()
    tg.render(READINGS[102], H)
    c = tg.last_pose
    check(a is None and b is not None and c is None, "f toggle switches joint frames on and off")
    fk1, fk2 = _fk(), _fk()
    pv = ph.PhantomHand(hz=0, joint_frames=True)
    o1, o2 = [], []
    for r in READINGS:
        o1.append(fk1(r.world_pts, "Left"))
        o2.append(fk2(r.world_pts, "Left"))
        pv.render(r, H)
    check(np.array_equal(np.array(o1), np.array(o2)), "commanded angles identical with joint frames on")


def test_representative_poses_montage():
    picks = [("open hand", I_OPEN), ("fist", I_FIST), ("pinch", I_PINCH), ("index curl", I_CURL)]
    tiles = []
    for label, i in picks:
        v = ph.PhantomHand(hz=0, joint_frames=True)
        for k in range(max(0, i - 20), i):
            v.render(READINGS[k], H)
        img = v.render(READINGS[i], H).copy()
        cv2.putText(
            img, f"#{i} {label}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        tiles.append(img)
        check(v.failures == 0 and v.frame_failures == 0, f"{label} (frame {i}) rendered with joint frames")
    montage = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(str(MONTAGE_OUT), montage)
    print(f"       montage -> {MONTAGE_OUT}")


if __name__ == "__main__":
    sys.exit(run_standalone(globals()))
