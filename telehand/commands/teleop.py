"""``teleop``: live teleoperation in any of the four retargeting modes.

The stable pipeline is

    Camera -> HandTracker (MediaPipe) -> FKRetargeter -> Hand.write_angles

with ``Hand`` applying the firmware limits, the host rate limit and the stall
kick. The phantom 3-D hand view (``--show-3d-pose``) is a side path: it reads
the same ``HandReading`` after the command has gone out and never feeds back
into control. The frame-based mapping behind ``--pose-mode frames`` is
experimental and is imported only when asked for.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .. import direct as _direct
from .. import latency as _latency
from .. import phantom as _phantom
from .. import profile as _profile
from .. import viz
from ..camera import Camera
from ..fkretarget import FKRetargeter, PinchCloser
from ..hand import JOINT_NAMES, NUM_JOINTS, Hand
from ..poses import PoseLibrary, PoseRetargeter
from ..tracker import SIGNAL_NAMES, HandTracker


def run(args) -> int:
    mode = args.mode
    if args.pose_mode == "frames" and mode != "fk":
        print(
            "ERROR: --pose-mode frames needs --mode fk (middle/ring/pinky and every fallback "
            "come from the FK solver).",
            file=sys.stderr,
        )
        return 1
    library = None
    if mode in ("poses", "hybrid"):
        library = PoseLibrary.load(Path(args.poses))
        if not args.allow_estimates:
            skipped = [a.name for a in library.anchors if a.complete and not a.measured]
            if skipped:
                print(
                    "Excluding poses whose robot joints were never measured: "
                    + ", ".join(skipped)
                    + "  (measure them with `run.py jog`, "
                    "or pass --allow-estimates)"
                )
                library.anchors = [a for a in library.anchors if a.measured]
        ready = library.complete_anchors
        if len(ready) < 2:
            print(f"ERROR: {args.poses} has fewer than two calibrated poses.")
            print("Run:  python3 run.py calibrate --hand " + args.hand)
            return 1
        print(f"Pose library: {len(ready)} anchors - " + ", ".join(a.name for a in ready))
        estimates = [a.name for a in ready if not a.measured]
        if estimates:
            print(
                "NOTE: robot joints are estimates for "
                + ", ".join(estimates)
                + "; measure them with `run.py jog` if those poses feel off."
            )
    direct_calib = None
    if mode in ("direct", "hybrid"):
        direct_calib = _direct.DirectCalibration.load(Path(args.direct_calibration))
        state = (
            "calibrated"
            if direct_calib.calibrated
            else "DEFAULT ranges - press 1 (open) / 2 (fist) to capture"
        )
        print(f"Direct mode ranges: {state}")
    print(f"Mode: {mode}")

    cap = Camera(args.camera)
    tracker = HandTracker(Path(args.model), prefer_hand=args.hand)

    hand = Hand(
        args.port,
        args.baud,
        dry_run=args.dry_run,
        keep_enabled=args.keep_enabled,
        stall_kick=args.stall_kick,
        max_deg_per_s=args.max_speed,
        max_current=args.max_current,
        skip_unchanged=args.skip_unchanged,
        skip_epsilon=args.skip_epsilon,
        resync_frames=args.resync_frames,
        no_serial_flush=args.no_serial_flush,
    )
    print(
        f"Serial: no-flush {'ON' if args.no_serial_flush else 'off'}, "
        f"skip-unchanged {'ON' if args.skip_unchanged else 'off'} "
        f"(eps {args.skip_epsilon:.2f} deg, resync every {args.resync_frames} writes)"
    )

    print(f"Connecting to hand on {args.port} ...")
    if args.home and not args.dry_run:
        print("Homing: the hand will move through its full range. Keep it clear.")
    hand.connect(home=args.home)
    print("Connected.")
    if args.dry_run:
        print("DRY RUN: tracking runs but no motion is commanded.")

    pose_rt = None
    if library is not None:
        pose_rt = PoseRetargeter(
            hand.limits, library, smoothing=args.smoothing, deadband_deg=args.deadband, power=args.blend_power
        )
    prof = _profile.HandProfile.load(Path(args.profile)) if mode == "fk" else None
    if mode == "fk":
        print(
            "Hand profile: "
            + (
                f"{args.profile}" + (f" ({prof.name})" if prof and prof.name else "")
                if prof
                else "none - run `run.py profile` for per-finger scaling"
            )
        )
    fk_rt = (
        FKRetargeter(hand.limits, smoothing=args.smoothing, deadband_deg=args.deadband, profile=prof)
        if mode == "fk"
        else None
    )
    intent = (prof.pinch_gap * 1000 + 12.0) if (prof and prof.pinch_gap) else 35.0
    closer = PinchCloser(intent_mm=intent) if (mode == "fk" and args.tactile and not args.dry_run) else None

    # Experimental frame-based mapping: imported and built only when asked
    # for, so the stable path never loads it. Built whenever --pose-log is on
    # as well, so both pose modes log the same columns.
    frame_mapper = pose_tracker = canon = frame_capture = frame_calib = None
    pose_log = None
    _framemap = _hp = None
    if mode == "fk" and (args.pose_mode == "frames" or args.pose_log):
        from .. import handpose as _hp
        from ..experimental import framemap as _framemap

        frame_calib = _framemap.FrameCalibration.load(Path(args.frames_calibration))
        frame_mapper = _framemap.FrameMapper(
            hand.limits,
            frame_calib,
            smoothing=args.smoothing,
            deadband_deg=args.deadband,
            abd_signal=args.thumb_abd,
            min_conf=args.frames_min_conf,
        )
        pose_tracker = _hp.HandPoseTracker(("thumb", "index"), history=args.frames_history)
        canon = _hp.Canonicalizer(assume_mirrored=args.mirror)
        frame_capture = _framemap.FrameRangeCapture()
        print(
            f"Pose mode: {args.pose_mode.upper()}"
            + (
                "  (thumb abd/flex + index from HandPose3D via "
                f"{args.thumb_abd}/curl; middle/ring/pinky and fallbacks from FK)"
                if args.pose_mode == "frames"
                else "  (FK solver; frame signals computed and logged only)"
            )
        )
        print(
            "Frame ranges: "
            + (
                "calibrated " + str(args.frames_calibration)
                if frame_calib.calibrated
                else "DEFAULTS - press 1 (open) / 2 (fist) to capture your own"
            )
        )
        if args.pose_log:
            pose_log = open(args.pose_log, "w")
            pose_log.write(_framemap.log_header())
            print(f"Pose log: {args.pose_log}")
    last_fm = [None]

    # Phantom 3-D hand: visualisation only. Its own canonicalizer, rendered
    # after the command has gone out, never consulted by control.
    phantom = (
        _phantom.PhantomHand(
            assume_mirrored=args.mirror, hz=args.pose_hz, joint_frames=args.show_joint_frames
        )
        if args.show_3d_pose
        else None
    )
    print(
        "3D hand view: "
        + (
            f"ON, joint frames {'ON' if args.show_joint_frames else 'off'}  "
            "(p toggles view, f toggles frames, [ ] rotate; visualisation only, "
            "control path unchanged)"
            if phantom
            else "off (press p to enable)"
        )
    )
    if closer:
        hand._sdk.set_sensor_enable(True)
        time.sleep(0.8)
        hand._sdk.set_finger_pressure_reset()
        print("Tactile pinch closure: ON (thumb-tip sensor + finger stall)")
    direct_rt = smoother = capture = None
    if direct_calib is not None:
        direct_rt = _direct.DirectRetargeter(hand.limits, direct_calib)
        smoother = _direct.LandmarkSmoother(
            alpha=args.lm_alpha, deadband=args.lm_deadband, use_kalman=not args.no_kalman
        )
        capture = _direct.RangeCapture()

    def reset_filters():
        if pose_rt:
            pose_rt.reset()
        if fk_rt:
            fk_rt.reset()
        if closer:
            closer.reset()
        if smoother:
            smoother.reset()
        if frame_mapper:
            frame_mapper.reset()
        if pose_tracker:
            pose_tracker.reset()

    diag = {"nearest": None, "dist": None, "top": []}
    # Latency instrumentation. Timing is always collected (perf_counter is
    # ~50 ns, against a 33 ms frame period); the CSV is only written on request.
    lat = _latency.LatencyLog(
        Path(args.latency_csv) if args.latency_csv else None, extra=["tracked", "engaged", "tx"]
    )
    if args.latency_csv:
        print(f"Latency CSV: {args.latency_csv}")
    rec = open(args.record, "w") if args.record else None
    rec_pts = [] if args.record else None
    rec_2d = [] if args.record else None  # parallel to rec_pts; filled after write_angles
    if rec:
        rec.write("t," + ",".join(SIGNAL_NAMES) + ",nearest,dist," + ",".join(JOINT_NAMES) + "\n")

    def compute_angles(reading, frame_shape):
        """Joint angles for the current mode; None if this mode has nothing yet."""
        pose_angles = pose_rt(reading.signals) if pose_rt else None
        if rec_pts is not None and reading.world_pts is not None:
            rec_pts.append((time.monotonic() - t0, reading.handedness, reading.world_pts.copy()))
        if pose_rt:
            w = pose_rt.weights(reading.signals)
            order = np.argsort(w)[::-1]
            sig = (np.asarray(reading.signals) - pose_rt._lo) / pose_rt._span
            dists = np.linalg.norm(pose_rt._norm - sig, axis=1)
            k = int(np.argmin(dists))
            diag["nearest"], diag["dist"] = pose_rt.names[k], float(dists[k])
            diag["top"] = [(pose_rt.names[i], float(w[i])) for i in order[:2]]
            if rec and pose_angles is not None:
                rec.write(
                    f"{time.monotonic() - t0:.3f},"
                    + ",".join(f"{v:.2f}" for v in reading.signals)
                    + f",{diag['nearest']},{diag['dist']:.3f},"
                    + ",".join(f"{v:.1f}" for v in pose_angles)
                    + "\n"
                )
        direct_angles = None
        if direct_rt and reading.landmarks_2d is not None:
            h, w = frame_shape[:2]
            norm = reading.landmarks_2d / np.array([w, h], dtype=np.float64)
            sm = smoother(norm)
            sig = _direct.bends_to_signals(_direct.finger_bends(sm))
            done = capture.feed(sig, direct_calib)
            if done:
                direct_calib.save(Path(args.direct_calibration))
                print(f"direct: captured {done} pose -> {args.direct_calibration}")
            direct_angles = direct_rt(sig)
        if mode == "fk":
            if reading.world_pts is None:
                return None
            # Chirality decides whether the palm normal is mirrored. Trust the
            # operator's --hand over MediaPipe's label, which has disagreed with
            # the operator in every session so far.
            # The frame is mirrored for display, so the landmark geometry is of
            # the opposite hand: a physical right hand arrives as a left one.
            geom = args.hand if not args.mirror else ("Left" if args.hand == "Right" else "Right")
            out = list(fk_rt(reading.world_pts, geom))
            if frame_mapper is not None:
                # Canonical right-hand landmarks -> wrist-local frames -> three
                # joints. Computed in both pose modes so the log always has
                # both signals; only frames mode sends the result.
                canon_pts = canon(reading.world_pts, reading.handedness)
                hp_pose = pose_tracker.update(canon_pts, int((time.monotonic() - t0) * 1000))
                fm = frame_mapper.map(hp_pose, out, canon)
                if frame_capture.active:
                    done = frame_capture.feed(fm.signals, frame_calib)
                    if done:
                        frame_calib.save(Path(args.frames_calibration))
                        print(
                            f"frames: captured {done} pose -> {args.frames_calibration}: "
                            + "  ".join(
                                f"{k} {frame_calib.ranges[k][0]:.0f}..{frame_calib.ranges[k][1]:.0f}"
                                for k in _framemap.FRAME_SIGNALS
                            )
                        )
                if args.pose_mode == "frames":
                    out = list(fm.angles)
                    diag["nearest"] = "frames " + "".join(fm.source)
                    diag["dist"] = fm.conf_index
                    diag["top"] = [(fm.reason or "ok", 1.0)]
                last_fm[0] = fm
            # The thumb abduction joint is slow to let go. While it still sits
            # more than a few degrees across from where it was sent, the fingers
            # hold their current curl instead of closing into its path.
            if not args.dry_run:
                meas = hand.read_angles()
                if meas[0] > out[0] + 6.0:
                    for i in range(2, 6):
                        out[i] = min(out[i], max(meas[i], 0.0))
                    diag["top"] = [("thumb clearing", 1.0)]
            if closer:
                tgt, _ = fk_rt.targets(np.asarray(reading.world_pts, dtype=np.float64), geom)
                dist_mm = [float(np.linalg.norm(v)) * 1000 for v in tgt[:4]]
                try:
                    pressure = max(hand._sdk.get_finger_pressure(1) or [0.0])
                except Exception:
                    pressure = 0.0
                sent = list(out)
                for i in range(4):
                    sent[2 + i] = hand.limits[2 + i].clamp(out[2 + i] + closer.extra[i])
                extra = closer.update(dist_mm, sent, hand.read_angles(), pressure)
                for i in range(4):
                    out[2 + i] = hand.limits[2 + i].clamp(out[2 + i] + extra[i])
                diag["nearest"] = "pinch:" + ",".join(
                    f"{'*' if closer.contact[i] else ''}{extra[i]:.0f}" for i in range(4)
                )
                diag["dist"] = min(dist_mm) / 100.0
                diag["top"] = [(f"p={pressure:.2f}", 1.0)]
            return out
        if mode == "poses":
            return pose_angles
        if mode == "direct":
            return direct_angles
        # hybrid: vendor-style linear map for the four fingers, our pinch-aware
        # pose blend for the two thumb joints.
        if direct_angles is None or pose_angles is None:
            return direct_angles or pose_angles
        return pose_angles[:2] + direct_angles[2:]

    view = viz.ViewControl()
    engaged = False
    lost_frames = 0
    alarm_state = [0] * NUM_JOINTS
    frame_no = 0
    mouse_attached = False
    warned_handedness = False
    t0 = time.monotonic()
    last_t = t0
    fps = 0.0
    angles: list[float] | None = None
    headless = args.no_display

    if headless:
        engaged = True
        print("Headless mode: engaging immediately. Ctrl+C to stop.")
    else:
        print("Press SPACE in the video window to engage.")

    try:
        while True:
            timer = _latency.FrameTimer()
            # The loop clock starts before the blocking read, so `loop` is the
            # true iteration period and 1000/median is the real frame rate.
            loop_t0 = time.perf_counter()
            with timer.stage("wait"):
                ok, frame = cap.read()
            if not ok:
                continue
            # How stale this frame already was when the loop received it.
            timer.mark("frame_age", (time.monotonic() - cap.frame_stamp) * 1000.0)
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            with timer.stage("track"):
                reading = tracker.process(frame, int((now - t0) * 1000))

            # Alarms come over the same RS485 bus as the motion commands, so
            # poll them a few times a second rather than every frame.
            frame_no += 1
            if frame_no % 15 == 0:
                alarm_state = hand.alarms()

            hand.last_tx_count = 0
            if reading is not None:
                if not warned_handedness and reading.handedness != args.hand:
                    print(
                        f"NOTE: tracking your {reading.handedness} hand "
                        f"(--hand is {args.hand}). Run `check` for details."
                    )
                    warned_handedness = True
                lost_frames = 0
                with timer.stage("retarget"):
                    new_angles = compute_angles(reading, frame.shape)
                if new_angles is not None:
                    angles = new_angles
                    if engaged:
                        with timer.stage("write"):
                            angles = hand.write_angles(angles)
                if pose_log is not None and last_fm[0] is not None:
                    pose_log.write(
                        _framemap.log_row(
                            now - t0, args.pose_mode, engaged, reading.signals, last_fm[0], canon, angles
                        )
                    )
                    last_fm[0] = None
                if rec_2d is not None and reading.world_pts is not None:
                    rec_2d.append(
                        (
                            reading.landmarks_2d.copy()
                            if reading.landmarks_2d is not None
                            else np.full((21, 2), np.nan, dtype=np.float32),
                            frame.shape[:2],
                        )
                    )
            else:
                lost_frames += 1
                # Tracking dropped out: hold the last commanded pose rather than
                # letting the hand snap anywhere.
                if lost_frames > args.lost_timeout_frames:
                    reset_filters()

            if headless:
                timer.mark("loop", (time.perf_counter() - loop_t0) * 1000.0)
                lat.add(
                    now - t0,
                    frame_no,
                    timer,
                    tx=hand.last_tx_count,
                    tracked=reading is not None,
                    engaged=engaged,
                )
                continue

            display_t0 = time.perf_counter()
            viz.draw_skeleton(frame, reading.landmarks_2d if reading else None)
            phantom_img = None
            if phantom is not None and phantom.enabled:
                with timer.stage("overlay"):
                    phantom_img = phantom.render(reading, height=frame.shape[0])
            if args.diag and diag["nearest"]:
                top = "  ".join(f"{n} {w:.0%}" for n, w in diag["top"])
                txt = f"nearest {diag['nearest']}  d={diag['dist']:.2f}   {top}"
                cv2.rectangle(
                    frame, (0, frame.shape[0] - 26), (frame.shape[1], frame.shape[0]), (30, 28, 26), -1
                )
                cv2.putText(
                    frame,
                    txt,
                    (8, frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (90, 200, 90) if diag["dist"] < 0.35 else (60, 170, 235),
                    1,
                    cv2.LINE_AA,
                )
            panel = viz.render_panel(
                frame.shape[0],
                angles if angles is not None else [lim.min_angle for lim in hand.limits],
                hand.limits,
                engaged=engaged,
                fps=fps,
                alarms=alarm_state,
                tracked=reading is not None,
                dry_run=args.dry_run,
                title=f"{'ENGAGED' if engaged else 'PAUSED'}  [{mode}]",
                measured=(hand.read_angles() if args.show_measured and not args.dry_run else None),
                **view.kwargs,
            )
            # Camera | phantom hand | robot panel. The robot view's mouse
            # rectangle is keyed to what sits left of it, so keep that current.
            left = frame if phantom_img is None else np.hstack([frame, phantom_img])
            view.camera_width = left.shape[1]
            composed = viz.compose(left, panel)
            cv2.imshow("telehand", composed)
            if not mouse_attached:
                view.attach("telehand")
                mouse_attached = True
            key = cv2.waitKey(1) & 0xFF
            timer.mark("display", (time.perf_counter() - display_t0) * 1000.0)
            timer.mark("loop", (time.perf_counter() - loop_t0) * 1000.0)
            lat.add(
                now - t0, frame_no, timer, tx=hand.last_tx_count, tracked=reading is not None, engaged=engaged
            )
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                engaged = not engaged
                reset_filters()
                print("ENGAGED" if engaged else "PAUSED")
            elif key == ord("o"):
                engaged = False
                hand.open_hand()
                print("opened hand, disengaged")
            elif key == ord("c"):
                hand.clear_alarms()
                print("alarms cleared")
            elif key == ord("p"):
                if phantom is None:
                    phantom = _phantom.PhantomHand(
                        assume_mirrored=args.mirror, hz=args.pose_hz, joint_frames=args.show_joint_frames
                    )
                else:
                    phantom.toggle()
                print(f"3D hand view {'ON' if phantom.enabled else 'off'}")
            elif key == ord("f") and phantom is not None:
                print(f"joint frames {'ON' if phantom.toggle_frames() else 'off'}")
            elif key in (ord("["), ord("]")) and phantom is not None:
                phantom.yaw += -10.0 if key == ord("[") else 10.0
            elif view.handle_key(key):
                pass
            elif key in (ord("1"), ord("2")) and capture is not None:
                which = key - ord("1")
                capture.start(which)
                which_pose = "OPEN hand" if which == 0 else "FIST"
                print(f"direct: hold the {which_pose} still, averaging 20 frames ...")
            elif key in (ord("1"), ord("2")) and frame_capture is not None:
                which = key - ord("1")
                frame_capture.start(which)
                which_pose = "OPEN hand, thumb out" if which == 0 else "FIST, thumb across"
                print(f"frames: hold the {which_pose} still, averaging 20 frames ...")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("Stopping...")
        cap.release()
        tracker.close()
        if phantom is not None:
            print(phantom.summary())
        if pose_log is not None:
            pose_log.close()
            print(
                f"pose log written to {args.pose_log}"
                + (
                    f"  (frame mapper fell back on {frame_mapper.fallbacks} of {frame_mapper.frames} frames)"
                    if frame_mapper
                    else ""
                )
            )
        if rec:
            rec.close()
            print(f"signals recorded to {args.record}")
            if rec_pts:
                npz = Path(args.record).with_suffix(".npz")
                np.savez_compressed(
                    npz,
                    t=np.array([p[0] for p in rec_pts]),
                    handedness=np.array([p[1] for p in rec_pts]),
                    world_pts=np.stack([p[2] for p in rec_pts]),
                    landmarks_2d=np.stack([q[0] for q in rec_2d]),
                    frame_size=np.array(rec_2d[-1][1]),
                )
                print(f"world landmarks recorded to {npz} ({len(rec_pts)} frames)")
        if not headless:
            cv2.destroyAllWindows()
        hand.disconnect()
        print("Disconnected.")
        lat.close()
        print(lat.summary())
        if lat.path:
            print(f"  latency CSV: {lat.path}")
    return 0
