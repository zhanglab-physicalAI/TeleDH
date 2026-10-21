# Architecture

## The stable path

```
              Camera (threaded V4L2, newest frame only)         camera.py
                 |  BGR frame + capture timestamp
                 v
          HandTracker.process()                                tracker.py
                 |  HandReading: 21 world landmarks (m), 21 image landmarks (px),
                 |               6 scalar signals (deg), handedness label
                 v
          FKRetargeter.__call__()                              fkretarget.py
                 |  fingertip-vector targets in the palm frame
                 |  damped Gauss-Newton over the DH116 kinematics (kinematics.py)
                 |  EMA smoothing (--smoothing) + deadband (--deadband)
                 |  6 firmware angles
                 v
          teleop loop                                          commands/teleop.py
                 |  thumb-clearing gate, tactile pinch closer (fk mode)
                 v
          Hand.write_angles()                                  hand.py
                 |  clamp to firmware limits -> host rate limit (--max-speed)
                 |  -> stall kick -> set_target_angle x6 + move_motors
                 v
          RS485 @ 500 kbaud  (sdk.py; --no-serial-flush patches out the vendor tcdrain)
                 v
               DH116
```

Everything above the hand runs once per camera frame (about 30 Hz).
`latency.py` timestamps each stage; `CAMERA->RS485` = frame_age + track +
retarget + write.

## The visualisation side path

```
   HandReading --> PhantomHand.render()   phantom.py   --> middle column of the window
                      |
                      +- Canonicalizer            handpose.py   chirality from the geometry
                      +- EMA on canonical points               (label flicker cannot flip it)
                      +- HandPose3D.from_landmarks (joint frames, when --show-joint-frames)
                      +- 2.5-D projection, painter's ordering, fog, floor shadow

   commanded angles --> HandRenderer      handview.py  --> right column (MuJoCo or convex hulls)
```

The phantom runs **after** `write_angles()` in the same iteration, owns its own
state, returns only an image, and `render()` never raises (a failing frame
yields a placeholder panel and is counted). `tests/test_control_replay.py`
asserts that rendering it between frames leaves every commanded angle
bit-identical.

`handpose.py` is a geometry module (canonical right-hand landmarks, palm frame,
minimal-rotation chain frames, wrist-local expression; the maths is in
[frames.md](frames.md)). It is stable and tested; on the stable path it is used
only by the phantom view.

The robot panel (`viz.py` + `handview.py`) is driven by the angles that
actually went out over RS485, so clamping, rate limiting and a joint pinned at
its limit are visible rather than hidden.

## Safety layers

| layer | where | default |
|---|---|---|
| firmware joint limits, read at connect | `Hand._read_limits` | - |
| host rate limit | `Hand.write_angles` | 180 deg/s (`--max-speed`) |
| firmware velocity | `Hand.connect` | 150 deg/s (see [experiments.md](experiments.md) before raising) |
| current limit | `Hand.connect` | 500 per mille (`--max-current`) |
| stall kick | `Hand._apply_stall_kick` | on |
| start paused, hold last pose on tracking loss | `commands/teleop.py` | - |
| one process holds the hardware | `session.py` (`.telehand/session.lock`) | - |
| stop + de-energize on every exit path | `Hand.disconnect` | on (`--keep-enabled` opts out) |

## Command dispatch

`run.py` calls `telehand.bootstrap.bootstrap()` (puts `vendor/` on `sys.path`,
quiets native logging) and hands over to `telehand.cli`. The parser lives in
`cli.py`; each subcommand is a module under `telehand/commands/` exposing
`run(args) -> int`, imported only when invoked, so `run.py stop` never pays for
OpenCV or MediaPipe. Commands that drive the hand or the camera run inside the
one-instance session lock; `stop`, `handview`, `cameras`, `latency` and
`frames` do not.

## Experimental components

`telehand/experimental/` is imported by the stable path only through
`experimental/cli.py`, a light module that registers the `stepresponse` and
`frames` subcommands; the modules behind them load when those commands run.
`--pose-mode frames` / `--pose-log` import `framemap` lazily inside the teleop
command. Importing `telehand.cli` therefore loads no experimental module beyond
`experimental/cli.py`.

```
experimental/framemap.py      FrameMapper: thumb/index joints from HandPose3D, FK fallback
experimental/frameviz.py      FrameStats + replay renderer for `run.py frames`
experimental/stepresponse.py  StepProbe: per-joint step response with controller-state logging
experimental/cli.py           their subcommands
```

## Where things are written

Every path is defined once, in `telehand/paths.py`.

| what | where | tracked |
|---|---|---|
| pose library | `calibration/poses.json` | yes |
| operator profiles, direct and frame calibrations | `calibration/handprofile*.json`, `direct_calibration.json`, `frames_calibration.json` | no |
| tactile maps, envelope, photos, canonical recordings | `data/` (see [data/README.md](../data/README.md)) | yes |
| benchmarks, latency CSVs, logs, screenshots | `data/benchmarks/`, `data/logs/`, `data/screenshots/` | no |
| per-machine state: camera choice, view angle, mesh cache, session lock | `.telehand/` | no |
| test fixtures | `tests/data/*.npz` | yes |
