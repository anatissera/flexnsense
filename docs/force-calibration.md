# Force calibration: servo current → newtons

Goal: find out whether arm B's servo current can stand in for a force sensor, and if so, fit the curve that converts it to a usable force signal.

Do this **first**. The admittance controller's force signal comes from here, and if the answer is no, the whole force path has to move onto the camera instead - better to know on day one.

**Units** (see AGENTS.md "Units"): you read and enter **grams** throughout this procedure, because that is what the PESOLA shows and what a human can weigh things in. The fit at the end reports both grams and **newtons**; the controller reads the newtons figure. 200 gf, the cap used throughout this doc, is **1.96 N**.

This goes through [NormaCore Station](./elrobot.md), not LeRobot - there is no LeRobot support for the ElRobot. `scripts/log_force_calibration.py` subscribes to Station's `st3215/inference` queue and reads `current_ma` per motor; it does not read `Present_Load`, because Station's own examples don't expose that field, only current, temperature, voltage and position.

## Why this might not work (read before you start)

The follower servos are confirmed (from the physical label, code `ST-3215-C018`) geared **345:1** - the same part as the SO-101 follower. External force reflected back through a high reduction ratio is divided by that ratio, and gearbox friction sits on top of it. The usual consequences:

- a **dead zone** - small external forces produce no measurable current change at all,
- **hysteresis** - current at 200 g differs depending on whether you got there going up or coming down,
- drift with motor temperature.

Feasibility estimate: 200 g at a ~20 cm moment arm is about 4 kg·cm. Against this servo's **rated (continuous) torque of 10 kg·cm at 12 V** (not the 30 kg·cm stall figure, which is a peak rating, not what the motor sustains while holding a pose) that is ~40% of rated torque - comfortably in range, current should move measurably. Stall current is rated at 2.7 A, so expect readings in the low hundreds of mA at 200 g, not fractions of a mA.

Because this is the exact same servo part as the one in the independently measured STS3215 backlash/repeatability/torque test (robonine.com), that test's numbers apply directly, not as a prior to recheck: measured backlash of ~0.87° (matches this part's own datasheet spec exactly), a firmware dead zone of 10 encoder counts (~0.88°), and - at their ~15 cm lever arm - an inability to reliably detect external forces below roughly **30-50 g (~0.3-0.5 N)**.

A 200 g (1.96 N) threshold sits comfortably above that ~30-50 g floor, so detecting the crossing is plausible - but expect a **hysteresis/softness band of several tens of grams** around whatever threshold you set, not a crisp cutoff. The up/down sweep below is what tells you the real number for this arm, not an assumed one.

## Where to get known weights

You do not need laboratory weights. In order of preference:

**1. The PESOLA itself.** It is already a calibrated 600 g instrument with 5 g graduations, and it is the thing that will be in the loop anyway. Hang any object from it, read the scale, and you now know that object's weight. You never need to know an object's true mass independently.

**2. Water.** 1 ml of water = 1.000 g, near enough at room temperature. A syringe or measuring cup plus a small bottle or zip bag gives you any value you want: 50 ml = 50 g, 100 ml = 100 g. Weigh the empty container first and subtract it. This is the most precise cheap option and it is continuously adjustable, which matters for filling in points near the threshold.

**3. A kitchen scale**, if the venue has one - then anything becomes a known weight.

**4. Coins**, as a fallback. Swiss and euro coins have standardised masses, but verify them on the PESOLA rather than trusting a table.

Aim to cover **0 to ~400 g** with points at least every 50 g, and put extra points around the 200 g threshold where accuracy actually matters.

## Rig

What you are calibrating is *force along the therapy axis*, not *weight hanging off the gripper*. The geometry has to match or the numbers are meaningless.

1. Put arm B in **the exact pose it will hold during therapy**. The current↔force mapping depends on arm configuration - the moment arm changes as the arm moves - so a calibration taken in a different pose will not transfer. This matters more on the ElRobot than it would on a 5-joint arm: 7 joints for a 1-DOF task means the "pose" includes which joints you chose to lock, see [elrobot.md](./elrobot.md).
2. Attach the PESOLA to the gripper using **the same attachment you will use with a patient**.
3. Arrange the pull so the force acts **along the direction the finger will pull**. Easiest version: orient the rig so that direction is vertical and hang the weights directly. If the therapy axis is horizontal, run a string from the PESOLA over a pulley at the table edge and hang the weights off that.
4. **Torque must be enabled** and the arm commanded to hold its pose. With torque off the motor draws no current and measures nothing.
5. **Station must already be running**, pointed at this bus (`./station --web ... --tcp ...`), and this arm's joint ranges must already be calibrated in the web UI - an uncalibrated motor's readings aren't meaningful. `scripts/calibrate.py` checks this before you start.

## Procedure

```bash
# 0. Station running, bus serial known (from the web UI), .env has STATION_HOST / FOLLOWER_BUS_SERIAL
python scripts/calibrate.py                             # confirm the bus is calibrated and all 8 motors answer
python scripts/log_force_calibration.py --probe          # confirm current readings look sane, unloaded

# 1. baseline: nothing attached, arm holding its pose
python scripts/log_force_calibration.py --tare

# 2. sweep up, then back down
python scripts/log_force_calibration.py --out data/calib_up.csv
python scripts/log_force_calibration.py --out data/calib_down.csv --direction down

# 3. fit (prints both the g/mA and the N/mA line; the controller reads the N one)
python scripts/log_force_calibration.py --fit data/calib_up.csv
```

In the logging mode the script prompts for the grams currently applied (what you read off the PESOLA), samples for a couple of seconds, and writes the mean and spread of current per motor. Workflow per point: hang the weight, let it settle, read the PESOLA, type that number, press Enter.

Run the sweep **up and then down over the same points**. Comparing the two passes is the entire point - the gap between them is the hysteresis, and it sets the floor on how precisely you can control force.

## What to look for

- **Dead zone** - the force below which current does not move. It has to be comfortably under 200 g (1.96 N) or the threshold is unenforceable.
- **Hysteresis gap** between the up and down passes, in grams. This is your realistic force resolution.
- **Noise floor** - the spread within a single static point, also in grams, via the fitted slope.
- **Which motor carries the signal.** The script logs current for all 8 motors on the bus; usually one or two dominate for a given pose and pull direction, the ones whose joint axis is most aligned with the pull. Use those, ignore the rest - and note which ones they are, it tells you which joints to lock per the redundancy decision in [elrobot.md](./elrobot.md).
- **Linearity.** Start with a straight-line fit and look at the residuals before reaching for anything fancier.

## Outcomes

**It works** (dead zone well under 200 g, hysteresis small relative to it): use current for the control loop and the camera for verification and the clinician-facing readout.

**It is marginal**: use current for the fast inner loop and let the camera correct its bias at a slower rate.

**It does not work**: force sensing moves entirely to the camera. The controller then has to run slower and more conservatively, with a tighter watchdog, because the only force signal can now be lost to glare or occlusion. Document the negative result - it is a real finding and it justifies the design that follows from it.
