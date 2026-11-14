# Force calibration: servo current → grams

Goal: find out whether arm B's servo current can stand in for a force sensor, and if so, fit the curve that converts it to grams.

Do this **first**. The admittance controller's force signal comes from here, and if the answer is no, the whole force path has to move onto the camera instead — better to know on day one.

## Why this might not work (read before you start)

The SO-101 follower servos are geared **1/345**. External force reflected back through a high reduction ratio is divided by that ratio, and gearbox friction sits on top of it. The usual consequences:

- a **dead zone** — small external forces produce no measurable current change at all,
- **hysteresis** — current at 200 g differs depending on whether you got there going up or coming down,
- drift with motor temperature.

Rough feasibility estimate: 200 g at a ~20 cm moment arm is about 4 kg·cm, against 15 kg·cm stall torque for the 7.4 V STS3215 (30 kg·cm for the 12 V). That is 13–27 % of stall, so `Present_Load` (0–1000 = 0–100 %) should read in the low hundreds. Detectable in principle. The up/down sweep below is what tells you whether friction ruins it in practice.

## Where to get known weights

You do not need laboratory weights. In order of preference:

**1. The PESOLA itself.** It is already a calibrated 600 g instrument with 5 g graduations, and it is the thing that will be in the loop anyway. Hang any object from it, read the scale, and you now know that object's weight. You never need to know an object's true mass independently.

**2. Water.** 1 ml of water = 1.000 g, near enough at room temperature. A syringe or measuring cup plus a small bottle or zip bag gives you any value you want: 50 ml = 50 g, 100 ml = 100 g. Weigh the empty container first and subtract it. This is the most precise cheap option and it is continuously adjustable, which matters for filling in points near the threshold.

**3. A kitchen scale**, if the venue has one — then anything becomes a known weight.

**4. Coins**, as a fallback. Swiss and euro coins have standardised masses, but verify them on the PESOLA rather than trusting a table.

Aim to cover **0 to ~400 g** with points at least every 50 g, and put extra points around the 200 g threshold where accuracy actually matters.

## Rig

What you are calibrating is *force along the therapy axis*, not *weight hanging off the gripper*. The geometry has to match or the numbers are meaningless.

1. Put arm B in **the exact pose it will hold during therapy**. The current↔force mapping depends on arm configuration — the moment arm changes as the arm moves — so a calibration taken in a different pose will not transfer.
2. Attach the PESOLA to the gripper using **the same attachment you will use with a patient**.
3. Arrange the pull so the force acts **along the direction the finger will pull**. Easiest version: orient the rig so that direction is vertical and hang the weights directly. If the therapy axis is horizontal, run a string from the PESOLA over a pulley at the table edge and hang the weights off that.
4. **Torque must be enabled** and the arm commanded to hold its pose. With torque off the motor draws no current and measures nothing.

## Procedure

```bash
# 0. ports in .env, arm connected and powered
python scripts/log_force_calibration.py --probe        # confirm which registers are readable

# 1. baseline: nothing attached, arm holding its pose
python scripts/log_force_calibration.py --tare

# 2. sweep up, then back down
python scripts/log_force_calibration.py --out data/calib_up.csv
python scripts/log_force_calibration.py --out data/calib_down.csv --direction down

# 3. fit
python scripts/log_force_calibration.py --fit data/calib_up.csv
```

In the logging mode the script prompts for the grams currently applied, samples for a couple of seconds, and writes the mean and spread per motor. Workflow per point: hang the weight, let it settle, read the PESOLA, type that number, press Enter.

Run the sweep **up and then down over the same points**. Comparing the two passes is the entire point — the gap between them is the hysteresis, and it sets the floor on how precisely you can control force.

## What to look for

- **Dead zone** — the force below which current does not move. It has to be comfortably under 200 g or the threshold is unenforceable.
- **Hysteresis gap** between the up and down passes, in grams. This is your realistic force resolution.
- **Noise floor** — the spread within a single static point, also in grams, via the fitted slope.
- **Which joint carries the signal.** Log all six motors; usually one or two dominate for a given pose and pull direction. Use those, ignore the rest.
- **Linearity.** Start with a straight-line fit and look at the residuals before reaching for anything fancier.

## Outcomes

**It works** (dead zone well under 200 g, hysteresis small relative to it): use current for the control loop and the camera for verification and the clinician-facing readout.

**It is marginal**: use current for the fast inner loop and let the camera correct its bias at a slower rate.

**It does not work**: force sensing moves entirely to the camera. The controller then has to run slower and more conservatively, with a tighter watchdog, because the only force signal can now be lost to glare or occlusion. Document the negative result — it is a real finding and it justifies the design that follows from it.
