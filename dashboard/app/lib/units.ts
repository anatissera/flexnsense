/**
 * The one place on the dashboard side where the gram-force <-> newton constant lives.
 *
 * Mirror of `lib/units.py`, which is the tested side: `tests/session/test_ts_port.py` fails if
 * the constant here drifts from the Python one. Do not hardcode 9.80665 (or 9.81, or / 1000)
 * anywhere else in the dashboard; import from here.
 */
export const STANDARD_GRAVITY_M_S2 = 9.80665;

/** 1 gf is the weight of 1 g under standard gravity: N = kg * m/s^2. */
export const gramsForceToNewtons = (gramsForce: number) =>
  (gramsForce / 1000.0) * STANDARD_GRAVITY_M_S2;

export const newtonsToGramsForce = (newtons: number) =>
  (newtons * 1000.0) / STANDARD_GRAVITY_M_S2;
