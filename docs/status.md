# What has and has not touched a patient

**Nothing here has touched a patient.** This is a research prototype, and the table below is
the honest version of what that means: what has run against the bench with a finger in the hook,
what exists only in software, and what has only ever run in simulation.

| | |
|---|---|
| Admittance control, `active` mode | runs, full sets completed on the real arm |
| Coordinated three-joint pull | runs |
| Gauge reader | runs, with a known intermittent fault (below) |
| `cpm` mode | implemented and tested in software, **has not moved a real finger** |
| Two-machine handover | protocol verified over a real socket, **never run with both arms powered** |
| Arm 2 policies | one ACT ran on the arm with a mediocre result; the newer ones are untested on it |
| Isaac Lab policies | simulation only |

## Known fault: the gauge reader can invert the scale

The reader occasionally swaps the two reference stickers, which inverts the whole scale, so a
reading `g` comes out as `570 - g`. It stays inverted until tracking breaks and re-acquires.

Diagnosed to the orientation tiebreak in [`arm1/gauge_reader/stickers.py`](../arm1/gauge_reader/stickers.py).
The calibrated profile is supposed to tell the two ends of the tube apart, but on this rig it
scores both orientations identically (1.09 against 1.09), because the area ratio between the two
stickers is itself about 1.09 and the elongation and redness thresholds overlap. With the score
tied, a nearest-neighbour tiebreak decides, and whichever way it falls it then perpetuates
itself frame to frame.

It was confirmed rather than guessed: de-inverting one block of a session log turns a jumbled
trace into a clean monotonic ramp from 2.2 to 18.2 gf.

**Not yet fixed.** It is a correctness bug in the reader, not a safety hole: an inverted reading
is still a reading, so the gate still releases on a lost marker. But an inverted scale means the
cap is enforced at the wrong force, so it has to be fixed before anything approaches a patient.

## What has never run on hardware

- `cpm` mode with a real finger in the hook. The motion itself has been walked pose to pose with
  nothing attached.
- The two-machine handover with both arms powered at once. The protocol, the lease and the
  socket have been exercised end to end, but against one arm and a stub.
- The `--checkpoint` policy path in `arm1/session/link.py`, which launches arm 2's trained policy
  from the orchestrator.
