# Running one session across two machines

The rig does not fit on one computer. Arm 2 and its three cameras are on one machine, arm 1 and
the force camera on another, and they have to run **one** session between them: one state
machine, one set of instructions the patient sees, one interlock.

This is the design for that, and the first half of it is a single decision that everything else
follows from.

## One brain, one remote actuator

**Arm 1's machine is the brain.** It runs the orchestrator, the rep machine, the control loop
and the interlock. Arm 2's machine is a remote actuator: it is told "move the hook to the index
finger" and reports back whether it did.

Not the other way around, and not split down the middle. The reason is that every safety
decision this system makes is a function of the force reading:

- `control.safety.SafetyGate` decides HOLD / YIELD / RELEASE from it, every cycle;
- `session.reps.RepMachine`'s deadman stops a device-driven phase on a single missing frame;
- `session.interlock.Interlock` will not hand custody to arm 2 without a fresh, confident
  reading showing arm 1 is unloaded.

Put the brain on the machine that owns the force camera and **not one of those decisions ever
depends on the network**. Put it anywhere else and the force reading crosses the wire at frame
rate, so a dropped packet becomes a safety event thirty times a second, and the deadman starts
firing on network jitter rather than on anything about the patient.

The cameras do not argue for the other arrangement, which is worth saying because it looks like
they might. In `gauge_reader/server.py` and `hand_tracking/server.py` the **browser** owns the
camera and ships frames to Python over the same WebSocket. So "whose machine has three cameras
plugged into it" is not what decides where code runs. What decides it is the USB bus each arm's
Station talks to, because that cannot be moved.

## The seam already exists

`session.arm2.HookMover` is the whole contract between the orchestrator and arm 2:

```python
def request(self, finger: str) -> None: ...
def poll(self) -> HookOutcome | None: ...   # None while still in progress
def abort(self) -> None: ...
```

That is already a network interface. It is poll-based rather than something to await, which was
done so the orchestrator could stay a pure state machine with no clock and no I/O, and the
payoff lands here: **`session.remote_arm2.RemoteHookMover` is a fourth implementation
alongside `ManualHookMover`, and nothing above it changes.** No async refactor, no second code path, and the orchestrator's
existing tests still cover the sequencing.

`session/snapshot.py` is the other half, and it is already a wire format: a frozen dataclass
with `to_dict()`, a `SCHEMA_VERSION`, a matching TypeScript type in
`dashboard/app/lib/sessionSnapshot.ts`, and a differential test keeping the two honest
(`tests/session/test_snapshot_parity.py`). The dashboard subscribes to a stream of these from
the brain. There is no reason to invent a second message shape, and a good reason not to: the
parity test only protects the shape that exists.

## What crosses the wire

Three streams, and they are deliberately different in kind.

| From | To | What | Rate |
| --- | --- | --- | --- |
| brain | arm 2 | hook move requests, aborts, custody leases | per block |
| arm 2 | brain | move outcomes, heartbeat | per block, plus a tick |
| brain | dashboard | `SessionSnapshot` | per control cycle |

The first two are **per block, not per cycle**, which is the property that makes this safe. Arm
2 moves the hook *between* blocks and never during a pull, so the control loop and the network
never run at the same time. A partition in the middle of a repetition cannot affect that
repetition at all: the brain carries on with its own camera and its own arm, finishes the
block, and then cannot start the next one until the link is back.

## The four rules a network must not break

These are not new rules. They are the existing ones, restated for the case where the other end
might simply stop answering.

**1. A dropped connection is not a completed hook move.** `session/arm2.py` already says
success is reported and never assumed, which is why `PolicyHookMover` raises instead of
returning a plausible outcome. Over a socket the same rule means a timeout resolves to
`HookResult.FAILED` or stays `None`; there is no path where silence becomes `CONFIRMED`. A hook
reported as moved when it was not means arm 1 starts pulling on a finger the hook is not
attached to.

**2. A dropped connection is not evidence of slack.** The interlock needs positive evidence
that arm 1 is unloaded before custody passes. That evidence is the brain's own force camera, so
it is local by construction and the network cannot manufacture it. Keeping it that way is most
of why the brain goes where it goes.

**3. Arm 2 stops when it stops hearing from the brain.** The grant of custody is a **lease with
a deadline**, not a flag. The actuator holds motion only while the lease is fresh and halts
when it expires, which is the same shape as the control loop refusing to move on a stale force
reading and the rep machine's deadman stopping a blind pull. A remote arm that keeps moving
because nobody told it to stop is the networked version of exactly the failure those two
guard against.

**4. Exactly one orchestrator exists.** Two state machines stepping independently would
diverge, and the thing they would diverge about is who has custody. The actuator holds no
session state and decides nothing; it executes and reports.

## Running it

The transport needs three packages the rest of the repo does not, the same way
`gauge_reader/server.py` and `hand_tracking/server.py` do, so they are an extra install rather
than part of `requirements.txt`:

```bash
uv pip install --python .venv/bin/python fastapi uvicorn websockets
```

On arm 2's machine:

```bash
.venv/bin/python -m session.link serve --host 0.0.0.0 --port 8770
```

It prints which finger to hook and waits for a person to confirm, which is `ManualHookMover`'s
contract over a wire. No policy is trained for the hook move (`session/arm2.py`), so there is
nothing else it could honestly do yet.

On arm 1's machine, `session.link.connect_to_actuator` returns a link, and a `RemoteHookMover`
around it goes where a `ManualHookMover` goes. `GET /health` on the actuator reports the schema
version it speaks, so a mismatch is visible before a session starts rather than as a refused
message in the middle of one.

## Order to build and merge it

Each of these is a branch and a PR, and the order is by what the next one needs.

1. **`feature/coordinated-pull`**. The pull law, the `cpm` mode, the shared tuning profile,
   the example protocols. Everything below assumes `Law.PULL` exists, because a session with a
   device-driven mode is the thing being distributed. **Done, awaiting review.**
2. **`feature/two-machine-session`**. The message schema, the lease, both ends of the link and
   the WebSocket under them (`session/wire.py`, `session/remote_arm2.py`, `session/link.py`).
   The four rules above get their tests here, against a fake link and a fake clock, which is
   the only way the timing failures are reproducible on demand. **Done, awaiting review.**
3. **`feature/snapshot-stream`**. The brain serving `SessionSnapshot` over a WebSocket, and the
   dashboard rendering it instead of running its own timeline. This is where `GuideHand.tsx`
   stops counting reps off `performance.now()` and starts showing what the arm is actually
   doing. The message shape and its TypeScript type already exist; what is missing is the route
   and the subscription.
4. **`feature/dashboard-prescription`**. The clinician-facing half: load a protocol from
   `config/protocols/`, override it for one session, start the session. Needs the snapshot
   stream first so there is something to show once it starts.

## The two things code cannot finish

Neither of the merged branches above should be called done until someone has stood at the
bench. Both are implemented and tested in software, and in both cases what is untested is the
part software cannot test:

- **`cpm` has not moved a real finger.** `tests/control/test_cpm_closed_loop.py` runs a whole
  set through the real rep machine, control loop and safety gate, with only the physics
  simulated, and it is what found the one real bug in the mode. But it assumes a commanded
  position is a reached position, and it cannot see backlash or the camera dropping frames in
  bursts. Run it with a hand on the e-stop and a low cap.
- **The two-machine handover has never run with both arms powered.** The protocol round-trips
  over a real socket in the tests, but no hook has moved because of a message.
