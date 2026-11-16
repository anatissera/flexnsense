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

## What actually goes over the socket

Four messages, and no others (`session.wire.MessageKind`). Every one is JSON, one object per
WebSocket text frame, on the `/arm2` route. These are the literal bytes, produced by
`session/wire.py` rather than transcribed by hand:

**Brain to actuator: move the hook.** This is the whole command. There is no "run this script"
and no shell: the actuator is told *which finger*, and it owns how that happens on its own
Station and its own arm.

```json
{"schema": 1, "kind": "hook_request", "finger": "index",
 "lease": {"move_id": 1, "issued_at": 1234.5, "period_s": 0.5}}
```

**Brain to actuator: the lease is still good.** Sent every cycle while the move is in flight.

```json
{"schema": 1, "kind": "heartbeat",
 "lease": {"move_id": 1, "issued_at": 1234.8, "period_s": 0.5}}
```

**Brain to actuator: stop.**

```json
{"schema": 1, "kind": "abort", "move_id": 1, "reason": "operator stopped the session"}
```

**Actuator to brain: what happened.** The only message that may complete a move. `result` is
`confirmed` or `failed`, and nothing else: there is no "probably".

```json
{"schema": 1, "kind": "hook_report", "move_id": 1, "result": "confirmed",
 "finger": "index", "detail": "confirmed by operator"}
```

### Why there is no "run this command" message

The obvious design is for the brain to send the actuator a command line to execute. This
protocol deliberately cannot do that, and the reason is the lease.

A shell command, once started, runs until it finishes. If the brain dies halfway through, the
far arm keeps going with nobody able to stop it, which is the networked form of the failure
`control.safety.SafetyGate` and `session.reps.RepMachine._deadman` both exist to prevent. So
the brain does not send *actions*, it sends an **authorisation with a deadline**: a finger to
hook, valid for `period_s` seconds, renewed by heartbeats for as long as the brain can still
see that renewing is safe. The actuator halts the moment that lease expires, so silence means
stop, which is the only safe meaning silence can have.

What the actuator does with the authorisation, which Station commands it writes to which
motors, is entirely its own. It holds no session state and decides nothing about the session
(rule 4 above); it executes and reports.

### The lease is timed by the receiver, not by comparing two clocks

Worth knowing because it was a real bug here, found by running the protocol against the
actuator and watching the arm do nothing.

`Lease.issued_at` is the brain's `time.monotonic()`. Two machines' monotonic clocks have **no
common origin**: each counts from its own boot. Comparing the brain's `issued_at` against the
actuator's own clock therefore does not ask "has the lease expired", it asks "which machine
booted first", and the answer is constant for a whole session:

  * actuator booted first (its clock reads smaller): every lease looks fresh for ever, which
    is precisely the flag-not-a-lease failure this design exists to prevent;
  * brain booted first: every lease looks long expired and the arm refuses to move at all.
    That is the harmless direction, and it is what this bench hit.

So the receiver times the lease on **its own** clock, from when the message arrived:
`Lease.expires_once_received_at(received_at)`, and `ActuatorSide` keeps the arrival time
beside the lease. `issued_at` is an identifier and a log line, never a deadline. `period_s`
travels with the lease, so the actuator enforces the brain's chosen period without being
configured to match it, and the brain can tighten the rein on one stroke without redeploying
anything.

`Lease.expires_at` still exists for the brain to reason about its own leases on its own clock.
The actuator must not use it, and the tests pin that: two leases differing only in `issued_at`
behave identically at the actuator.

### Trying it without the other machine

```bash
# one terminal: the actuator, which is what Justo's laptop runs
.venv/bin/python -m session.link serve --port 8770

# another: the four messages, by hand
.venv/bin/python - <<'EOF'
import asyncio, json, websockets
from session.wire import Lease, encode_hook_request, encode_heartbeat, decode

async def go():
    async with websockets.connect("ws://127.0.0.1:8770/arm2") as ws:
        lease = Lease(move_id=1, issued_at=0.0, period_s=5.0)
        await ws.send(json.dumps(encode_hook_request("index", lease)))
        print("sent the request; the actuator now prints which finger to hook")
        print("report back:", decode(json.loads(await ws.recv())))
asyncio.run(go())
EOF
```

`session/remote_arm2.py` is both ends of this in code, and
`tests/session/test_remote_arm2.py` exercises the timing failures against a fake link and a
fake clock, which is the only way a lease expiring mid-move is reproducible on demand.

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
  over a real socket, a request reaches the actuator and the lease expires on its own clock,
  all verified here. What has not happened is a hook moving because of a message, with both
  Stations running and an arm on each.
