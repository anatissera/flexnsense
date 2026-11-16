# The dashboard's two builds

The same dashboard runs in two situations, and they differ in one environment variable.

| | Bench build | Demonstration build |
|---|---|---|
| Where | `localhost`, next to the rig | deployed, anyone with the link |
| `NEXT_PUBLIC_DEMO` | unset | `1` |
| Gauge reader, WiLoR | probed on `127.0.0.1`, reported live or not running | never probed |
| Force readings | the camera on the PESOLA, when the gauge server runs | simulated, labelled, in the gauge's place |
| Top bar | the status of the two local servers | an amber notice that nothing is connected |

One codebase, because two demos that drift apart are two demos to debug. The flag decides what the
page *says*; at the bench, a probe decides what it *shows*.

## Why the public build cannot reach hardware

It is not a policy, it is the network. The gauge reader (`ws://127.0.0.1:8766/force`) and the WiLoR
tracker (`ws://127.0.0.1:8765/track`) listen on the loopback address of whichever machine runs them,
which is not the visitor's. Even with a tunnel, a page served over HTTPS reaching `ws://` is blocked
outright in Safari and depends on the browser elsewhere. So the demonstration build does not pretend:
it does not probe, and it says on arrival that nothing is attached.

## What a visitor can actually do

Everything. The point of the demonstration build is that no control is dead.

**Real, in their browser:**

- the MyoHand model, stepped by MuJoCo compiled to WebAssembly;
- hand tracking from their own laptop camera (MediaPipe, in the page: frames are measured and
  discarded, never uploaded and never stored);
- the repetition machine's decisions, cues, counting and stop conditions, which are
  `session/reps.py` ported to TypeScript;
- the analysis (range of motion, rate of force development, hysteresis, the recovery index) and the
  exported reports;
- the assistant, text and voice, including its tool calls into the page;
- the voice stop.

**Simulated and labelled:** every force reading, the patient, the arm.

**Only at the bench:** the PESOLA gauge camera, the two ElRobot arms, WiLoR.

The full inventory is in `dashboard/app/lib/demoInventory.ts` and is what the "What is real here?"
panel renders, so there is one answer rather than one per place it is asked.

## The two demonstration patients

`dashboard/app/lib/demo/patients.ts`. Invented people, real prescriptions: the plans are
`config/protocols/*.json` copied verbatim and parsed through `protocolFromJSON`, the same validation
`session/protocol.py` applies on the rig. A plan that would be refused there is refused here.

- **Jordan Woods**, five weeks after a zone II flexor repair. `active` mode: the patient does the
  work and the arm gives way at the cap.
- **Margrit Imboden**, 81, no surgery, a hand losing range. `cpm`: the arm takes the finger out and
  brings it back and the patient is passive throughout.

Previous and Next patient switch between them. **Load** on a block of the plan fills the rep machine
with exactly that prescription (finger, mode, repetitions, cap, hold, stretch target) and does not
start it. Editing any control afterwards drops the "loaded from the plan" claim, because the panel
must never show a prescription that differs from what Start would run.

## The stops

Three ways to stop a set, all through `RepMachine.abandon()`, so a stopped set is recorded the same
way however it was stopped:

1. the rep machine's own Stop;
2. **Stop session** in the top bar, which is reachable without finding the right panel first;
3. **Voice stop**, which listens for `stop`, `halt`, `para`, `alto`, `basta`, `stopp`, `anhalten`,
   `arrête`, `halte` and the rest.

The voice recogniser takes one language at a time, hence the picker (English, Spanish, German,
French), but the matcher accepts every language's words whichever is picked. A false positive costs
one restarted set; a missed stop is the failure that matters. It needs a browser with the Web Speech
API: Chrome, Edge and Safari have it, Firefox does not, and the control says so rather than sitting
there dead.

**None of this is the safety system.** A browser tab is not what keeps a real finger safe. On the rig
that is the force gate in `control/safety.py` (which releases on a reading it cannot trust, including
no reading at all), `session/interlock.py`, and Station's own current limits in `station.yaml`. All
three work with the browser closed.

## Hints

Notes next to the controls, on by default, switched off from the bar. They answer "what happens if I
press this with nothing connected", which a general disclaimer cannot. A tooltip was the obvious
alternative and the wrong one: nobody hovers a control they have not noticed.

## Running it

```bash
cd dashboard
pnpm install
cp .env.example .env            # GEMINI_API_KEY or OPENAI_API_KEY for the assistant

pnpm dev                                    # bench build: probes 127.0.0.1 for the two servers
NEXT_PUBLIC_DEMO=1 pnpm dev                 # demonstration build

pnpm build && pnpm start                    # production build
pnpm lint
```

The dashboard needs a screen of at least 1280 by 800. Below that it shows a card saying so, with a
way through for anyone who wants to look anyway: the layout is a clinician's instrument panel, and
folding it into a phone would make it a different product rather than a smaller one.

## Deploying the demonstration build

Vercel, with the project's **Root Directory set to `dashboard`** (the repository root is the Python
project).

- Environment variables: `NEXT_PUBLIC_DEMO=1`, and a key for the assistant (`GEMINI_API_KEY` or
  `OPENAI_API_KEY`; see below for which features each one buys).
- Turn **Deployment Protection off**, or visitors meet a login instead of the dashboard.
- `public/` is about 32 MB of meshes, model and WebAssembly. It is static and served from the CDN.

### The assistant runs on an OpenAI or a Gemini key

The text assistant takes either: set `GEMINI_API_KEY` and it uses Gemini's OpenAI-compatible
chat-completions endpoint, tool calls included, or `OPENAI_API_KEY` and it uses OpenAI. Gemini wins
when both are set, and `GEMINI_MODEL` or `OPENAI_MODEL` overrides the default model.

The **voice** assistant is OpenAI-only and not portable by changing a key. `/api/realtime` mints an
ephemeral secret and the browser then speaks OpenAI's realtime event protocol directly; Google's
Live API is a different protocol, so it would be a new client rather than a new base URL. With only
a Gemini key, the text assistant works and the voice button says why it does not.

The **emergency stop by voice** is unaffected by any of this. It is the browser's own Web Speech
API: no key, no provider, no cost.

### The assistant costs money on a public page

`/api/chat` and `/api/realtime` both reach a model provider on our key, and `/api/realtime` mints an
ephemeral OpenAI client secret for anyone who asks for one. Both are rate limited per address in
`dashboard/app/api/ratelimit.ts` (20/min for text, 6/min for voice sessions), which is a guard
against a loop rather than a security boundary: the limit is per serverless instance, so the real
ceiling is that number times however many are warm.

**Set a spend limit on whichever key you deploy with.** That is the only thing that actually bounds the damage, and
it lives outside this repository.
