# Protocols

One file per prescription: which finger, which mode, how many reps, what force cap, how long
to hold. `session.protocol.Protocol.load` reads these, and `Protocol.check_against` warns about
anything the bench numbers in `config/calibration/bench.json` say will not work on this rig.

This is the half a clinician decides. The other half, the machine's own tuning (damping, frame
handling, joint speeds), is in `config/session-tuning.json` and is not per patient. The
dividing question is "would a different patient change this number?"

Name each file for the condition and the mode, because those two together are what makes a
protocol make sense: the same finger and the same cap mean different exercises under different
modes, and the mode is chosen by what is being treated.

## These are examples

Every file here is made up, for the simulation and for tests, and each is named in its `name`
field so that nobody can mistake one for a real prescription.

### tendon-repair-active.json

Active mode: the patient flexes and the arm gives way at a constant force. This is the mode
that drives the exercise off the patient's own effort, which is the point after a tendon
repair, where the tendon has to glide under load the patient themselves generates.

Numbers chosen to be plausible at about week five and to sit comfortably inside this rig:

- caps of 100 to 150 gf, against the 200 gf clinical figure, because the first thing worth
  confirming is never that the controller is good;
- a session ceiling of 250 gf, leaving every block real headroom to give way in before the
  gate would latch RELEASE;
- the same finger twice in a row where the cap steps up, so the orchestrator does not move the
  hook between two blocks that do not need it;
- fingers ordered index, middle, ring, little, which is the order the hook reaches them.

Reaching a 150 gf cap takes about 26 mm of tether stretch on the measured 56.2 N/m tether,
against a measured index excursion of 114 mm, so most of the finger's range is still exercise
rather than run-up. `ARMED` is what takes that slack up before the rep starts.

### age-related-weakness-cpm.json

`cpm` mode: the arm takes the finger out to the stretch target and brings it back, and the
patient does nothing in either beat. The mode to prescribe when the useful thing is many
identical repetitions rather than the patient's own effort, which is what maintaining range in
a weak hand asks for.

Everything about the shape of this file follows from the arm doing the moving:

- **`rom_target_m` is required**, and is the thing that actually ends each stroke. Under the
  cap nothing else tells a pulling arm to stop. 10 to 12 mm here, against measured excursions
  of 98 to 114 mm, so this is a gentle fraction of each finger's range.
- **The caps are lower**, 80 to 90 gf, and they are a *ceiling* rather than the thing the
  exercise works against. A stroke that ends at the cap means the finger was stiffer than
  prescribed and it stopped short of the target, which the session reports as the clinical
  finding it is, not as a failed repetition.
- **More reps, shorter holds**: nothing in a repetition waits on a person, so each one is the
  same motion and they are quick.
- Smaller targets and caps on the ring and little fingers, which measure shorter excursions.

Do not confuse this mode with `passive`. Both leave the patient passive during the stretch;
`passive` then asks them to flex back, and `cpm` does not. Prescribing one and running the
other would be a different exercise.

## What has run on hardware, and what has not

`active` is the mode with bench hours behind it: a full set has been completed end to end on
the real arm. `cpm` has the whole path implemented and tested in software, including the
control law (`control.loop.PullPlan`) that nothing implemented before, but **it has not yet
moved a real finger**. Treat its numbers as a starting point to confirm on the bench with a
hand on the e-stop, not as proven ones.

`passive` shares its work beat with `cpm` and so is implemented too, but its reset beat hands
control back to the patient, and that handover has had no bench time either.

## After editing any of this

Run `Protocol.check_against` against the bench calibration. It is a warning channel, not
validation: an unmeasured rig is a normal state for a prescription to be written in, so the
hard failure belongs where something tries to *run* on an unmeasured number.
