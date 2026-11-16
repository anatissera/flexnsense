"""What the patient is told, in one table.

Three consumers need the same words: the spoken voice, the 3D guide hand, and the session log.
Today those words are two strings inlined in `GuideHand.tsx` (`"Flex"` and `"Release"`), which
is fine for one component and breaks the moment a second one needs them. Keyed by mode and
phase here, so they cannot drift apart, and
`tests/session/test_instructions.py` checks the table covers every mode and phase that
`session.modes` can produce.

Three of these rows carry more weight than they look:

- The work-phase cue in `active` is the first moment the patient feels the arm move *with*
  them instead of resisting. Being told it is supposed to happen is the difference between
  that feeling like the exercise working and feeling like the device letting go.
- Both `RESET` rows announce a change of driver, which is the moment a patient is most likely
  to fight the machine.
- Every device-driven row says "relax", because a patient tensing against an arm that is
  pulling is the one thing a force cap cannot distinguish from a stiff joint. That includes
  `ARMED`, where the arm is taking up slack: a patient who tenses there adds force the arm
  reads as pre-tension, so it stops short and the repetition starts from the wrong place.
  "Hold still" was the first wording and it is ambiguous, since it can be read as tensing to
  stay put.
"""
from __future__ import annotations

from session.modes import MODES, Phase

#: `(mode name, phase)` to the sentence the patient hears and reads. `{finger}` is substituted
#: from the block. Phases with no entry are silent on purpose: nothing is asked of the patient
#: while a repetition is being counted or a set is ending.
INSTRUCTIONS: dict[tuple[str, Phase], str] = {
    ("active", Phase.ARMED): "Relax your hand while the arm takes up the slack",
    ("active", Phase.CUEING): "Start flexing your {finger} finger, slowly",
    ("active", Phase.RAMP): "Keep flexing",
    ("active", Phase.WORK): "Keep going, the arm is following you",
    ("active", Phase.HOLD): "Hold it there",
    ("active", Phase.RESET): "Relax. The arm will bring your finger back",
    ("passive", Phase.ARMED): "Relax your hand while the arm takes up the slack",
    ("passive", Phase.CUEING): "Relax your {finger} finger and let the arm move it",
    ("passive", Phase.RAMP): "Stay relaxed",
    ("passive", Phase.WORK): "Stay relaxed, let it stretch",
    ("passive", Phase.HOLD): "Hold here, stay relaxed",
    ("passive", Phase.RESET): "Now flex your {finger} finger back to the start",
    # cpm asks nothing of the patient in either beat, so every row says so. The RESET row is
    # the one that differs most from the other modes: there is no change of driver to announce
    # here, because the arm had it all along, and telling a patient to "now flex back" in this
    # mode would be asking for exactly the fighting the cap cannot distinguish from stiffness.
    ("cpm", Phase.ARMED): "Relax your hand while the arm takes up the slack",
    ("cpm", Phase.CUEING): "Relax your {finger} finger. The arm will do the work",
    ("cpm", Phase.RAMP): "Stay relaxed",
    ("cpm", Phase.WORK): "Stay relaxed, let it stretch",
    ("cpm", Phase.HOLD): "Hold here, stay relaxed",
    ("cpm", Phase.RESET): "Stay relaxed. The arm is bringing your finger back",
}

#: Phases whose wording does not depend on the mode.
SHARED: dict[Phase, str] = {
    Phase.IDLE: "Ready when you are",
    Phase.REP_DONE: "Good. Next one",
    Phase.SET_DONE: "That set is done",
    Phase.ABORTED: "Stop and relax your hand",
}


def instruction_key(mode: str, phase: Phase) -> str:
    """A stable identifier for one row, for the log and for the dashboard's own copy.

    Shared phases key on the phase alone, so a client can cache the wording without caring
    which mode produced it.
    """
    return f"{phase.value}" if phase in SHARED else f"{mode}.{phase.value}"


def instruction_for(mode: str, phase: Phase, finger: str) -> str:
    """The sentence to show and speak. Raises if the table has no row, rather than going quiet.

    A silent device that is about to pull on a finger is worse than a crash, so a missing row
    is a bug to be found in a test and not a blank label at the bedside.
    """
    if phase in SHARED:
        return SHARED[phase]
    try:
        template = INSTRUCTIONS[(mode, phase)]
    except KeyError:
        raise KeyError(
            f"no instruction for mode {mode!r} in phase {phase.value!r}. Every mode in "
            f"session.modes.MODES needs a row for every non-terminal phase; known modes are "
            f"{sorted(MODES)}"
        ) from None
    return template.format(finger=finger)
