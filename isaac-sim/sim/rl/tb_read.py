"""Print the latest values of the task's numbers from a training run's TensorBoard file (works for skrl and rsl_rl).

    ~/IsaacLab/.venv/bin/python ~/flexnsense_sim/rl/tb_read.py ~/IsaacLab/logs/skrl/flex_team/<run> [text to match]
"""

import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run, match = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "Metrics")
acc = EventAccumulator(sorted(glob.glob(run + "/events.out.tfevents.*"))[-1], size_guidance={"scalars": 0})
acc.Reload()
for tag in sorted(acc.Tags()["scalars"]):
    if match.lower() in tag.lower():
        ev = acc.Scalars(tag)
        pick = [ev[int(i * (len(ev) - 1) / 4)] for i in range(5)] if len(ev) >= 5 else ev
        print(f"{tag.split('/')[-1][:34]:34s} " + "  ".join(f"{e.value:7.3f}@{e.step}" for e in pick))
