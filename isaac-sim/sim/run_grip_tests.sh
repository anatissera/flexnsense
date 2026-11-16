#!/usr/bin/env bash
# Run 08_physical_scale.py once per jaw setting ("stiffness,damping,force"), in parallel, without recording,
# and print the grip measurements of each run. On the workstation:  bash ~/flexnsense_sim/run_grip_tests.sh 40,2,2.94 300,10,6
source ~/isaac-env.sh; cd ~/IsaacLab
for j in "$@"; do
  DBG_STOP_T=${STOP:-1e9} DBG_OVER=${OVER:-0.06} DBG_JAW=$j timeout 540 uv run --extra isaacsim python ~/flexnsense_sim/08_physical_scale.py --no_record --tag "t_$j${SUF:-}" > ~/"t_$j${SUF:-}.log" 2>&1 &
done
wait
for j in "$@"; do
  echo "=== jaws $j"
  grep -E "^\[(TWO|GRIPCHK|PHYS)\]|Traceback|Error" ~/"t_$j${SUF:-}.log" | grep -v "root pose\|travel of\|ignores" \
    | grep -E "PHYS|STOPPED|summary|Trace|Error|grip |lift |shift|lower|open |up " | cut -c1-270 | tail -${TAIL:-24}
done
