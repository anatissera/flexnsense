#!/usr/bin/env bash
# Stop any running training or play of the Flex tasks on the workstation.
pkill -f "launch.py (train|play)" && echo stopped || echo "nothing running"
