#!/usr/bin/env bash
# Copy our simulation code and the ElRobot model to the workstation, or results back.
#   cloud/nebius/sync.sh up                  sim/ -> ~/flexnsense_sim/   and   assets/elrobot/ -> ~/flexnsense-sim/assets/elrobot/
#   cloud/nebius/sync.sh down REMOTE LOCAL   copy a file or folder from the VM
# The VM must be running (start.sh). Adapted from the Microduck project's sync.sh (reference/sync.microduck.sh).
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
VM=$(vm_id); IP=$(vm_ip "$VM"); [ -n "$IP" ] || { log "VM not running"; exit 1; }
RS=(rsync -a -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new")
case "${1:-}" in
  up)
    "$HERE/ssh.sh" 'mkdir -p ~/flexnsense_sim ~/flexnsense-sim/assets/elrobot'
    "${RS[@]}" --exclude __pycache__ "$ROOT/sim/" "$VM_USER@$IP:flexnsense_sim/"
    "${RS[@]}" "$ROOT/assets/elrobot/" "$VM_USER@$IP:flexnsense-sim/assets/elrobot/"
    log "synced sim/ and the ElRobot model to $IP" ;;
  down)
    [ -n "${2:-}" ] && [ -n "${3:-}" ] || { echo "usage: sync.sh down REMOTE LOCAL"; exit 1; }
    mkdir -p "$(dirname "$3")"
    "${RS[@]}" "$VM_USER@$IP:$2" "$3"
    log "copied $2 -> $3" ;;
  *) echo "usage: sync.sh up | down REMOTE LOCAL"; exit 1 ;;
esac
