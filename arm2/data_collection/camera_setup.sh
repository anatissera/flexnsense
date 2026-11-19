#!/usr/bin/env bash
# Fix camera exposure BEFORE starting Station.
# While Station is running it takes the cameras (libuvc) and /dev/videoN disappear,
# so this has to run with Station stopped. Settings live in the camera and usually
# survive until it is unplugged or the PC reboots.
#
# Cameras are identified by USB hub port (both KD-USB cameras share the same by-id name).
# If you move a camera to another port, update the paths below (ls -l /dev/v4l/by-path/).
set -euo pipefail

TABLE_CAM=/dev/v4l/by-path/pci-0000:06:00.3-usb-0:2.1:1.0-video-index0    # KD-USB, overhead/table view
GRIPPER_CAM=/dev/v4l/by-path/pci-0000:06:00.3-usb-0:2.2:1.0-video-index0  # KD-USB, gripper/wrist view
C270_CAM=/dev/v4l/by-path/pci-0000:06:00.3-usb-0:2.3:1.0-video-index0     # Logitech C270

# Manual exposure (KD-USB range 1-5000). Lower = darker. Tune these two values.
TABLE_EXPOSURE=${TABLE_EXPOSURE:-500}
GRIPPER_EXPOSURE=${GRIPPER_EXPOSURE:-500}

kd_manual() {  # device exposure
  v4l2-ctl -d "$1" \
    -c auto_exposure=1 \
    -c exposure_dynamic_framerate=0 \
    -c exposure_time_absolute="$2" \
    -c power_line_frequency=1 \
    -c backlight_compensation=0
  echo "$1 -> manual exposure $2"
}

if pgrep -x station >/dev/null; then
  echo "Station is running: stop it first (Ctrl+C), then rerun this script." >&2
  exit 1
fi

[ -e "$TABLE_CAM" ]   && kd_manual "$TABLE_CAM" "$TABLE_EXPOSURE"     || echo "table cam not found"
[ -e "$GRIPPER_CAM" ] && kd_manual "$GRIPPER_CAM" "$GRIPPER_EXPOSURE" || echo "gripper cam not found"
# Europe mains is 50 Hz; the C270 ships set to 60 Hz, which causes flicker under LED/fluorescent light.
[ -e "$C270_CAM" ]    && v4l2-ctl -d "$C270_CAM" -c power_line_frequency=1 && echo "$C270_CAM -> 50 Hz" || echo "C270 not found"
