"use client";

import { useCallback, useEffect, useState } from "react";

/**
 * The cameras the browser can see, kept current as they are plugged in or out. Labels stay empty until
 * a camera permission has been granted once, so callers should `refresh()` after their first getUserMedia.
 */
export function useCameraDevices() {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const refresh = useCallback(async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      setDevices(all.filter((d) => d.kind === "videoinput"));
    } catch { /* no mediaDevices (insecure context): the select stays empty and Start reports it */ }
  }, []);
  useEffect(() => {
    const md = navigator.mediaDevices;
    if (!md) return;
    const initial = setTimeout(refresh, 0);
    md.addEventListener("devicechange", refresh);
    return () => {
      clearTimeout(initial);
      md.removeEventListener("devicechange", refresh);
    };
  }, [refresh]);
  return { devices, refresh };
}

/** getUserMedia video constraints for a chosen camera ("" means the browser default). */
export function videoConstraints(deviceId: string, width: number, height: number): MediaTrackConstraints {
  return deviceId ? { deviceId: { exact: deviceId }, width, height } : { width, height };
}

/** Drop-down of every available camera. Changing it while a camera is running restarts that camera. */
export default function CameraSelect({ label, value, devices, onChange }: {
  label: string;
  value: string;
  devices: MediaDeviceInfo[];
  onChange: (deviceId: string) => void;
}) {
  return (
    <label className="t-small muted flex items-center gap-1">
      {label}
      <select className="btn w-40 max-w-full truncate" value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Default camera</option>
        {devices.map((d, i) => (
          <option key={d.deviceId || i} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>
        ))}
        {/* A camera that was chosen and then unplugged stays visible instead of silently snapping to default. */}
        {value && !devices.some((d) => d.deviceId === value) && <option value={value}>Unavailable camera</option>}
      </select>
    </label>
  );
}
