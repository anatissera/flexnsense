/**
 * Fixed-point formatting that matches Python's `f"{x:.2f}"`.
 *
 * `Number.prototype.toFixed` rounds an exact tie (0.125 to 2 places) away from zero, where
 * Python rounds it to even. The session notes go into a log that is compared against the
 * Python machine's, so the two have to print the same digits.
 */
export function formatFixed(x: number, places: number): string {
  if (!Number.isFinite(x)) return String(x);
  const abs = Math.abs(x);
  let out = abs.toFixed(places);
  // toFixed gives the exact decimal expansion of the double when asked for enough digits, so a
  // tie shows up as a 5 followed by nothing but zeros.
  const full = abs.toFixed(Math.min(100, places + 60));
  const dot = full.indexOf(".");
  const rest = dot < 0 ? "" : full.slice(dot + 1 + places);
  if (rest.length > 0 && rest[0] === "5" && /^0*$/.test(rest.slice(1))) {
    const truncated = places === 0 ? full.slice(0, dot) : full.slice(0, dot + 1 + places);
    const lastDigit = Number(truncated[truncated.length - 1]);
    // Round half to even: keep the truncated value when its last digit is already even.
    if (lastDigit % 2 === 0) out = truncated;
  }
  return x < 0 ? `-${out}` : out;
}
