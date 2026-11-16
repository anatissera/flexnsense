/**
 * A small, rude rate limit for the two routes that spend money.
 *
 * The demonstration build is public and unauthenticated, and both routes reach OpenAI on our key:
 * `/api/chat` with a conversation, `/api/realtime` by minting an ephemeral client secret that the
 * browser then uses directly. Without a limit, one visitor with a loop empties the key's budget
 * and the demo is dead for everybody else.
 *
 * In-memory and per instance, which is the honest description: a serverless deployment may run
 * several instances, so the real ceiling is this number times however many are warm. That is
 * fine. This is a guard against a loop and a bored visitor, not a security boundary. The actual
 * ceiling on the damage is the spend limit on the OpenAI key, which lives outside this repo.
 */

type Bucket = { count: number; resetAt: number };

const buckets = new Map<string, Bucket>();

/** Drop buckets that have expired, so a long-lived instance does not grow a map of every visitor. */
function sweep(now: number) {
  if (buckets.size < 1000) return;
  for (const [key, bucket] of buckets) if (bucket.resetAt <= now) buckets.delete(key);
}

/**
 * The caller's address as well as a proxy can tell us. Vercel sets `x-forwarded-for`; the first
 * entry is the client. With no header at all everyone shares one bucket, which fails closed.
 */
export function clientKey(req: Request): string {
  const forwarded = req.headers.get("x-forwarded-for");
  const first = forwarded?.split(",")[0]?.trim();
  return first || req.headers.get("x-real-ip") || "unknown";
}

export type RateLimit = { ok: true } | { ok: false; retryAfterS: number };

/** Count one request against `key`. At most `limit` per `windowS`. */
export function take(key: string, limit: number, windowS: number): RateLimit {
  const now = Date.now();
  sweep(now);
  const bucket = buckets.get(key);
  if (!bucket || bucket.resetAt <= now) {
    buckets.set(key, { count: 1, resetAt: now + windowS * 1000 });
    return { ok: true };
  }
  if (bucket.count >= limit) {
    return { ok: false, retryAfterS: Math.max(1, Math.ceil((bucket.resetAt - now) / 1000)) };
  }
  bucket.count += 1;
  return { ok: true };
}

/** The 429 to send back, with the header a well-behaved client will wait on. */
export const tooMany = (retryAfterS: number) =>
  Response.json(
    { error: `Too many requests on the public demo. Try again in ${retryAfterS} s.` },
    { status: 429, headers: { "Retry-After": String(retryAfterS) } },
  );
