"use client";

import dynamic from "next/dynamic";

// Simulated numbers differ in the last float digits between Node and the
// browser, so render client-only to avoid hydration mismatches.
const SimApp = dynamic(() => import("./SimApp"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center muted">
      Loading simulator…
    </div>
  ),
});

export default function SimAppClient() {
  return <SimApp />;
}
