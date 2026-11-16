"use client";

import dynamic from "next/dynamic";

// The card decides whether to open from localStorage, which only exists in the browser, so it is
// rendered client-side only. Prerendering it would mean a first paint that always says "no card"
// and a hydration pass that disagrees.
const DemoIntro = dynamic(() => import("./DemoIntro"), { ssr: false });

export default function DemoIntroClient() {
  return <DemoIntro />;
}
