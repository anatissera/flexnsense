"use client";

import dynamic from "next/dynamic";
import type { ViewerHooks } from "./MyoViewer";

const MyoViewer = dynamic(() => import("./MyoViewer"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center muted">
      Loading viewer…
    </div>
  ),
});

export default function MyoViewerClient(props: ViewerHooks) {
  return <MyoViewer {...props} />;
}
