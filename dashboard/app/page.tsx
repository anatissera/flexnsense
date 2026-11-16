import SimApp from "./components/SimAppClient";
import DemoBanner from "./components/DemoBanner";
import DemoIntro from "./components/DemoIntroClient";
import ScreenGate from "./components/ScreenGate";
import { HardwareProvider } from "./components/HardwareProvider";

/**
 * The dashboard is a column: the build's own statement of what it is, then the dashboard itself
 * filling what is left. The bar is in the layout rather than over the view so that it can never
 * cover a control, and the dashboard's panels, which anchor to their container's corners, move
 * down with it on their own.
 */
export default function Home() {
  return (
    <main className="relative flex h-screen w-full flex-col bg-black">
      <HardwareProvider>
        <ScreenGate>
          <DemoBanner />
          <div className="relative min-h-0 flex-1">
            <SimApp />
          </div>
          <DemoIntro />
        </ScreenGate>
      </HardwareProvider>
    </main>
  );
}
