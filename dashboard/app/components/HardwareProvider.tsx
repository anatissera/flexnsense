"use client";

import { createContext, useContext, type ReactNode } from "react";
import { useHardware, type Hardware } from "../lib/demo";

/**
 * One probe for the whole page.
 *
 * The top bar and the panel badges both need to know what is attached, and two components each
 * opening their own sockets at the bench would be two sockets to the same servers for one fact.
 * The provider probes once and everybody reads the answer.
 */
const Ctx = createContext<Hardware>({ wilor: "unknown", gauge: "unknown" });

export function HardwareProvider({ children }: { children: ReactNode }) {
  return <Ctx.Provider value={useHardware()}>{children}</Ctx.Provider>;
}

export const useHardwareStatus = () => useContext(Ctx);
