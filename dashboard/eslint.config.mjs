import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Vendored, not ours: MediaPipe's emscripten output. Linting it produced five errors and
    // 260 warnings we cannot act on, which hid the one error that was actually in our code.
    "public/mediapipe/**",
  ]),
]);

export default eslintConfig;
