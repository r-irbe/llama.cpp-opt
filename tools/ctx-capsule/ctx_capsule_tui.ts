#!/usr/bin/env -S node --experimental-strip-types
/**
 * Pure TypeScript FOOP & Hexagonal `ctx-capsule` Control Plane TUI & CLI Entrypoint.
 *
 * Delegates directly to the Hexagonal Architecture domain & adapter stack in
 * `pi-harness-bridge/src/capsule/index.ts`.
 *
 * Usage:
 *   node --experimental-strip-types tools/ctx-capsule/ctx_capsule_tui.ts --render-once
 *   node --experimental-strip-types tools/ctx-capsule/ctx_capsule_tui.ts --export-toml
 *   node --experimental-strip-types tools/ctx-capsule/ctx_capsule_tui.ts --model qwen38_27b --template long_horizon_1m
 *   node --experimental-strip-types tools/ctx-capsule/ctx_capsule_tui.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const workspaceRoot = path.resolve(__dirname, "../..");

async function resolveRunCapsuleCli(): Promise<
  (opts: { argv: readonly string[]; workspaceRoot: string }) => Promise<number>
> {
  const parentDir = path.resolve(__dirname, "../../..");
  const candidates: string[] = [path.join(parentDir, "ctx-capsule/src/index.ts")];
  if (fs.existsSync(parentDir)) {
    for (const entry of fs.readdirSync(parentDir)) {
      candidates.push(path.join(parentDir, entry, "packages/pi-harness-bridge/src/capsule/index.ts"));
    }
  }
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      const mod = (await import(pathToFileURL(candidate).href)) as {
        runCapsuleCli: (opts: { argv: readonly string[]; workspaceRoot: string }) => Promise<number>;
      };
      return mod.runCapsuleCli;
    }
  }
  const pkgMod = (await import("@r-irbe/ctx-capsule")) as {
    runCapsuleCli: (opts: { argv: readonly string[]; workspaceRoot: string }) => Promise<number>;
  };
  return pkgMod.runCapsuleCli;
}

const runCapsuleCli = await resolveRunCapsuleCli();

const exitCode = (process.exitCode = await runCapsuleCli({ argv: process.argv.slice(2), workspaceRoot }));

