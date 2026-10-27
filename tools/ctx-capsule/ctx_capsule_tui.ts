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

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { runCapsuleCli } from "../../../tacit-mui/packages/pi-harness-bridge/src/capsule/index.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const workspaceRoot = path.resolve(__dirname, "../..");

const exitCode = await runCapsuleCli(process.argv.slice(2), workspaceRoot);
process.exitCode = exitCode;
