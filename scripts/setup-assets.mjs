#!/usr/bin/env node
/**
 * Prepare the self-hosted MediaPipe assets.
 *
 * Two jobs:
 *   1. Copy the WASM runtime out of `node_modules/@mediapipe/tasks-vision`.
 *   2. Download the `.task` model files and verify them against `models.lock.json`.
 *
 * Why self-host rather than use the CDN pattern from Google's docs? The docs load
 * `@latest`, which silently version-drifts — and the 0.10 → 1.0 jump shows that is a
 * real event, not a hypothetical. Pinned, self-hosted assets mean a deployment is
 * reproducible and works offline (docs/AUDIT.md, V1 and V2).
 *
 * Why download rather than commit the binaries? They total ~50 MB. Committing them
 * would bloat every clone forever, and a checksum-verified fetch gives the same
 * integrity guarantee. `models.lock.json` is the integrity record — commit it.
 *
 * Usage:
 *   node scripts/setup-assets.mjs                # copy + fetch + verify
 *   node scripts/setup-assets.mjs --update-lock  # record checksums for new files
 *   node scripts/setup-assets.mjs --offline      # copy WASM only, skip downloads
 *   node scripts/setup-assets.mjs --allow-missing # warn instead of failing
 *
 * `npm run build` uses --allow-missing so development works without network; the
 * deploy workflow runs it strictly so a release can never ship without models.
 */
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PUBLIC_DIR = join(ROOT, 'packages/web/public');
const WASM_SOURCE = join(ROOT, 'node_modules/@mediapipe/tasks-vision/wasm');
const WASM_DEST = join(PUBLIC_DIR, 'mediapipe/wasm');
const MODEL_DEST = join(PUBLIC_DIR, 'models');
const LOCK_PATH = join(ROOT, 'models.lock.json');

/**
 * WASM files to copy.
 *
 * Only the SIMD build is shipped. SIMD has been universal in browsers since ~2021,
 * and the no-SIMD fallback is another ~11 MB for a population that effectively no
 * longer exists. Add `vision_wasm_nosimd_internal.*` here if that ever changes.
 */
const WASM_FILES = ['vision_wasm_internal.js', 'vision_wasm_internal.wasm'];

/**
 * Model files, pinned to the `latest` alias Google publishes per model.
 *
 * The alias is not a version, so `models.lock.json` is what actually pins them: a
 * checksum change surfaces as a failed verification rather than silently shipping a
 * different model.
 */
const MODELS = [
  {
    name: 'hand_landmarker.task',
    url: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    required: true,
  },
  {
    name: 'pose_landmarker_lite.task',
    url: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
    required: false,
  },
];

const args = new Set(process.argv.slice(2));
const updateLock = args.has('--update-lock');
const offline = args.has('--offline');
const allowMissing = args.has('--allow-missing');

/** Human-readable byte size. */
const mb = (bytes) => `${(bytes / 1024 / 1024).toFixed(2)} MB`;

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function sha256(path) {
  return createHash('sha256')
    .update(await readFile(path))
    .digest('hex');
}

async function loadLock() {
  try {
    return JSON.parse(await readFile(LOCK_PATH, 'utf8'));
  } catch {
    return {};
  }
}

async function copyRuntime(name, source, destination, files, hint) {
  if (!(await exists(source))) {
    throw new Error(`${name} WASM not found. Run \`npm install\` first — ${hint}`);
  }
  await mkdir(destination, { recursive: true });

  let total = 0;
  for (const file of files) {
    const from = join(source, file);
    const to = join(destination, file);
    await copyFile(from, to);
    const { size } = await stat(to);
    total += size;
    console.log(`  ✓ ${file} (${mb(size)})`);
  }
  console.log(`  ${name}: ${mb(total)} — gzip brings this to roughly a third.`);
}

async function copyWasm() {
  await copyRuntime(
    'MediaPipe',
    WASM_SOURCE,
    WASM_DEST,
    WASM_FILES,
    '@mediapipe/tasks-vision ships it.',
  );
}

async function fetchModel(model, lock) {
  const destination = join(MODEL_DEST, model.name);
  const recorded = lock[model.name];

  // Already present and matching? Nothing to do.
  if (await exists(destination)) {
    const digest = await sha256(destination);
    if (recorded === undefined) {
      if (updateLock) {
        const { size } = await stat(destination);
        lock[model.name] = digest;
        console.log(`  + ${model.name} recorded (${mb(size)})`);
        return true;
      }
      console.warn(`  ! ${model.name} present but not in models.lock.json`);
      console.warn('    Run with --update-lock to record it, then commit the lockfile.');
      return true;
    }
    if (digest === recorded) {
      console.log(`  ✓ ${model.name} verified`);
      return true;
    }
    if (!updateLock) {
      throw new Error(
        `${model.name} does not match models.lock.json.\n` +
          `  expected ${recorded}\n  actual   ${digest}\n` +
          'Delete the file and re-run, or use --update-lock if the change is intended.',
      );
    }
    lock[model.name] = digest;
    console.log(`  ~ ${model.name} checksum updated`);
    return true;
  }

  if (offline) {
    console.warn(`  - ${model.name} missing (offline mode)`);
    return !model.required;
  }

  console.log(`  ↓ ${model.name}`);
  let response;
  try {
    response = await fetch(model.url);
  } catch (cause) {
    if (allowMissing) {
      console.warn(`  ! ${model.name} could not be downloaded: ${cause.message ?? cause}`);
      return !model.required;
    }
    throw new Error(`Could not reach ${model.url}. Check your network.`, { cause });
  }
  if (!response.ok) {
    // A blocked egress proxy answers 403 rather than refusing the connection, so an
    // HTTP error is treated the same as unreachable.
    if (allowMissing) {
      console.warn(`  ! ${model.name} unavailable: HTTP ${response.status}`);
      return !model.required;
    }
    throw new Error(`${model.url} returned HTTP ${response.status}`);
  }

  await mkdir(MODEL_DEST, { recursive: true });
  const bytes = Buffer.from(await response.arrayBuffer());
  await writeFile(destination, bytes);

  const digest = createHash('sha256').update(bytes).digest('hex');
  if (recorded !== undefined && recorded !== digest) {
    throw new Error(
      `Downloaded ${model.name} does not match models.lock.json.\n` +
        `  expected ${recorded}\n  actual   ${digest}\n` +
        'The upstream `latest` alias has moved. Review the change, then --update-lock.',
    );
  }
  if (recorded === undefined) {
    lock[model.name] = digest;
    console.log(`  + ${model.name} (${mb(bytes.length)}) checksum recorded`);
  } else {
    console.log(`  ✓ ${model.name} (${mb(bytes.length)}) verified`);
  }

  // Cloudflare Pages rejects any single asset above 25 MiB at deploy time.
  if (bytes.length > 25 * 1024 * 1024) {
    console.warn(
      `  ! ${model.name} is ${mb(bytes.length)}, above Cloudflare Pages' 25 MiB asset limit.`,
    );
    console.warn('    See docs/DEPLOY.md for how to serve it from GitHub Releases instead.');
  }
  return true;
}

async function main() {
  console.log('MediaPipe assets');
  console.log('• WASM runtime');
  await copyWasm();

  console.log('• Models');
  const lock = await loadLock();
  const before = JSON.stringify(lock);

  let allRequiredPresent = true;
  for (const model of MODELS) {
    const ok = await fetchModel(model, lock);
    if (!ok && model.required) allRequiredPresent = false;
  }

  if (JSON.stringify(lock) !== before) {
    await writeFile(LOCK_PATH, `${JSON.stringify(lock, null, 2)}\n`);
    console.log(`• Wrote ${LOCK_PATH.replace(`${ROOT}/`, '')} — commit this file.`);
  }

  if (!allRequiredPresent) {
    const note = 'A required model is missing. The Translator will not start.';
    if (allowMissing) {
      console.warn(`\n${note}`);
      console.warn('Run `npm run setup:assets` with network access to fix.');
      return;
    }
    console.error(`\n${note}`);
    process.exitCode = 1;
    return;
  }
  console.log('\nDone.');
}

main().catch((error) => {
  console.error(`\n${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
