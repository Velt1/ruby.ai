#!/usr/bin/env node
/**
 * Stage the Python `ruby_ai` engine and generate `runtime-manifest.json`
 * inside `desktop/src-tauri/resources/engine/` for Tauri bundling.
 */

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DESKTOP_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(DESKTOP_DIR, "..");
const STAGE_ROOT = path.join(DESKTOP_DIR, "src-tauri", "resources", "engine");

const SKIP_DIRS = new Set(["__pycache__", ".pytest_cache", ".ruff_cache"]);
const SKIP_SUFFIXES = [".pyc", ".pyo"];
const TORCH_PKG_RE = /^torch(?:vision|audio)?(?:\s*[<>=!~;\[]|$)/i;

/**
 * Strip inline `# ...` comments outside quoted strings in TOML lines.
 * @param {string} line
 * @returns {string}
 */
function stripTomlComment(line) {
  let quote = null;
  let escaped = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (quote === '"' && ch === "\\") {
      escaped = true;
      continue;
    }
    if (ch === '"' || ch === "'") {
      if (quote === ch) {
        quote = null;
      } else if (quote === null) {
        quote = ch;
      }
      continue;
    }
    if (ch === "#" && quote === null) {
      return line.slice(0, i);
    }
  }
  return line;
}

/**
 * Parse a TOML section's string literal or multiline array values cleanly without external deps.
 * @param {string} tomlText
 */
function parsePyprojectManifestData(tomlText) {
  const lines = tomlText.split(/\r?\n/);
  let currentSection = "";
  let inDependencies = false;
  const rawDepLines = [];
  let version = null;
  let pythonVersion = "3.12";
  let cudaIndexUrl = null;

  for (const rawLine of lines) {
    const clean = stripTomlComment(rawLine).trim();
    if (!clean) continue;

    if (!inDependencies && /^\[+.*\]+$/.test(clean)) {
      currentSection = clean.replace(/^\[+|\]+$/g, "").trim();
      continue;
    }

    if (currentSection === "project") {
      if (!inDependencies) {
        const verMatch = clean.match(/^version\s*=\s*["']([^"']+)["']/);
        if (verMatch) {
          version = verMatch[1].trim();
          continue;
        }
        const depsStart = clean.match(/^dependencies\s*=\s*(.*)$/);
        if (depsStart) {
          const rest = depsStart[1];
          rawDepLines.push(rest);
          if (!rest.includes("]")) {
            inDependencies = true;
          }
          continue;
        }
      } else {
        rawDepLines.push(clean);
        if (clean.includes("]")) {
          inDependencies = false;
        }
        continue;
      }
    }

    if (currentSection === "tool.ruff") {
      const targetPy = clean.match(/^target-version\s*=\s*["']py(\d)(\d+)["']/);
      if (targetPy) {
        pythonVersion = `${targetPy[1]}.${targetPy[2]}`;
      }
    }

    if (currentSection === "tool.uv.index") {
      const urlMatch = clean.match(/^url\s*=\s*["']([^"']+)["']/);
      if (urlMatch && /pytorch\.org\/whl\/cu/i.test(urlMatch[1])) {
        cudaIndexUrl = urlMatch[1].trim();
      }
    }
  }

  if (!version) {
    throw new Error("Could not locate [project].version in pyproject.toml");
  }

  const joinedDeps = rawDepLines.join(" ");
  const dependencies = [];
  const strRe = /"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'/g;
  let match;
  while ((match = strRe.exec(joinedDeps)) !== null) {
    const dep = (match[1] ?? match[2] ?? "").trim();
    if (dep) dependencies.push(dep);
  }

  if (dependencies.length === 0) {
    throw new Error("Could not parse [project].dependencies from pyproject.toml");
  }

  const baseDependencies = [];
  const torchDependencies = [];
  for (const dep of dependencies) {
    if (TORCH_PKG_RE.test(dep)) {
      torchDependencies.push(dep);
    } else {
      baseDependencies.push(dep);
    }
  }

  const torchIndexes = {
    cuda: cudaIndexUrl ?? "https://download.pytorch.org/whl/cu126",
    cpu: "https://download.pytorch.org/whl/cpu",
    mps: null,
  };

  const hashPayload = JSON.stringify({
    python_version: pythonVersion,
    base_dependencies: baseDependencies,
    torch_dependencies: torchDependencies,
    torch_indexes: torchIndexes,
  });
  const depsHash = crypto.createHash("sha256").update(hashPayload).digest("hex");

  return {
    schema_version: 1,
    engine_version: version,
    python_version: pythonVersion,
    base_dependencies: baseDependencies,
    torch_dependencies: torchDependencies,
    torch_indexes: torchIndexes,
    deps_hash: depsHash,
  };
}

/**
 * Recursively copy Python source files into target directory, excluding bytecode caches.
 * @param {string} srcDir
 * @param {string} dstDir
 * @returns {number}
 */
function copyPythonTree(srcDir, dstDir) {
  fs.mkdirSync(dstDir, { recursive: true });
  let count = 0;
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    if (SKIP_DIRS.has(entry.name)) continue;
    const srcPath = path.join(srcDir, entry.name);
    const dstPath = path.join(dstDir, entry.name);
    if (entry.isDirectory()) {
      count += copyPythonTree(srcPath, dstPath);
    } else if (entry.isFile()) {
      if (SKIP_SUFFIXES.some((s) => entry.name.endsWith(s))) continue;
      if (!entry.name.endsWith(".py")) continue;
      fs.copyFileSync(srcPath, dstPath);
      count += 1;
    }
  }
  return count;
}

function main() {
  const pyprojectPath = path.join(REPO_ROOT, "pyproject.toml");
  const srcPackageDir = path.join(REPO_ROOT, "src", "ruby_ai");

  if (!fs.existsSync(pyprojectPath) || !fs.existsSync(srcPackageDir)) {
    throw new Error(`Expected ruby_ai source repository at ${REPO_ROOT}`);
  }

  const pyprojectText = fs.readFileSync(pyprojectPath, "utf8");
  const manifest = parsePyprojectManifestData(pyprojectText);

  fs.rmSync(STAGE_ROOT, { recursive: true, force: true });
  fs.mkdirSync(STAGE_ROOT, { recursive: true });

  fs.copyFileSync(pyprojectPath, path.join(STAGE_ROOT, "pyproject.toml"));
  const copiedFiles = copyPythonTree(srcPackageDir, path.join(STAGE_ROOT, "src", "ruby_ai"));

  fs.writeFileSync(
    path.join(STAGE_ROOT, "runtime-manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8"
  );

  console.log(
    `[stage-engine] Staged ruby_ai v${manifest.engine_version} (${copiedFiles} .py files, ` +
      `${manifest.base_dependencies.length} base deps, ${manifest.torch_dependencies.length} torch deps, ` +
      `hash=${manifest.deps_hash.slice(0, 12)}) -> ${STAGE_ROOT}`
  );
}

main();
