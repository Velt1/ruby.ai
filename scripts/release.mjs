#!/usr/bin/env node
/**
 * Sync semantic version across all ruby.ai packages, optionally verify,
 * and prepare/execute commit + tag + push for automated GitHub Releases.
 */

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");

const SEMVER_RE = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/;

function printUsage() {
  console.log(`Usage: node scripts/release.mjs <semver> [options]

Synchronizes the version across:
  - pyproject.toml
  - src/ruby_ai/__init__.py
  - desktop/package.json
  - desktop/src-tauri/tauri.conf.json
  - desktop/src-tauri/Cargo.toml

Options:
  --no-verify   Skip ruff, pytest, stage:engine, and typecheck verification
  --commit      Stage modified version files and create git commit "chore: release v<semver>"
  --tag         Create annotated git tag "v<semver>" (implies --commit if working tree dirty)
  --push        Push current branch and tag "v<semver>" to origin (implies --commit --tag)
  -h, --help    Show this help text
`);
}

/**
 * Declarative registry of versioned project files so additional targets can be added cleanly.
 */
const VERSION_TARGETS = [
  {
    file: "pyproject.toml",
    update(content, version) {
      return replaceSectionField(content, "project", "version", version);
    },
  },
  {
    file: "src/ruby_ai/__init__.py",
    update(content, version) {
      if (!/__version__\s*=/.test(content)) {
        throw new Error("Missing __version__ assignment in src/ruby_ai/__init__.py");
      }
      return content.replace(/(__version__\s*=\s*["'])[^"']+(["'])/, `$1${version}$2`);
    },
  },
  {
    file: "desktop/package.json",
    update(content, version) {
      const parsed = JSON.parse(content);
      parsed.version = version;
      return `${JSON.stringify(parsed, null, 2)}\n`;
    },
  },
  {
    file: "desktop/src-tauri/tauri.conf.json",
    update(content, version) {
      const parsed = JSON.parse(content);
      parsed.version = version;
      return `${JSON.stringify(parsed, null, 2)}\n`;
    },
  },
  {
    file: "desktop/src-tauri/Cargo.toml",
    update(content, version) {
      return replaceSectionField(content, "package", "version", version);
    },
  },
];

function replaceSectionField(tomlText, sectionName, fieldName, nextValue) {
  const newline = tomlText.includes("\r\n") ? "\r\n" : "\n";
  const lines = tomlText.split(/\r?\n/);
  let inTargetSection = false;
  let replaced = false;
  const fieldRe = new RegExp(`^(${fieldName}\\s*=\\s*["'])([^"']+)(["'].*)$`);

  const updated = lines.map((line) => {
    const trimmed = line.trim();
    if (/^\[+[^\]]+\]+$/.test(trimmed)) {
      const header = trimmed.replace(/^\[+|\]+$/g, "").trim();
      inTargetSection = header === sectionName;
      return line;
    }
    if (inTargetSection && !replaced) {
      const m = line.match(fieldRe);
      if (m) {
        replaced = true;
        return `${m[1]}${nextValue}${m[3]}`;
      }
    }
    return line;
  });

  if (!replaced) {
    throw new Error(`Could not update [${sectionName}].${fieldName}`);
  }
  return updated.join(newline);
}

function runCommand(cmd, cwd = REPO_ROOT) {
  console.log(`\n> ${cmd}`);
  execSync(cmd, { cwd, stdio: "inherit" });
}

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args.includes("-h") || args.includes("--help")) {
    printUsage();
    process.exit(0);
  }

  let rawVersion = null;
  let noVerify = false;
  let doCommit = false;
  let doTag = false;
  let doPush = false;

  for (const arg of args) {
    if (arg === "--no-verify") {
      noVerify = true;
    } else if (arg === "--commit") {
      doCommit = true;
    } else if (arg === "--tag") {
      doTag = true;
    } else if (arg === "--push") {
      doPush = true;
    } else if (!arg.startsWith("-") && !rawVersion) {
      rawVersion = arg;
    } else {
      console.error(`Unknown argument: ${arg}`);
      printUsage();
      process.exit(2);
    }
  }

  if (doPush) {
    doTag = true;
    doCommit = true;
  } else if (doTag) {
    doCommit = true;
  }

  if (!rawVersion) {
    console.error("Error: missing <semver> argument.");
    printUsage();
    process.exit(2);
  }

  const version = rawVersion.replace(/^v/i, "").trim();
  if (!SEMVER_RE.test(version)) {
    console.error(`Error: '${rawVersion}' is not a valid SemVer string (expected e.g. 0.2.0).`);
    process.exit(2);
  }

  for (const target of VERSION_TARGETS) {
    const absPath = path.join(REPO_ROOT, target.file);
    const prev = fs.readFileSync(absPath, "utf8");
    const next = target.update(prev, version);
    fs.writeFileSync(absPath, next, "utf8");
    console.log(`[release] Updated ${target.file} -> ${version}`);
  }

  if (!noVerify) {
    runCommand("uv run ruff check .");
    runCommand("uv run pytest -v");
    runCommand("pnpm --dir desktop stage:engine");
    runCommand("pnpm --dir desktop typecheck");
  } else {
    runCommand("pnpm --dir desktop stage:engine");
  }

  const tagName = `v${version}`;
  if (doCommit) {
    const filesToStage = VERSION_TARGETS.map((t) => t.file);
    const lockPath = path.join(REPO_ROOT, "desktop", "src-tauri", "Cargo.lock");
    if (fs.existsSync(lockPath)) {
      filesToStage.push("desktop/src-tauri/Cargo.lock");
    }
    runCommand(`git add ${filesToStage.join(" ")}`);
    runCommand(`git commit -m "chore: release ${tagName}"`);
  }

  if (doTag) {
    runCommand(`git tag -a ${tagName} -m "Release ${tagName}"`);
  }

  if (doPush) {
    runCommand(`git push origin HEAD`);
    runCommand(`git push origin ${tagName}`);
    console.log(`\n[release] Pushed ${tagName}. GitHub Actions will build and publish the OTA release.`);
    return;
  }

  console.log(`\n[release] Version ${version} prepared. Next steps to publish OTA update:`);
  if (!doCommit) {
    console.log(`  git add pyproject.toml src/ruby_ai/__init__.py desktop/package.json desktop/src-tauri/tauri.conf.json desktop/src-tauri/Cargo.toml`);
    console.log(`  git commit -m "chore: release ${tagName}"`);
  }
  if (!doTag) {
    console.log(`  git tag -a ${tagName} -m "Release ${tagName}"`);
  }
  console.log(`  git push origin HEAD && git push origin ${tagName}`);
}

main();
