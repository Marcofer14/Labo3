"use strict";

const fs = require("fs");
const path = require("path");

const root = "/showdown";

function patchFile(file, patcher) {
  const fullPath = path.join(root, file);
  const before = fs.readFileSync(fullPath, "utf8");
  const after = patcher(before);
  if (after !== before) {
    fs.writeFileSync(fullPath, after);
    console.log(`patched ${file}`);
  } else {
    console.log(`already patched ${file}`);
  }
}

patchFile("dist/sim/battle-stream.js", source => {
  let updated = source;
  if (!updated.includes('var import_state = require("./state");')) {
    updated = updated.replace(
      'var import_battle = require("./battle");',
      'var import_battle = require("./battle");\nvar import_state = require("./state");'
    );
  }
  if (!updated.includes('case "requeststate":')) {
    updated = updated.replace(
      '      case "requestlog":\n        this.push(`requesteddata\n${this.battle.inputLog.join("\\n")}`);\n        break;',
      '      case "requeststate":\n        this.push(`requesteddata\n${JSON.stringify(import_state.State.serializeBattle(this.battle))}`);\n        break;\n      case "requestlog":\n        this.push(`requesteddata\n${this.battle.inputLog.join("\\n")}`);\n        break;'
    );
  }
  return updated;
});

patchFile("dist/server/room-battle.js", source => {
  if (source.includes("async getSerializedState()")) return source;
  return source.replace(
    "  async getInputLog() {\n",
    '  async getSerializedState() {\n    void this.stream.write(">requeststate");\n    const statePromise = new Promise((resolve, reject) => {\n      if (!this.dataResolvers) this.dataResolvers = [];\n      this.dataResolvers.push([resolve, reject]);\n    });\n    const result = await statePromise;\n    if (!result || !result[0]) return null;\n    return JSON.parse(result[0]);\n  }\n  async getInputLog() {\n'
  );
});

patchFile("dist/server/index.js", source => {
  if (source.includes('require("./live-state-bridge").start();')) return source;
  return source.replace(
    "  setupGlobals();\n}).then(() => {",
    '  setupGlobals();\n  try {\n    require("./live-state-bridge").start();\n  } catch (error) {\n    console.error("Failed to start showdown-live-state bridge:", error?.stack || error);\n  }\n}).then(() => {'
  );
});
