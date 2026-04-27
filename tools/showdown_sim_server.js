#!/usr/bin/env node
"use strict";

const http = require("http");

const {Battle, Teams} = require("/showdown/dist/sim");
const {State} = require("/showdown/dist/sim/state");

const PORT = Number(process.env.SHOWDOWN_SIM_PORT || 9001);
const DEFAULT_SEED = [1, 2, 3, 4];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function jsonResponse(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

function shortError(error) {
  return String(error?.stack || error || "").slice(0, 700);
}

function noteError(diagnostics, stage, error, extra = {}) {
  if (!diagnostics) return;
  diagnostics.skippedBranches += 1;
  diagnostics.errorCount += 1;
  diagnostics.stageCounts[stage] = (diagnostics.stageCounts[stage] || 0) + 1;
  if (diagnostics.errors.length < 10) {
    diagnostics.errors.push({stage, error: shortError(error), ...extra});
  }
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", chunk => {
      body += chunk;
      if (body.length > 8 * 1024 * 1024) {
        reject(new Error("request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

function packedTeam(teamText) {
  if (!teamText) return "";
  if (!teamText.includes("\n") && teamText.includes("|")) return teamText;
  return Teams.pack(Teams.import(teamText));
}

function normalizeChoice(choice) {
  return String(choice || "")
    .trim()
    .replace(/^\/choose\s+/i, "")
    .replace(/^\/team\s+/i, "team ")
    .replace(/^\/forfeit\s*$/i, "forfeit");
}

function choiceHasTera(choice) {
  return /\bterastallize\b/i.test(choice);
}

function otherSide(side) {
  return side === "p1" ? "p2" : "p1";
}

function resetBattleLog(battle) {
  battle.log = [];
  battle.sentLogPos = 0;
}

function cloneBattle(battle) {
  const clone = State.deserializeBattle(State.serializeBattle(battle));
  resetBattleLog(clone);
  return clone;
}

function createBattle(payload, diagnostics) {
  const battle = new Battle({
    formatid: payload.format || "gen9vgc2026regi",
    seed: payload.seed || DEFAULT_SEED,
    strictChoices: true,
  });
  battle.azRepairs = 0;
  battle.azHistoryFailed = false;

  const p1Team = packedTeam(payload.team_p1 || payload.team || "");
  const p2Team = packedTeam(payload.team_p2 || payload.team || "");
  battle.setPlayer("p1", {name: "p1", team: p1Team});
  battle.setPlayer("p2", {name: "p2", team: p2Team});

  for (const item of payload.history || []) {
    const side = item.side;
    const choice = normalizeChoice(item.choice);
    if (!side || !choice) continue;
    if (!applyHistoryChoice(battle, side, choice, diagnostics, item.key || "")) {
      battle.azHistoryFailed = true;
      break;
    }
  }
  return battle;
}

function createOfflineBattle(payload) {
  const battle = new Battle({
    formatid: payload.format || "gen9vgc2026regi",
    seed: payload.seed || DEFAULT_SEED,
    strictChoices: true,
  });
  const p1Team = packedTeam(payload.team_p1 || payload.team || "");
  const p2Team = packedTeam(payload.team_p2 || payload.team || "");
  battle.setPlayer("p1", {name: "p1", team: p1Team});
  battle.setPlayer("p2", {name: "p2", team: p2Team});
  applyTeamPreviewIfNeeded(battle, payload);
  resetBattleLog(battle);
  return battle;
}

function defaultTeamChoice(request) {
  const pokemonCount = request?.side?.pokemon?.length || 6;
  const maxTeamSize = request?.maxTeamSize || Math.min(4, pokemonCount);
  return `team ${Array.from({length: maxTeamSize}, (_, i) => i + 1).join("")}`;
}

function applyTeamPreviewIfNeeded(battle, payload = {}) {
  const p1Request = activeRequestForSide(battle, "p1");
  const p2Request = activeRequestForSide(battle, "p2");
  const p1Choice = normalizeChoice(payload.team_choice_p1 || defaultTeamChoice(p1Request));
  const p2Choice = normalizeChoice(payload.team_choice_p2 || defaultTeamChoice(p2Request));
  if (p1Request?.teamPreview) battle.choose("p1", p1Choice);
  if (p2Request?.teamPreview) battle.choose("p2", p2Choice);
}

function applyHistoryChoice(battle, side, choice, diagnostics, key = "") {
  try {
    battle.choose(side, choice);
    return true;
  } catch (error) {
    noteError(diagnostics, "history-choice", error, {
      side,
      choice,
      key,
      ...battleContext(battle),
    });
    battle.azRepairs += 1;
    return false;
  }
}

function activeSummary(battle, sideID) {
  const side = battle.sides[sideID === "p1" ? 0 : 1];
  if (!side) return [];
  return side.active.map(mon => {
    if (!mon) return "empty";
    const hp = Number.isFinite(mon.hp) && Number.isFinite(mon.maxhp)
      ? `${mon.hp}/${mon.maxhp}`
      : "?";
    const status = mon.status ? ` ${mon.status}` : "";
    return `${mon.species}:${hp}${status}`;
  });
}

function battleContext(battle) {
  return {
    turn: battle.turn,
    requestState: battle.requestState,
    p1Active: activeSummary(battle, "p1"),
    p2Active: activeSummary(battle, "p2"),
  };
}

function activeRequestForSide(battle, sideID) {
  const side = battle.sides[sideID === "p1" ? 0 : 1];
  if (side && side.activeRequest) return side.activeRequest;
  try {
    const requests = battle.getRequests(battle.requestState);
    return requests[sideID === "p1" ? 0 : 1] || null;
  } catch {
    return null;
  }
}

function availableSwitches(request) {
  const choices = [];
  const pokemon = request?.side?.pokemon || [];
  for (let i = 0; i < pokemon.length; i++) {
    const mon = pokemon[i];
    const condition = String(mon.condition || "");
    if (mon.active) continue;
    if (condition.endsWith(" fnt") || condition === "0 fnt") continue;
    choices.push(`switch ${i + 1}`);
  }
  return choices;
}

function moveTargets(move) {
  const target = String(move.target || "").toLowerCase();
  if (target === "normal" || target === "any" || target === "adjacentfoe") {
    return [" -1", " -2"];
  }
  if (target === "adjacentally" || target === "adjacentallyorself") {
    return [" 1", " 2"];
  }
  if (target === "randomnormal") return [""];
  return [""];
}

function slotChoices(request, slot) {
  if (!request) return ["default"];
  const switches = availableSwitches(request);

  if (request.forceSwitch?.[slot]) {
    return switches.length ? switches : ["pass"];
  }

  const active = request.active?.[slot];
  if (!active) return ["pass"];

  const choices = [];
  for (const move of active.moves || []) {
    if (move.disabled) continue;
    const moveID = move.id || move.move;
    if (!moveID) continue;
    for (const target of moveTargets(move)) {
      choices.push(`move ${moveID}${target}`);
      if (active.canTerastallize) {
        choices.push(`move ${moveID} terastallize${target}`);
      }
    }
  }
  if (!active.trapped) choices.push(...switches);
  return choices.length ? choices : ["pass"];
}

function roughChoiceScore(choice) {
  let score = 0;
  const lower = choice.toLowerCase();
  score += (lower.match(/\bmove\b/g) || []).length * 3;
  score -= (lower.match(/\bswitch\b/g) || []).length * 1;
  score -= (lower.match(/\bpass\b/g) || []).length * 4;
  if (lower.includes("terastallize")) score += 1;
  if (lower.includes("protect") || lower.includes("detect")) score -= 1;
  if (lower.includes("recover")) score -= 1;
  return score;
}

function combineChoices(first, second) {
  const choices = [];
  for (const a of first) {
    for (const b of second) {
      const switchA = a.match(/^switch\s+(\d+)/i)?.[1];
      const switchB = b.match(/^switch\s+(\d+)/i)?.[1];
      if (switchA && switchB && switchA === switchB) continue;
      if (choiceHasTera(a) && choiceHasTera(b)) continue;
      choices.push(`${a}, ${b}`);
    }
  }
  return choices;
}

function isLegalChoice(battle, side, choice) {
  try {
    const clone = cloneBattle(battle);
    clone.choose(side, choice);
    return true;
  } catch {
    return false;
  }
}

function legalChoicesForSide(battle, side, maxChoices) {
  const request = activeRequestForSide(battle, side);
  if (!request || request.wait) return [];
  if (request.teamPreview) {
    const pokemonCount = request.side?.pokemon?.length || 6;
    const maxTeamSize = request.maxTeamSize || Math.min(4, pokemonCount);
    const defaultOrder = Array.from({length: maxTeamSize}, (_, i) => i + 1).join("");
    return [`team ${defaultOrder}`];
  }

  const first = slotChoices(request, 0);
  const second = slotChoices(request, 1);
  let choices = combineChoices(first, second);
  choices = choices.filter(choice => isLegalChoice(battle, side, choice));
  choices.sort((a, b) => roughChoiceScore(b) - roughChoiceScore(a));
  if (maxChoices > 0) choices = choices.slice(0, maxChoices);
  return choices;
}

function moveSnapshot(battle, moveID) {
  const move = battle.dex.moves.get(moveID);
  return {
    id: move.id || String(moveID || ""),
    name: move.name || String(moveID || ""),
    type: move.type || "",
    category: move.category || "",
    basePower: Number(move.basePower || 0),
    accuracy: move.accuracy === true ? 100 : Number(move.accuracy || 100),
    priority: Number(move.priority || 0),
    target: move.target || "",
  };
}

function pokemonMoveIDs(pokemon) {
  if (!pokemon) return [];
  if (Array.isArray(pokemon.moveSlots)) {
    return pokemon.moveSlots.map(slot => slot.id || slot.move || "").filter(Boolean);
  }
  if (Array.isArray(pokemon.moves)) return pokemon.moves.filter(Boolean);
  return [];
}

function pokemonSnapshot(battle, pokemon) {
  if (!pokemon) return null;
  const types = typeof pokemon.getTypes === "function" ? pokemon.getTypes() : pokemon.types || [];
  return {
    species: pokemon.species || pokemon.name || "",
    name: pokemon.name || pokemon.species || "",
    hp: Number(pokemon.hp || 0),
    maxhp: Number(pokemon.maxhp || 0),
    hp_fraction: pokemon.maxhp ? clamp(Number(pokemon.hp || 0) / Number(pokemon.maxhp), 0, 1) : 0,
    fainted: Boolean(pokemon.fainted || pokemon.hp <= 0),
    active: Boolean(pokemon.isActive),
    status: pokemon.status || "",
    item: pokemon.item || "",
    ability: pokemon.ability || pokemon.baseAbility || "",
    tera_type: pokemon.teraType || "",
    terastallized: pokemon.terastallized || "",
    types: Array.from(types || []).map(type => String(type)),
    boosts: {...(pokemon.boosts || {})},
    volatiles: Object.keys(pokemon.volatiles || {}),
    moves: pokemonMoveIDs(pokemon).map(moveID => moveSnapshot(battle, moveID)),
  };
}

function sideSnapshot(battle, sideID, legalChoices) {
  const side = battle.sides[sideID === "p1" ? 0 : 1];
  if (!side) return {};
  return {
    name: side.name,
    id: side.id,
    active: side.active.map(pokemon => pokemonSnapshot(battle, pokemon)),
    team: side.pokemon.map(pokemon => pokemonSnapshot(battle, pokemon)),
    side_conditions: Object.keys(side.sideConditions || {}),
    can_tera: legalChoices.some(choice => choice.includes("terastallize")),
  };
}

function safeLegalChoicesForSide(battle, side, maxChoices) {
  try {
    return legalChoicesForSide(battle, side, maxChoices);
  } catch {
    return [];
  }
}

function winnerSideID(battle) {
  if (!battle.winner) return "";
  const side = battle.sides.find(item => item.name === battle.winner);
  return side?.id || "";
}

function fieldSnapshot(battle) {
  return {
    weather: battle.field?.weather || "",
    terrain: battle.field?.terrain || "",
    pseudo_weather: Object.keys(battle.field?.pseudoWeather || {}),
  };
}

function exportOfflineState(battle, maxChoices = 0) {
  resetBattleLog(battle);
  const legalP1 = safeLegalChoicesForSide(battle, "p1", maxChoices);
  const legalP2 = safeLegalChoicesForSide(battle, "p2", maxChoices);
  return {
    state: State.serializeBattle(battle),
    turn: battle.turn,
    request_state: battle.requestState,
    ended: Boolean(battle.ended || battle.winner),
    winner: battle.winner || "",
    winner_side: winnerSideID(battle),
    score: {
      p1: scoreBattle(battle, "p1"),
      p2: scoreBattle(battle, "p2"),
    },
    field: fieldSnapshot(battle),
    sides: {
      p1: sideSnapshot(battle, "p1", legalP1),
      p2: sideSnapshot(battle, "p2", legalP2),
    },
    legal: {
      p1: legalP1,
      p2: legalP2,
    },
  };
}

function hpFraction(pokemon) {
  if (!pokemon || pokemon.fainted) return 0;
  if (!pokemon.maxhp) return 0;
  return clamp(pokemon.hp / pokemon.maxhp, 0, 1);
}

function scoreBattle(battle, perspective) {
  if (battle.ended || battle.winner) {
    if (!battle.winner) return 0;
    const winnerSide = battle.sides.find(side => side.name === battle.winner)?.id;
    return winnerSide === perspective ? 1 : -1;
  }

  const own = battle.sides[perspective === "p1" ? 0 : 1].pokemon;
  const opp = battle.sides[perspective === "p1" ? 1 : 0].pokemon;
  const ownHp = own.reduce((total, mon) => total + hpFraction(mon), 0) / Math.max(1, own.length);
  const oppHp = opp.reduce((total, mon) => total + hpFraction(mon), 0) / Math.max(1, opp.length);
  const ownAlive = own.filter(mon => !mon.fainted && mon.hp > 0).length / Math.max(1, own.length);
  const oppAlive = opp.filter(mon => !mon.fainted && mon.hp > 0).length / Math.max(1, opp.length);
  return clamp(0.65 * (ownHp - oppHp) + 0.35 * (ownAlive - oppAlive), -1, 1);
}

function chooseBoth(baseBattle, side, ownChoice, opponentChoice) {
  const battle = cloneBattle(baseBattle);
  const opponent = otherSide(side);
  if (side === "p1") {
    battle.choose(side, ownChoice);
    if (opponentChoice) battle.choose(opponent, opponentChoice);
  } else {
    if (opponentChoice) battle.choose(opponent, opponentChoice);
    battle.choose(side, ownChoice);
  }
  return battle;
}

function responseAggregate(values, policy, worstWeight) {
  if (!values.length) return 0;
  const worst = Math.min(...values);
  if (policy === "minimax") return worst;
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  if (policy === "mean") return mean;
  const weight = clamp(Number(worstWeight), 0, 1);
  return (1 - weight) * mean + weight * worst;
}

function bestValue(battle, perspective, depth, maxChoices, policy, worstWeight, diagnostics) {
  if (depth <= 0 || battle.ended || battle.winner) return scoreBattle(battle, perspective);
  let choices = [];
  try {
    choices = legalChoicesForSide(battle, perspective, maxChoices);
  } catch (error) {
    noteError(diagnostics, "best-legal-choices", error, {
      perspective,
      depth,
      ...battleContext(battle),
    });
    return scoreBattle(battle, perspective);
  }
  if (!choices.length) return scoreBattle(battle, perspective);
  let best = -Infinity;
  for (const choice of choices) {
    best = Math.max(
      best,
      candidateValue(battle, perspective, choice, depth, maxChoices, policy, worstWeight, diagnostics)
    );
  }
  return Number.isFinite(best) ? best : scoreBattle(battle, perspective);
}

function candidateValue(
  battle,
  perspective,
  ownChoice,
  depth,
  maxChoices,
  policy,
  worstWeight,
  diagnostics,
  fixedOpponentChoices = null
) {
  const opponent = otherSide(perspective);
  let opponentChoices = [];
  if (fixedOpponentChoices) {
    opponentChoices = fixedOpponentChoices;
  } else {
    try {
      opponentChoices = legalChoicesForSide(battle, opponent, maxChoices);
    } catch (error) {
      noteError(diagnostics, "opponent-legal-choices", error, {
        perspective,
        opponent,
        ownChoice,
        depth,
        ...battleContext(battle),
      });
    }
  }
  if (!opponentChoices.length) opponentChoices = [""];

  const values = [];
  for (const opponentChoice of opponentChoices) {
    try {
      const next = chooseBoth(battle, perspective, ownChoice, opponentChoice);
      const value = depth > 1
        ? bestValue(next, perspective, depth - 1, maxChoices, policy, worstWeight, diagnostics)
        : scoreBattle(next, perspective);
      if (Number.isFinite(value)) values.push(value);
    } catch (error) {
      noteError(diagnostics, "candidate-branch", error, {
        perspective,
        ownChoice,
        opponentChoice,
        depth,
        ...battleContext(battle),
      });
      continue;
    }
  }
  return values.length ? responseAggregate(values, policy, worstWeight) : scoreBattle(battle, perspective);
}

function evaluate(payload) {
  const depth = Math.max(1, Number(payload.depth || 1));
  const maxChoices = Math.max(1, Number(payload.max_choices || 12));
  const policy = ["minimax", "mean", "robust"].includes(payload.opponent_policy)
    ? payload.opponent_policy
    : "robust";
  const robustWorstWeight = Number.isFinite(Number(payload.robust_worst_weight))
    ? Number(payload.robust_worst_weight)
    : 0.35;
  const side = payload.side || "p1";
  const diagnostics = {errors: [], errorCount: 0, skippedBranches: 0, stageCounts: {}};
  const battle = createBattle(payload, diagnostics);
  if (battle.azHistoryFailed) {
    return {
      ok: false,
      error: "could not replay simulator history exactly",
      turn: battle.turn,
      repairs: battle.azRepairs || 0,
      simulation_errors: diagnostics.errorCount,
      skipped_branches: diagnostics.skippedBranches,
      error_stage_counts: diagnostics.stageCounts,
      errors: diagnostics.errors,
    };
  }
  resetBattleLog(battle);
  const values = [];
  let rootOpponentChoices = [];
  try {
    rootOpponentChoices = legalChoicesForSide(battle, otherSide(side), maxChoices);
  } catch (error) {
    noteError(diagnostics, "root-opponent-legal-choices", error, {
      side,
      depth,
      ...battleContext(battle),
    });
  }
  if (!rootOpponentChoices.length) rootOpponentChoices = [""];
  for (const rawCandidate of payload.candidates || []) {
    const candidate = normalizeChoice(rawCandidate);
    if (!candidate || candidate === "forfeit") {
      values.push(-1);
      continue;
    }
    try {
      values.push(
        candidateValue(
          battle,
          side,
          candidate,
          depth,
          maxChoices,
          policy,
          robustWorstWeight,
          diagnostics,
          rootOpponentChoices
        )
      );
    } catch (error) {
      noteError(diagnostics, "root-candidate", error, {
        side,
        candidate,
        depth,
        ...battleContext(battle),
      });
      values.push(scoreBattle(battle, side));
    }
  }
  return {
    ok: true,
    values,
    turn: battle.turn,
    repairs: battle.azRepairs || 0,
    simulation_errors: diagnostics.errorCount,
    skipped_branches: diagnostics.skippedBranches,
    error_stage_counts: diagnostics.stageCounts,
    errors: diagnostics.errors,
    opponent_policy: policy,
    robust_worst_weight: clamp(robustWorstWeight, 0, 1),
    request_state: battle.requestState,
  };
}

function evaluateOffline(payload) {
  const depth = Math.max(1, Number(payload.depth || 1));
  const maxChoices = Math.max(1, Number(payload.max_choices || 12));
  const policy = ["minimax", "mean", "robust"].includes(payload.opponent_policy)
    ? payload.opponent_policy
    : "robust";
  const robustWorstWeight = Number.isFinite(Number(payload.robust_worst_weight))
    ? Number(payload.robust_worst_weight)
    : 0.35;
  const side = payload.side || "p1";
  const diagnostics = {errors: [], errorCount: 0, skippedBranches: 0, stageCounts: {}};
  const battle = State.deserializeBattle(payload.state);
  resetBattleLog(battle);
  const values = [];
  let rootOpponentChoices = [];
  try {
    rootOpponentChoices = legalChoicesForSide(battle, otherSide(side), maxChoices);
  } catch (error) {
    noteError(diagnostics, "root-opponent-legal-choices", error, {
      side,
      depth,
      ...battleContext(battle),
    });
  }
  if (!rootOpponentChoices.length) rootOpponentChoices = [""];
  for (const rawCandidate of payload.candidates || []) {
    const candidate = normalizeChoice(rawCandidate);
    if (!candidate || candidate === "forfeit") {
      values.push(-1);
      continue;
    }
    try {
      values.push(
        candidateValue(
          battle,
          side,
          candidate,
          depth,
          maxChoices,
          policy,
          robustWorstWeight,
          diagnostics,
          rootOpponentChoices
        )
      );
    } catch (error) {
      noteError(diagnostics, "root-candidate", error, {
        side,
        candidate,
        depth,
        ...battleContext(battle),
      });
      values.push(scoreBattle(battle, side));
    }
  }
  return {
    ok: true,
    values,
    turn: battle.turn,
    repairs: 0,
    simulation_errors: diagnostics.errorCount,
    skipped_branches: diagnostics.skippedBranches,
    error_stage_counts: diagnostics.stageCounts,
    errors: diagnostics.errors,
    opponent_policy: policy,
    robust_worst_weight: clamp(robustWorstWeight, 0, 1),
    request_state: battle.requestState,
  };
}

function offlineStart(payload) {
  const battle = createOfflineBattle(payload);
  return {ok: true, battle: exportOfflineState(battle, Number(payload.max_choices || 0))};
}

function offlineState(payload) {
  const battle = State.deserializeBattle(payload.state);
  return {ok: true, battle: exportOfflineState(battle, Number(payload.max_choices || 0))};
}

function offlineChoose(payload) {
  const battle = State.deserializeBattle(payload.state);
  resetBattleLog(battle);
  const choices = payload.choices || {};
  const p1Choice = normalizeChoice(choices.p1 || "");
  const p2Choice = normalizeChoice(choices.p2 || "");
  const applied = [];
  if (p1Choice) {
    battle.choose("p1", p1Choice);
    applied.push(["p1", p1Choice]);
  }
  if (p2Choice) {
    battle.choose("p2", p2Choice);
    applied.push(["p2", p2Choice]);
  }
  resetBattleLog(battle);
  return {
    ok: true,
    applied,
    battle: exportOfflineState(battle, Number(payload.max_choices || 0)),
  };
}

async function handle(req, res) {
  if (req.method === "GET" && req.url === "/health") {
    return jsonResponse(res, 200, {ok: true});
  }
  if (req.method !== "POST") {
    return jsonResponse(res, 404, {ok: false, error: "not found"});
  }
  try {
    const body = await readBody(req);
    const payload = JSON.parse(body || "{}");
    if (req.url === "/evaluate") {
      return jsonResponse(res, 200, evaluate(payload));
    }
    if (req.url === "/offline/start") {
      return jsonResponse(res, 200, offlineStart(payload));
    }
    if (req.url === "/offline/state") {
      return jsonResponse(res, 200, offlineState(payload));
    }
    if (req.url === "/offline/evaluate") {
      return jsonResponse(res, 200, evaluateOffline(payload));
    }
    if (req.url === "/offline/choose") {
      return jsonResponse(res, 200, offlineChoose(payload));
    }
    return jsonResponse(res, 404, {ok: false, error: "not found"});
  } catch (error) {
    console.error(error?.stack || error);
    return jsonResponse(res, 500, {ok: false, error: String(error?.stack || error)});
  }
}

if (process.argv.includes("--self-test")) {
  console.log(JSON.stringify({ok: true, exports: {Battle: !!Battle, Teams: !!Teams, State: !!State}}));
} else {
  http.createServer(handle).listen(PORT, "0.0.0.0", () => {
    console.log(`showdown-sim listening on ${PORT}`);
  });
}
