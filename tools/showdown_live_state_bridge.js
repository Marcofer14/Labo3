"use strict";

const http = require("http");
const {URL} = require("url");

const DEFAULT_PORT = 9002;

function jsonResponse(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

function normalizeRoomID(roomid) {
  return String(roomid || "").trim().toLowerCase();
}

async function getBattleState(roomid) {
  const id = normalizeRoomID(roomid);
  if (!id) {
    return {ok: false, error: "missing roomid"};
  }
  const room = global.Rooms?.get(id);
  if (!room) {
    return {ok: false, error: `room not found: ${id}`};
  }
  if (!room.battle) {
    return {ok: false, error: `room is not a battle: ${id}`};
  }
  if (typeof room.battle.getSerializedState !== "function") {
    return {ok: false, error: "live-state patch missing getSerializedState"};
  }
  const state = await room.battle.getSerializedState();
  return {
    ok: true,
    roomid: id,
    turn: room.battle.turn || 0,
    ended: Boolean(room.battle.ended),
    state,
  };
}

function authorized(req) {
  const token = process.env.SHOWDOWN_LIVE_STATE_TOKEN || "";
  if (!token) return true;
  return req.headers["x-showdown-state-token"] === token;
}

function start() {
  if (global.__showdownLiveStateBridgeStarted) return;
  global.__showdownLiveStateBridgeStarted = true;
  const port = Number(process.env.SHOWDOWN_LIVE_STATE_PORT || DEFAULT_PORT);
  const server = http.createServer(async (req, res) => {
    try {
      const parsed = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
      if (parsed.pathname === "/health") {
        return jsonResponse(res, 200, {ok: true});
      }
      if (!authorized(req)) {
        return jsonResponse(res, 403, {ok: false, error: "forbidden"});
      }
      if (req.method === "GET" && parsed.pathname === "/battle-state") {
        const result = await getBattleState(parsed.searchParams.get("roomid"));
        return jsonResponse(res, result.ok ? 200 : 404, result);
      }
      return jsonResponse(res, 404, {ok: false, error: "not found"});
    } catch (error) {
      return jsonResponse(res, 500, {ok: false, error: String(error?.stack || error)});
    }
  });
  server.listen(port, "0.0.0.0", () => {
    console.log(`showdown-live-state listening on ${port}`);
  });
}

module.exports = {start};
