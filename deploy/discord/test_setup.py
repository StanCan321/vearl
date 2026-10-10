#!/usr/bin/env python3
"""Tests setup_server.py against a small fake Discord API on localhost (test values only, no real token or server)."""
import json, os, re, sys, threading, http.server

os.environ["DISCORD_API_BASE"] = "http://127.0.0.1:0"           # replaced below once the fake server has a port
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

state = {"roles": [{"id": "900", "name": "@everyone"}], "channels": [], "messages": {}, "next": 1000, "seen_auth": set(), "calls": 0}


class Fake(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        raw = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def handle_any(self, method):
        state["seen_auth"].add(self.headers.get("Authorization"))
        state["calls"] += 1
        p = self.path.split("?")[0]
        if method == "GET" and re.fullmatch(r"/guilds/\d+/roles", p):
            return self._send(state["roles"])
        if method == "POST" and re.fullmatch(r"/guilds/\d+/roles", p):
            b = self._body(); r = dict(b, id=str(state["next"])); state["next"] += 1; state["roles"].append(r); return self._send(r)
        if method == "GET" and re.fullmatch(r"/guilds/\d+/channels", p):
            return self._send(state["channels"])
        if method == "POST" and re.fullmatch(r"/guilds/\d+/channels", p):
            b = self._body(); c = dict(b, id=str(state["next"])); state["next"] += 1
            c["parent_id"] = b.get("parent_id"); state["channels"].append(c); state["messages"][c["id"]] = []; return self._send(c)
        m = re.fullmatch(r"/channels/(\d+)/messages", p)
        if m and method == "GET":
            return self._send(state["messages"].get(m.group(1), [])[:1])
        if m and method == "POST":
            state["messages"][m.group(1)].append(self._body()); return self._send({"id": "m"})
        self._send({"message": "unknown"}, 404)

    def do_GET(self): self.handle_any("GET")
    def do_POST(self): self.handle_any("POST")


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["DISCORD_API_BASE"] = "http://127.0.0.1:%d" % srv.server_address[1]
import setup_server as s
s.BASE = os.environ["DISCORD_API_BASE"]

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond); print(("PASS  " if cond else "FAIL  ") + name)

made = s.apply("test-token-not-real", "900")
names = {c["name"]: c for c in state["channels"]}
check("3 roles created", [r["name"] for r in state["roles"][1:]] == ["Team", "Moderator", "Miner"])
check("5 categories and 15 channels", sum(1 for c in state["channels"] if c["type"] == 4) == 5 and sum(1 for c in state["channels"] if c["type"] == 0) == 15)
check("channels sit under their category", names["rules"]["parent_id"] == names["START HERE"]["id"] and names["team-chat"]["parent_id"] == names["TEAM"]["id"])
check("read-only channel denies SEND to @everyone", any(o["id"] == "900" and int(o["deny"]) == 1 << 11 for o in names["rules"]["permission_overwrites"]))
check("read-only channel lets the Team send", any(o["id"] == [r for r in state["roles"] if r["name"] == "Team"][0]["id"] and int(o["allow"]) & (1 << 11) for o in names["rules"]["permission_overwrites"]))
check("a normal channel has no overwrite", names["general"]["permission_overwrites"] == [])
check("team category hides VIEW from @everyone", any(o["id"] == "900" and int(o["deny"]) == 1 << 10 for o in names["TEAM"]["permission_overwrites"]))
check("team channel is private too", any(o["id"] == "900" and int(o["deny"]) == 1 << 10 for o in names["team-chat"]["permission_overwrites"]))
check("first messages posted in 5 channels", sum(1 for v in state["messages"].values() if v) == 5)
check("every message fits Discord's 2000 characters", all(len(m["content"]) <= 2000 for v in state["messages"].values() for m in v))
check("no admin permission on any role", all(not int(r.get("permissions", "0")) & 8 for r in state["roles"][1:]))
check("token is sent as 'Bot <token>' only", state["seen_auth"] == {"Bot test-token-not-real"})
before = (len(state["channels"]), len(state["roles"]), sum(len(v) for v in state["messages"].values()))
again = s.apply("test-token-not-real", "900")
check("second run creates nothing (idempotent)", again == [] and before == (len(state["channels"]), len(state["roles"]), sum(len(v) for v in state["messages"].values())))
print("\nALL PASS" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
