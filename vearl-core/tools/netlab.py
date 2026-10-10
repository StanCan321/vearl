"""Test bench of REAL `node` processes on one machine: genesis, start/abrupt stop, state. Used by soak.py
(endurance) and fork_test.py (partition then merge of two chains)."""
import json, os, re, signal, subprocess, sys, tempfile, time, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
import chain_ref as C

B = os.environ.get("B", "target/release")
CH = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _polymod(v):
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    c = 1
    for x in v:
        b = c >> 25
        c = (c & 0x1FFFFFF) << 5 ^ x
        for i in range(5):
            c ^= gen[i] if (b >> i) & 1 else 0
    return c


def bech32m(hrp, data):
    acc = bits = 0
    five = []
    for b in data:
        acc = (acc << 8) | b
        bits += 8
        while bits >= 5:
            bits -= 5
            five.append((acc >> bits) & 31)
    if bits:
        five.append((acc << (5 - bits)) & 31)
    exp = [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]
    pm = _polymod(exp + five + [0] * 6) ^ 0x2BC830A3
    return hrp + "1" + "".join(CH[d] for d in five + [(pm >> 5 * (5 - i)) & 31 for i in range(6)])


def addr(seed):
    """vrl1... address of the PUBLIC seed key `seed` (devnet only)."""
    return bech32m("vrl", bytes.fromhex(C.pub_hex(seed)))


def run(*args, timeout=60):
    r = subprocess.run([f"{B}/{args[0]}"] + list(args[1:]), capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


class Lab:
    def __init__(self, nodes, funded=range(1, 7), window=120, divisor=256, retarget="Fixed", faucet_seed=7, n=128):
        """nodes: {name: dict(port, seed|None, api?, extra=[], peers=[names])}"""
        self.dir = tempfile.mkdtemp(prefix="vearl-lab-")
        alloc = []
        for s in funded:
            alloc += ["--alloc", f"{addr(s)}=200000"]
        g = run("vearl-genesis", "--network", "lab", "--faucet", addr(faucet_seed), "--retarget", retarget, "--target-divisor", str(divisor), *alloc)
        gj = json.loads(g)
        gj["timing"] = {"challenge_secs": window, "serve_secs": 15}
        gj["params"]["n"] = n   # width of the chain (multiple of 128, up to 1024)
        self.genesis = f"{self.dir}/genesis.json"
        json.dump(gj, open(self.genesis, "w"), indent=1)
        self.nodes, self.procs, self.logs = nodes, {}, {}

    def start(self, name, peers=None):
        n = self.nodes[name]
        cmd = [f"{B}/node", "--genesis", self.genesis, "--listen", f"127.0.0.1:{n['port']}", "--data-dir", f"{self.dir}/{name}"]
        if n["seed"]:
            cmd += ["--mine-seed", str(n["seed"]), "--insecure-devnet-keys"]
        if n.get("api"):
            cmd += ["--api", f"127.0.0.1:{n['api']}"]
        for p in (n["peers"] if peers is None else peers):
            cmd += ["--peer", f"127.0.0.1:{self.nodes[p]['port']}"]
        cmd += n["extra"]
        self.logs[name] = open(f"{self.dir}/{name}.log", "ab")
        env = dict(os.environ, VEARL_DEBUG_MEMPOOL="1")
        self.procs[name] = subprocess.Popen(cmd, stdout=self.logs[name], stderr=subprocess.STDOUT, env=env)

    def kill(self, name):
        self.procs[name].send_signal(signal.SIGKILL)
        self.procs[name].wait()

    def addrof(self, name):
        return f"127.0.0.1:{self.nodes[name]['port']}"

    def status(self, name):
        o = run("vearl-cli", "status", self.addrof(name))
        m = re.search(r"height: (\d+), tip: \[([\d, ]+)\], work: (\d+), mempool: (\d+)", o)
        return (int(m.group(1)), tuple(int(x) for x in m.group(2).split(", ")), int(m.group(3)), int(m.group(4))) if m else None

    def api_status(self, name):
        port = self.nodes[name].get("api")
        if not port:
            return None
        try:
            return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5))
        except Exception:
            return None

    def stop_all(self):
        for p in self.procs.values():
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
