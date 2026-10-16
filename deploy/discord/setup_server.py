#!/usr/bin/env python3
"""Builds the Vearl Discord server (prototype): roles, categories, channels, permissions and the first messages.

    python deploy/discord/setup_server.py            show the plan only, touches nothing (no token needed)
    python deploy/discord/setup_server.py --apply    create everything on the server

The bot token is never taken from the command line and never written anywhere: it comes from the environment variable
DISCORD_BOT_TOKEN, or is typed hidden when the script asks for it. The server id comes from DISCORD_GUILD_ID or the question.
Safe to run again: roles and channels that already exist are kept, and a channel is only filled with its first message
when it is empty.

The bot needs these permissions on the server (not Administrator): Manage Roles, Manage Channels, View Channels,
Send Messages, Read Message History. Only the Python standard library is used.
"""
import getpass, json, os, sys, time, urllib.error, urllib.request

BASE = os.environ.get("DISCORD_API_BASE", "https://discord.com/api/v10")

VIEW, KICK, BAN, SEND, MANAGE_MESSAGES, MODERATE = 1 << 10, 1 << 1, 1 << 2, 1 << 11, 1 << 13, 1 << 40

# name, colour (the site's palette), permissions, shown apart in the member list
ROLES = [
    ("Team", 0xF4BCD6, KICK | BAN | MANAGE_MESSAGES | MODERATE, True),
    ("Moderator", 0xB9A6FF, KICK | BAN | MANAGE_MESSAGES | MODERATE, True),
    ("Miner", 0x9FF0E2, 0, False),
]

SITE, PAPER, MAIL = "https://vearl.org", "https://vearl.org/whitepaper", "contact@vearl.org"
DISCORD = "https://discord.gg/vearl"

WELCOME = (
    "**Welcome to Vearl.**\n\n"
    "Vearl is **Proof of Useful Work · Verified Compute**: GPU miners run real AI models for clients, and every answer is "
    "verified before anyone is paid. *Bitcoin mines hashes. Vearl mines AI.*\n\n"
    "**Start here**\n"
    "1. Read #rules\n"
    "2. Follow #announcements\n"
    "3. Questions? Ask in #questions\n\n"
    f"Website: {SITE}\nWhitepaper: {PAPER}\n\n"
    "Public testnet: planned for **23 October 2026**."
)
RULES = (
    "**Rules**\n\n"
    "1. **Be respectful.** No insults, harassment or discrimination.\n"
    "2. **No spam**, referral links, or unsolicited promotion of other projects.\n"
    "3. **No price talk and no financial advice.** Testnet tokens have no value and are not for sale.\n"
    "4. **Stay safe.** The team will never message you first, and will never ask for a seed phrase, private key or password. "
    "Report anyone who does.\n"
    "5. **English** in the main channels, so that everyone can follow.\n"
    "6. **Bugs:** post them in #bug-reports with what you did, what you expected and what happened.\n"
    "7. Moderators may remove content or ban accounts to protect the community.\n"
)
LINKS = (
    "**Official links**\n\n"
    f"Website: {SITE}\nWhitepaper: {PAPER}\nContact: {MAIL}\nDiscord: {DISCORD}\n\n"
    "Anything else claiming to be Vearl is not official."
)
TESTNET = (
    "**Public testnet**\n\n"
    "Status: closed testnet today. The **public testnet is planned for 23 October 2026**: several machines, external miners "
    "and a block explorer.\n\n"
    "**Mining requirements:** an NVIDIA GPU with compute capability 8.0 or newer (Ampere and later: RTX 30/40 series, A10, A100, H100). "
    "CPU mining works for testing only.\n\n"
    "Miners who help secure the public testnet are planned to receive Vearl tokens on mainnet as a thank-you. This is a plan, "
    "not a guarantee: the rules and amounts will be announced, and there is no mainnet date yet. "
    "**Testnet tokens themselves have no value and are not for sale.**"
)
FAQ = (
    "**FAQ**\n\n"
    "**What is Proof of Useful Work?** Miners multiply 8-bit integer matrices on GPUs. The same products can belong to a "
    "client's AI job, so the work that secures the chain also produces a result someone needs.\n\n"
    "**How is a result checked?** The client re-checks it with Freivalds' test (worst-case error 2^-24 with 24 rounds) or exact "
    "recomputation. One wrong operation can be proven with a proof of about a kilobyte, and the miner loses its bond.\n\n"
    "**Which AI models run today?** TinyStories-33M (text), DeiT-Tiny (photos), BGE-base and MiniLM (text embeddings), ResNet-20, MNIST. "
    "They are a first generation and will be considerably improved.\n\n"
    "**Has the code been audited?** Not by a third party yet. See the limits in the whitepaper.\n\n"
    f"More: {PAPER}"
)

# (category, private, [(channel, topic, read_only, first message or None)])
STRUCTURE = [
    ("START HERE", False, [
        ("welcome", "Start here", True, WELCOME),
        ("rules", "Community rules", True, RULES),
        ("announcements", "Official news only", True, None),
        ("links", "Official links", True, LINKS),
    ]),
    ("COMMUNITY", False, [
        ("general", "Talk about Vearl, mining and verified AI", False, None),
        ("introductions", "Say hello", False, None),
        ("questions", "Ask anything about Vearl", False, None),
        ("faq", "Frequently asked questions", True, FAQ),
    ]),
    ("TESTNET AND MINING", False, [
        ("public-testnet", "Planned for 23 October 2026", True, TESTNET),
        ("mining-help", "Set-up help for miners", False, None),
        ("gpu-benchmarks", "Share your GPU results", False, None),
        ("bug-reports", "Report what does not work", False, None),
    ]),
    ("TECHNICAL", False, [
        ("protocol-and-research", "Consensus, proofs, verification", False, None),
        ("verified-ai-workloads", "Models, quantization, accuracy", False, None),
    ]),
    ("TEAM", True, [
        ("team-chat", "Private: Team and Moderators", False, None),
    ]),
]


class ApiError(Exception):
    pass


class Discord:
    def __init__(self, token):
        self.headers = {"Authorization": "Bot " + token, "Content-Type": "application/json",
                        "User-Agent": "DiscordBot (https://vearl.org, 1.0)"}

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        for _ in range(8):
            req = urllib.request.Request(BASE + path, data=data, method=method, headers=self.headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    raw = r.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")
                if e.code == 429:                                   # rate limit: wait as told, then retry
                    try:
                        wait = float(json.loads(text).get("retry_after", 1))
                    except Exception:
                        wait = 1.0
                    time.sleep(wait + 0.25)
                    continue
                raise ApiError("%s %s -> HTTP %d: %s" % (method, path, e.code, text[:300]))
        raise ApiError("%s %s: too many rate-limit retries" % (method, path))


def plan():
    print("Plan (nothing is created yet):")
    print("  roles:", ", ".join(r[0] for r in ROLES))
    for cat, private, chans in STRUCTURE:
        print("  [%s]%s" % (cat, "  (private: Team and Moderator only)" if private else ""))
        for name, topic, ro, msg in chans:
            print("      #%-24s %s%s%s" % (name, topic, "  [read-only]" if ro else "", "  [first message]" if msg else ""))


def apply(token, guild):
    d = Discord(token)
    roles = {r["name"]: r for r in d.call("GET", "/guilds/%s/roles" % guild)}
    made = []
    for name, color, perms, hoist in ROLES:
        if name not in roles:
            roles[name] = d.call("POST", "/guilds/%s/roles" % guild,
                                 {"name": name, "color": color, "permissions": str(perms), "hoist": hoist, "mentionable": False})
            made.append("role " + name)
            time.sleep(0.4)
    everyone, team, mod = guild, roles["Team"]["id"], roles["Moderator"]["id"]
    existing = d.call("GET", "/guilds/%s/channels" % guild)
    by_key = {(c["name"].lower(), c.get("parent_id")): c for c in existing}

    def overwrites(private, read_only):
        o = []
        if private:
            o.append({"id": everyone, "type": 0, "allow": "0", "deny": str(VIEW)})
            for rid in (team, mod):
                o.append({"id": rid, "type": 0, "allow": str(VIEW | SEND), "deny": "0"})
        elif read_only:
            o.append({"id": everyone, "type": 0, "allow": "0", "deny": str(SEND)})
            for rid in (team, mod):
                o.append({"id": rid, "type": 0, "allow": str(SEND), "deny": "0"})
        return o

    for cat, private, chans in STRUCTURE:
        c = by_key.get((cat.lower(), None))
        if not c:
            c = d.call("POST", "/guilds/%s/channels" % guild,
                       {"name": cat, "type": 4, "permission_overwrites": overwrites(private, False)})
            by_key[(cat.lower(), None)] = c
            made.append("category " + cat)
            time.sleep(0.4)
        for name, topic, ro, msg in chans:
            ch = by_key.get((name, c["id"]))
            if not ch:
                ch = d.call("POST", "/guilds/%s/channels" % guild,
                            {"name": name, "type": 0, "topic": topic, "parent_id": c["id"],
                             "permission_overwrites": overwrites(private, ro)})
                by_key[(name, c["id"])] = ch
                made.append("channel #" + name)
                time.sleep(0.4)
            if msg and not d.call("GET", "/channels/%s/messages?limit=1" % ch["id"]):
                d.call("POST", "/channels/%s/messages" % ch["id"], {"content": msg[:2000]})
                made.append("message in #" + name)
                time.sleep(0.6)
    print("Done. Created: %s" % (", ".join(made) if made else "nothing (everything already exists)"))
    return made


def main():
    plan()
    if "--apply" not in sys.argv:
        print("\nThis was only the plan. Run again with --apply to create it on your server.")
        return
    token = os.environ.get("DISCORD_BOT_TOKEN") or getpass.getpass("Bot token (hidden, not stored): ").strip()
    guild = os.environ.get("DISCORD_GUILD_ID") or input("Server id (Settings > Advanced > Developer Mode, right-click the server > Copy Server ID): ").strip()
    if not token or not guild.isdigit():
        sys.exit("A token and a numeric server id are needed.")
    try:
        apply(token, guild)
    except ApiError as e:
        sys.exit("Discord refused: %s\nCheck that the bot is on the server with Manage Roles, Manage Channels, View Channels, Send Messages." % e)


if __name__ == "__main__":
    main()
