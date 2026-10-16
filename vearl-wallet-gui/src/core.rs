//! Everything the window does that is not drawing: wallet files, the network client, the activity list, the registration.
//! It only uses the functions of `vearl-core` (the same ones as the command-line wallet), so a wallet created here opens with
//! `vearl-wallet` and the other way round.
use bip39::Mnemonic;
use std::path::{Path, PathBuf};
use vearl_core::crypto::{pubkey, PubKey};
use vearl_core::net::{request, Msg};
use vearl_core::state::COIN;
use vearl_core::types::{SignedTx, Tx, TxKind};
use vearl_core::wallet::*;

pub const SYMBOL: &str = "VRL";
/// Used when no `network.json` sits next to the program: the seed nodes of the public testnet.
pub const DEFAULT_NODES: [&str; 3] = ["seed.vearl.org:9000", "vearl.org:9000", "103.89.12.219:9000"];
pub const DEFAULT_EXPLORER: &str = "https://vearl.org";
pub const FEE_DEFAULT: &str = "0.01";

/// Where the program keeps its files: `%APPDATA%\Vearl` on Windows, `~/.local/share/vearl` (or `$XDG_DATA_HOME/vearl`) elsewhere.
pub fn data_dir() -> PathBuf {
    // VEARL_WALLET_HOME: keep every file of the program in one folder of your choice (portable use, recordings, tests)
    if let Some(h) = std::env::var_os("VEARL_WALLET_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(h);
    }
    #[cfg(windows)]
    {
        if let Some(a) = std::env::var_os("APPDATA") {
            return PathBuf::from(a).join("Vearl");
        }
    }
    if let Some(x) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(x).join("vearl");
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".local").join("share").join("vearl")
}

pub fn wallets_dir() -> PathBuf {
    data_dir().join("wallets")
}

// ------------------------------------------------------------------------------------------------ settings

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Node to ask first ("HOST:PORT"); empty = the seed nodes of the release (network.json).
    pub node: String,
    /// Block explorer used for the activity list and the links.
    pub explorer: String,
    /// Wallet files opened from other folders.
    pub extra_wallets: Vec<String>,
    pub last_wallet: String,
    /// Back to the start screen after this many minutes without any input (0 = never).
    pub autolock_minutes: u32,
    /// Windows: hide the window from screenshots and recordings while the recovery words are on screen.
    pub protect_capture: bool,
    /// No animations (accessibility, slow computers).
    pub reduce_motion: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { node: String::new(), explorer: DEFAULT_EXPLORER.into(), extra_wallets: vec![], last_wallet: String::new(), autolock_minutes: 10, protect_capture: true, reduce_motion: false }
    }
}

impl Settings {
    fn file() -> PathBuf {
        data_dir().join("settings.json")
    }
    pub fn load() -> Settings {
        let mut s = Settings::default();
        if let Ok(txt) = std::fs::read_to_string(Self::file()) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                let get = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
                if let Some(n) = get("node") {
                    s.node = n;
                }
                if let Some(e) = get("explorer").filter(|e| !e.is_empty()) {
                    s.explorer = e;
                }
                if let Some(l) = get("last_wallet") {
                    s.last_wallet = l;
                }
                if let Some(m) = v.get("autolock_minutes").and_then(|x| x.as_u64()) {
                    s.autolock_minutes = m.min(240) as u32;
                }
                if let Some(b) = v.get("protect_capture").and_then(|x| x.as_bool()) {
                    s.protect_capture = b;
                }
                if let Some(b) = v.get("reduce_motion").and_then(|x| x.as_bool()) {
                    s.reduce_motion = b;
                }
                if let Some(a) = v.get("extra_wallets").and_then(|x| x.as_array()) {
                    s.extra_wallets = a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect();
                }
            }
        }
        s
    }
    pub fn save(&self) {
        let v = serde_json::json!({
            "node": self.node, "explorer": self.explorer, "extra_wallets": self.extra_wallets, "last_wallet": self.last_wallet,
            "autolock_minutes": self.autolock_minutes, "protect_capture": self.protect_capture, "reduce_motion": self.reduce_motion,
        });
        let _ = std::fs::create_dir_all(data_dir());
        let f = Self::file();
        let _ = std::fs::write(&f, serde_json::to_string_pretty(&v).unwrap());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600));
        }
    }
}

// ------------------------------------------------------------------------------------------------ network.json

#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    pub name: String,
    pub nodes: Vec<String>,
    pub explorer: Option<String>,
}

/// Reads the `network.json` of the release (next to the program) when there is one: seed nodes are written "HOST:PORT@PUBLIC_KEY".
pub fn load_network() -> Network {
    let dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    let from_file = dir.and_then(|d| std::fs::read_to_string(d.join("network.json")).ok()).and_then(|t| parse_network(&t));
    from_file.unwrap_or_else(|| Network { name: "vearl-testnet".into(), nodes: DEFAULT_NODES.iter().map(|s| s.to_string()).collect(), explorer: None })
}

pub fn parse_network(text: &str) -> Option<Network> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let nodes: Vec<String> = v.get("peers")?.as_array()?.iter().filter_map(|p| p.as_str()).map(|p| p.split('@').next().unwrap_or(p).trim().to_string()).filter(|p| !p.is_empty()).collect();
    if nodes.is_empty() {
        return None;
    }
    Some(Network { name: v.get("name").and_then(|n| n.as_str()).unwrap_or("vearl-testnet").to_string(), nodes, explorer: v.get("explorer").and_then(|e| e.as_str()).map(str::to_string) })
}

/// The nodes to try, in order: the one chosen in the settings, then those of the release.
pub fn node_list(settings: &Settings, net: &Network) -> Vec<String> {
    let mut out = vec![];
    let own = settings.node.trim();
    if !own.is_empty() {
        out.push(own.to_string());
    }
    for n in &net.nodes {
        if !out.contains(n) {
            out.push(n.clone());
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ wallet files

#[derive(Clone, Debug)]
pub struct WalletEntry {
    pub path: PathBuf,
    pub name: String,
    pub address: String,
}

pub fn entry_of(path: &Path) -> Result<WalletEntry, String> {
    let ks = Keystore::load(path)?;
    parse_address(&ks.address)?;
    let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "wallet".into());
    Ok(WalletEntry { path: path.to_path_buf(), name, address: ks.address })
}

/// Wallet files found in the wallet folder, next to the program (wallets made with `vearl-wallet new`) and in the settings.
pub fn list_wallets(settings: &Settings) -> Vec<WalletEntry> {
    let mut dirs = vec![wallets_dir()];
    if let Some(d) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        dirs.push(d);
    }
    let mut out: Vec<WalletEntry> = vec![];
    let mut seen = std::collections::HashSet::new();
    let mut consider = |p: PathBuf, out: &mut Vec<WalletEntry>| {
        let key = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
        if seen.insert(key) {
            if let Ok(e) = entry_of(&p) {
                out.push(e);
            }
        }
    };
    for d in dirs {
        if let Ok(rd) = std::fs::read_dir(&d) {
            let mut files: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json") && p.is_file()).collect();
            files.sort();
            for f in files {
                // the files of the release (network.json, genesis.json...) are not keystores: entry_of() rejects them
                consider(f, &mut out);
            }
        }
    }
    for p in &settings.extra_wallets {
        consider(PathBuf::from(p), &mut out);
    }
    out
}

/// A file name typed by the user -> a path inside the wallet folder ("my wallet" -> my-wallet.json). Refuses anything with folders.
pub fn wallet_path_for(name: &str) -> Result<PathBuf, String> {
    let n = name.trim();
    if n.is_empty() {
        return Err("give the wallet a name".into());
    }
    if n.len() > 60 || !n.chars().all(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.')) || n.starts_with('.') {
        return Err("use letters, digits, spaces, - and _ only (60 characters at most)".into());
    }
    let stem = n.trim_end_matches(".json").replace(' ', "-");
    Ok(wallets_dir().join(format!("{stem}.json")))
}

/// Writes a new wallet file (never overwrites): the phrase is encrypted with the password (Argon2id + ChaCha20-Poly1305).
pub fn create_wallet(path: &Path, mnemonic: &Mnemonic, password: &str) -> Result<WalletEntry, String> {
    if path.exists() {
        return Err(format!("{} already exists: choose another name", path.display()));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let ks = Keystore::create(mnemonic, password, KdfParams::STRONG)?;
    ks.save(path)?;
    // read it back and unlock it once: a wallet that cannot be opened must never be announced as created
    let back = Keystore::load(path)?.unlock(password)?;
    if back.to_entropy() != mnemonic.to_entropy() {
        let _ = std::fs::remove_file(path);
        return Err("the wallet file could not be verified after writing".into());
    }
    entry_of(path).map_err(|_| "the wallet file could not be read back".to_string())
}

pub fn unlock(path: &Path, password: &str) -> Result<Mnemonic, String> {
    Keystore::load(path)?.unlock(password)
}

/// Re-encrypts the wallet with a new password: the new file is written and checked beside the old one, then replaces it.
pub fn change_password(path: &Path, old: &str, new: &str) -> Result<(), String> {
    let m = unlock(path, old)?;
    let ks = Keystore::create(&m, new, KdfParams::STRONG)?;
    let tmp = path.with_extension("json.new");
    let _ = std::fs::remove_file(&tmp);
    ks.save(&tmp)?;
    if Keystore::load(&tmp)?.unlock(new)?.to_entropy() != m.to_entropy() {
        let _ = std::fs::remove_file(&tmp);
        return Err("the new file could not be verified".into());
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        e.to_string()
    })
}

/// Words of the recovery phrase typed by the user -> a phrase. Case, extra spaces and line breaks do not matter.
pub fn clean_phrase(text: &str) -> String {
    text.split_whitespace().map(|w| w.to_lowercase()).collect::<Vec<_>>().join(" ")
}

/// Words that are not in the BIP-39 list (to flag them while the user types).
pub fn unknown_words(phrase: &str) -> Vec<String> {
    let list = bip39::Language::English.word_list();
    phrase.split_whitespace().filter(|w| !list.contains(&w.to_lowercase().as_str())).map(str::to_string).collect()
}

pub fn password_strength(p: &str) -> (u8, &'static str) {
    let n = p.chars().count();
    let classes = [p.chars().any(|c| c.is_lowercase()), p.chars().any(|c| c.is_uppercase()), p.chars().any(|c| c.is_ascii_digit()), p.chars().any(|c| !c.is_alphanumeric())].iter().filter(|b| **b).count();
    if n < 8 {
        (0, "too short (8 characters minimum)")
    } else if n < 12 || classes < 2 {
        (1, "weak")
    } else if n < 16 || classes < 3 {
        (2, "fair")
    } else {
        (3, "strong")
    }
}

// ------------------------------------------------------------------------------------------------ amounts

/// 1234567.5 -> "1,234,567.50000000" with the integer part in groups of three.
pub fn fmt_grouped(v: u64) -> (String, String) {
    let int = (v / COIN).to_string();
    let mut out = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    (out, format!("{:08}", v % COIN))
}

/// "vrl1yhur00d2...xqzvvp76". Works on characters, not bytes: the text can come from a server (the explorer) and must never panic the window.
pub fn short_addr(a: &str) -> String {
    let c: Vec<char> = a.chars().collect();
    if c.len() > 24 {
        format!("{}...{}", c[..12].iter().collect::<String>(), c[c.len() - 8..].iter().collect::<String>())
    } else {
        a.to_string()
    }
}

/// At most `n` characters of a text received from a server.
fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).filter(|c| !c.is_control()).collect()
}

// ------------------------------------------------------------------------------------------------ the node

#[derive(Clone, Debug, PartialEq)]
pub struct NodeStatus {
    pub node: String,
    pub height: u64,
    pub peers: usize,
    pub mempool: usize,
    /// `Some(true)`: the node is on the network of your release (same genesis hash). `Some(false)`: ANOTHER network. `None`: not checked
    /// (no genesis.json next to the program).
    pub genesis_ok: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Account {
    pub balance: u64,
    pub nonce: u64,
    pub next_nonce: u64,
}

fn ask(node: &str, m: &Msg) -> Result<Msg, String> {
    request(node, m).map_err(|e| format!("{node}: {e}"))
}

/// The genesis hash of the release (genesis.json next to the program): the network this wallet is meant for.
pub fn expected_genesis() -> Option<[u8; 32]> {
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let txt = std::fs::read_to_string(dir.join("genesis.json")).ok()?;
    let (_, cfg) = vearl_core::genesis::parse_genesis(&txt).ok()?;
    Some(vearl_core::net::genesis_hash(&cfg))
}

/// Asks one node for its status and checks that it is on the expected network.
pub fn status_of(node: &str, expected: Option<[u8; 32]>) -> Result<NodeStatus, String> {
    let mut c = vearl_core::net::Conn::connect(node).map_err(|e| format!("{node}: {e}"))?;
    c.set_timeout(std::time::Duration::from_secs(10));
    c.send(&Msg::GetStatus).map_err(|e| format!("{node}: {e}"))?;
    let mut genesis: Option<[u8; 32]> = None;
    for _ in 0..8 {
        match c.recv().map_err(|e| format!("{node}: {e}"))? {
            Msg::Hello { genesis: g, .. } => genesis = Some(g),
            Msg::Status { height, peers, mempool, .. } => {
                let genesis_ok = match (expected, genesis) {
                    (Some(e), Some(g)) => Some(e == g),
                    _ => None,
                };
                return Ok(NodeStatus { node: node.to_string(), height, peers, mempool, genesis_ok });
            }
            _ => continue,
        }
    }
    Err(format!("{node}: no status in the reply"))
}

/// First node of the list that answers: its status.
pub fn node_status(nodes: &[String]) -> Result<NodeStatus, String> {
    let expected = expected_genesis();
    let mut last = "no node configured".to_string();
    for n in nodes {
        match status_of(n, expected) {
            Ok(st) => return Ok(st),
            Err(e) => last = e,
        }
    }
    Err(format!("no node answers ({last})"))
}

pub fn account_of(node: &str, pk: PubKey) -> Result<Account, String> {
    match ask(node, &Msg::GetAccount(pk))? {
        Msg::Account { balance, nonce, next_nonce } => Ok(Account { balance, nonce, next_nonce }),
        m => Err(format!("unexpected reply: {m:?}").chars().take(120).collect()),
    }
}

/// Signs and submits a transfer exactly as `vearl-wallet send` does. Returns the transaction id and the address it was paid from.
pub fn send_transfer(node: &str, mnemonic: &Mnemonic, to: &str, amount: u64, fee: u64, pq: bool) -> Result<(String, String), String> {
    let to = parse_address(to)?;
    // never sign for a node that is on another network than the release of this program
    if status_of(node, expected_genesis())?.genesis_ok == Some(false) {
        return Err("This node is on a different network than your release (the genesis hash does not match). Nothing was signed or sent.".into());
    }
    let sk = derive_account(mnemonic, 0);
    let pqk = pq.then(|| derive_pq_account(mnemonic, 0));
    let from = pqk.as_ref().map_or_else(|| pubkey(&sk), |k| k.pubkey());
    let from_addr = if pq { pq_address_of(&from) } else { address_of(&from) };
    let acc = account_of(node, from)?;
    if acc.balance < amount.saturating_add(fee) {
        return Err(format!("insufficient balance: {} {SYMBOL} available on {}", fmt_amount(acc.balance), short_addr(&from_addr)));
    }
    let tx0 = Tx { from, nonce: acc.next_nonce, fee, kind: TxKind::Transfer { to, amount } };
    let tx = match &pqk {
        Some(k) => SignedTx::new_pq(tx0, k),
        None => SignedTx::new(tx0, &sk),
    };
    match ask(node, &Msg::NewTx(tx.clone()))? {
        Msg::Ack { ok: true, .. } => Ok((tx.id().iter().map(|b| format!("{b:02x}")).collect(), from_addr)),
        Msg::Ack { ok: false, info } => Err(format!("transaction refused: {info}")),
        m => Err(format!("unexpected reply: {m:?}").chars().take(120).collect()),
    }
}

// ------------------------------------------------------------------------------------------------ signatures and registration

pub fn hex_of(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn sign_text(mnemonic: &Mnemonic, message: &str) -> (String, String) {
    let sk = derive_account(mnemonic, 0);
    (address_of(&pubkey(&sk)), hex_of(&sign_message(&sk, message.as_bytes())))
}

pub fn verify_text(address: &str, message: &str, signature_hex: &str) -> Result<bool, String> {
    let pk = parse_address(address)?;
    let s = signature_hex.trim();
    if s.len() % 2 != 0 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("the signature must be hexadecimal text".into());
    }
    let sig: Vec<u8> = (0..s.len() / 2).filter_map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()).collect();
    Ok(verify_message(&pk, message.as_bytes(), &sig))
}

/// The three lines of the registration of the testnet rewards (docs/19, section 5), signed with the account key.
pub fn sign_registration(mnemonic: &Mnemonic, contact: &str) -> Result<vearl_core::regmail::Registration, String> {
    vearl_core::regmail::check_contact(contact)?;
    let sk = derive_account(mnemonic, 0);
    let address = address_of(&pubkey(&sk));
    let message = vearl_core::regmail::registration_message(&address, contact);
    let signature_hex = hex_of(&sign_message(&sk, message.as_bytes()));
    Ok(vearl_core::regmail::Registration { address, message, signature_hex })
}

// ------------------------------------------------------------------------------------------------ activity (block explorer)

#[derive(Clone, Debug, PartialEq)]
pub struct TxRow {
    pub id: String,
    /// "in" or "out"
    pub direction: String,
    /// What it is: "Transfer" or the kind of the transaction (job, bid...).
    pub label: String,
    pub counterparty: Option<String>,
    pub amount: Option<u64>,
    pub fee: u64,
    pub pending: bool,
    pub height: Option<u64>,
    pub confirmations: Option<u64>,
}

fn parse_amount_str(v: &serde_json::Value) -> Option<u64> {
    parse_amount(v.as_str()?).ok()
}

fn label_of(kind: &str) -> String {
    let words = clip(&kind.replace('_', " "), 40);
    let mut c = words.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => "Transaction".into(),
    }
}

/// `/api/address/{addr}/txs` of the explorer -> rows (newest first, pending ones first).
pub fn parse_txs(json: &str) -> Result<Vec<TxRow>, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("unreadable reply: {e}"))?;
    let arr = v.get("txs").and_then(|t| t.as_array()).ok_or("unexpected reply from the explorer")?;
    let mut rows = vec![];
    for t in arr {
        let kind = t.get("kind").cloned().unwrap_or_default();
        let ty = kind.get("type").and_then(|x| x.as_str()).unwrap_or("transaction");
        let direction = t.get("direction").and_then(|x| x.as_str()).unwrap_or("out").to_string();
        let from = t.get("from").and_then(|x| x.as_str()).map(str::to_string);
        let to = kind.get("to").and_then(|x| x.as_str()).map(str::to_string);
        let (from, to) = (from.map(|a| clip(&a, 90)), to.map(|a| clip(&a, 90)));
        let (counterparty, amount) = if ty == "transfer" {
            (if direction == "in" { from } else { to }, kind.get("amount").and_then(parse_amount_str))
        } else {
            (None, kind.get("escrow").or_else(|| kind.get("price")).and_then(parse_amount_str))
        };
        rows.push(TxRow {
            id: clip(t.get("id").and_then(|x| x.as_str()).unwrap_or(""), 64),
            direction,
            label: if ty == "transfer" { "Transfer".into() } else { label_of(ty) },
            counterparty,
            amount,
            fee: t.get("fee").and_then(parse_amount_str).unwrap_or(0),
            pending: t.get("status").and_then(|x| x.as_str()) == Some("pending"),
            height: t.get("height").and_then(|x| x.as_u64()),
            confirmations: t.get("confirmations").and_then(|x| x.as_u64()),
        });
    }
    Ok(rows)
}

pub fn fetch_activity(explorer: &str, address: &str) -> Result<Vec<TxRow>, String> {
    let url = format!("{}/api/address/{}/txs?limit=50", explorer.trim_end_matches('/'), address);
    parse_txs(&crate::httpget::get(&url)?)
}

/// Is this address in the public list of registered addresses (docs/19)?
pub fn is_registered(address: &str) -> Result<bool, String> {
    let body = crate::httpget::get(vearl_core::regmail::PUBLIC_LIST_URL)?;
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    Ok(v.get("addresses").and_then(|a| a.as_array()).is_some_and(|a| a.iter().any(|x| x.as_str() == Some(address))))
}

/// What the public leaderboard says about one address (the points are computed from the blocks of the chain only, docs/19).
#[derive(Clone, Debug, PartialEq)]
pub struct Standing {
    pub rank: u64,
    pub blocks: u64,
    pub useful_blocks: u64,
    pub points: f64,
    pub active_days: u64,
    pub eligible: bool,
    pub reasons: Vec<String>,
}

/// `leaderboard.json` -> the line of this address, or `None` when it has not mined a block (yet).
pub fn parse_standing(json: &str, address: &str) -> Result<Option<Standing>, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("unreadable reply: {e}"))?;
    let miners = v.get("miners").and_then(|m| m.as_array()).ok_or("unexpected leaderboard format")?;
    Ok(miners.iter().find(|m| m.get("address").and_then(|a| a.as_str()) == Some(address)).map(|m| Standing {
        rank: m.get("rank").and_then(|x| x.as_u64()).unwrap_or(0),
        blocks: m.get("blocks").and_then(|x| x.as_u64()).unwrap_or(0),
        useful_blocks: m.get("useful_blocks").and_then(|x| x.as_u64()).unwrap_or(0),
        points: m.get("points").and_then(|x| x.as_f64()).unwrap_or(0.0),
        active_days: m.get("active_days").and_then(|x| x.as_u64()).unwrap_or(0),
        eligible: m.get("eligible").and_then(|x| x.as_bool()).unwrap_or(false),
        reasons: m.get("reasons").and_then(|x| x.as_array()).map(|r| r.iter().filter_map(|s| s.as_str().map(str::to_string)).collect()).unwrap_or_default(),
    }))
}

pub fn fetch_standing(address: &str) -> Result<Option<Standing>, String> {
    let url = vearl_core::regmail::PUBLIC_LIST_URL.replace("registered.json", "leaderboard.json");
    parse_standing(&crate::httpget::get(&url)?, address)
}

pub fn tx_url(explorer: &str, id: &str) -> String {
    format!("{}/explorer/#/tx/{}", explorer.trim_end_matches('/'), id)
}

/// Opens a link in the default browser (only http(s) links: the text may come from settings).
pub fn open_link(url: &str) {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return;
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = std::process::Command::new("rundll32").args(["url.dll,FileProtocolHandler", url]).creation_flags(0x0800_0000).spawn();
    }
    #[cfg(not(windows))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}

/// Shows the folder of a file in the file manager.
pub fn reveal(path: &Path) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("explorer").arg(format!("/select,{}", path.display())).spawn();
    }
    #[cfg(not(windows))]
    {
        if let Some(d) = path.parent() {
            let _ = std::process::Command::new("xdg-open").arg(d).spawn();
        }
    }
}

#[cfg(test)]
pub static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_file_is_read() {
        let n = parse_network(r#"{"name":"t","peers":["seed.vearl.org:9000@abcd","1.2.3.4:9000@ef"],"listen":"0.0.0.0:9000"}"#).unwrap();
        assert_eq!(n.nodes, vec!["seed.vearl.org:9000", "1.2.3.4:9000"]);
        assert!(parse_network(r#"{"peers":[]}"#).is_none());
        assert!(parse_network("nonsense").is_none());
    }

    #[test]
    fn node_order_puts_the_chosen_node_first_without_duplicates() {
        let net = Network { name: "x".into(), nodes: vec!["a:1".into(), "b:1".into()], explorer: None };
        let mut s = Settings::default();
        assert_eq!(node_list(&s, &net), vec!["a:1", "b:1"]);
        s.node = "b:1".into();
        assert_eq!(node_list(&s, &net), vec!["b:1", "a:1"]);
    }

    #[test]
    fn amounts_are_grouped() {
        assert_eq!(fmt_grouped(0), ("0".into(), "00000000".into()));
        assert_eq!(fmt_grouped(123_456_789 * COIN + 5_000_000), ("123,456,789".into(), "05000000".into()));
        assert_eq!(fmt_grouped(999 * COIN).0, "999");
        assert_eq!(fmt_grouped(1000 * COIN).0, "1,000");
    }

    #[test]
    fn wallet_names_cannot_escape_the_folder() {
        assert!(wallet_path_for("../evil").is_err());
        assert!(wallet_path_for("a/b").is_err());
        assert!(wallet_path_for("a\\b").is_err());
        assert!(wallet_path_for("").is_err());
        assert!(wallet_path_for(".hidden").is_err());
        let p = wallet_path_for("My wallet").unwrap();
        assert!(p.ends_with("My-wallet.json"));
        assert!(wallet_path_for("main.json").unwrap().ends_with("main.json"));
    }

    #[test]
    fn phrase_is_cleaned_and_checked() {
        assert_eq!(clean_phrase("  Abandon  ABOUT\nabout "), "abandon about about");
        assert!(unknown_words("abandon about").is_empty());
        assert_eq!(unknown_words("abandon zzzz"), vec!["zzzz"]);
    }

    #[test]
    fn password_strength_levels() {
        assert_eq!(password_strength("short").0, 0);
        assert_eq!(password_strength("alllowercase").0, 1);
        assert!(password_strength("Correct-Horse-Battery-9").0 >= 2);
        assert_eq!(password_strength("Correct-Horse-Battery-Staple-9!").0, 3);
    }

    #[test]
    fn wallet_roundtrip_matches_the_command_line_wallet() {
        let dir = std::env::temp_dir().join(format!("vearl-gui-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.json");
        let m = new_mnemonic();
        // a light key derivation would make the test faster but create_wallet uses the production one: that is what must be tested
        let e = create_wallet(&path, &m, "a-long-enough-password").unwrap();
        assert_eq!(e.address, address_of(&pubkey(&derive_account(&m, 0))));
        assert!(create_wallet(&path, &m, "another-password-1").is_err(), "never overwrites");
        assert!(unlock(&path, "wrong-password-1").is_err());
        assert_eq!(unlock(&path, "a-long-enough-password").unwrap().to_entropy(), m.to_entropy());
        change_password(&path, "a-long-enough-password", "a-brand-new-password").unwrap();
        assert!(unlock(&path, "a-long-enough-password").is_err());
        assert_eq!(unlock(&path, "a-brand-new-password").unwrap().to_entropy(), m.to_entropy());
        assert!(change_password(&path, "wrong-password-1", "whatever-long-1").is_err());
        assert_eq!(unlock(&path, "a-brand-new-password").unwrap().to_entropy(), m.to_entropy(), "a failed change leaves the wallet untouched");
        assert!(!path.with_extension("json.new").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn signatures_verify_and_reject_tampering() {
        let m = new_mnemonic();
        let (addr, sig) = sign_text(&m, "hello");
        assert_eq!(verify_text(&addr, "hello", &sig), Ok(true));
        assert_eq!(verify_text(&addr, "hellO", &sig), Ok(false));
        assert!(verify_text(&addr, "hello", "xyz").is_err());
        assert!(verify_text("vrl1notanaddress", "hello", &sig).is_err());
    }

    #[test]
    fn registration_is_signed_like_the_command_line_one() {
        let m = new_mnemonic();
        let r = sign_registration(&m, "me@example.org").unwrap();
        assert_eq!(r.message, vearl_core::regmail::registration_message(&r.address, "me@example.org"));
        assert_eq!(verify_text(&r.address, &r.message, &r.signature_hex), Ok(true));
        assert!(sign_registration(&m, "").is_err());
    }

    #[test]
    fn text_from_a_server_cannot_break_the_window() {
        // multi-byte characters around the cut points, and a huge label
        let weird = "é".repeat(40);
        assert_eq!(short_addr(&weird).chars().count(), 12 + 3 + 8);
        assert_eq!(short_addr("vrl1short"), "vrl1short");
        let json = format!(r#"{{"txs":[{{"id":"{}","from":"{}","fee":"0.0","kind":{{"type":"{}","to":"{}","amount":"1.0"}},"direction":"in","status":"pending"}}]}}"#, "x".repeat(500), weird, "a_b".repeat(100), weird);
        let rows = parse_txs(&json).unwrap();
        assert!(rows[0].id.chars().count() <= 64 && rows[0].label.chars().count() <= 40);
        assert!(rows[0].counterparty.as_deref().unwrap_or("").chars().count() <= 90);
    }

    #[test]
    fn settings_survive_a_save() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("vearl-gui-settings-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("VEARL_WALLET_HOME", &dir);
        let mut s = Settings::default();
        assert_eq!((s.autolock_minutes, s.protect_capture, s.reduce_motion), (10, true, false));
        s.autolock_minutes = 30;
        s.protect_capture = false;
        s.reduce_motion = true;
        s.node = "1.2.3.4:9000".into();
        s.save();
        let back = Settings::load();
        assert_eq!(back, s);
        std::env::remove_var("VEARL_WALLET_HOME");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn leaderboard_line_is_found() {
        let json = r#"{"miners":[{"address":"vrl1a","rank":1,"blocks":10,"useful_blocks":2,"points":12.5,"active_days":3,"eligible":true,"reasons":[]},
                                 {"address":"vrl1b","rank":2,"blocks":1,"useful_blocks":0,"points":1.25,"active_days":1,"eligible":false,"reasons":["not registered"]}]}"#;
        let b = parse_standing(json, "vrl1b").unwrap().unwrap();
        assert_eq!((b.rank, b.blocks, b.active_days, b.eligible), (2, 1, 1, false));
        assert_eq!(b.reasons, vec!["not registered"]);
        assert_eq!(parse_standing(json, "vrl1a").unwrap().unwrap().points, 12.5);
        assert!(parse_standing(json, "vrl1zzz").unwrap().is_none());
        assert!(parse_standing("{}", "vrl1a").is_err());
    }

    #[test]
    fn explorer_rows_are_read() {
        let json = r#"{"address":"vrl1x","txs":[
          {"id":"aa","from":"vrl1other","nonce":3,"fee":"0.01000000","kind":{"type":"transfer","to":"vrl1x","amount":"12.50000000"},"status":"pending","direction":"in"},
          {"id":"bb","from":"vrl1x","nonce":2,"fee":"0.00000000","kind":{"type":"transfer","to":"vrl1dest","amount":"1.00000000"},"status":"confirmed","height":100,"confirmations":7,"direction":"out"},
          {"id":"cc","from":"vrl1x","nonce":1,"fee":"0.00000000","kind":{"type":"submit_job","escrow":"5.00000000","deadline":9},"status":"confirmed","height":90,"confirmations":17,"direction":"out"}]}"#;
        let rows = parse_txs(json).unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows[0].pending && rows[0].direction == "in" && rows[0].counterparty.as_deref() == Some("vrl1other") && rows[0].amount == Some(1_250_000_000));
        assert!(!rows[1].pending && rows[1].counterparty.as_deref() == Some("vrl1dest") && rows[1].confirmations == Some(7));
        assert_eq!(rows[2].label, "Submit job");
        assert_eq!(rows[2].amount, Some(500_000_000));
        assert!(parse_txs("{}").is_err());
        assert!(parse_txs("<html>").is_err());
    }
}
