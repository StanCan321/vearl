//! Testnet faucet: hands out a small amount (without value) to an address, with limits per address, per IP and per day.
//! The faucet key is that of a dedicated wallet, funded by the genesis; it must never be that of a
//! personal wallet. Behind a reverse proxy, the IP seen is the proxy's: apply a limit there too.
use crate::crypto::*;
use crate::httpmini::{Request, Response};
use crate::net::{request, Msg};
use crate::types::*;
use crate::wallet::{address_of, fmt_amount, parse_address};
use ed25519_dalek::SigningKey;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct FaucetConfig {
    pub node: String,
    pub amount: u64,
    pub cooldown: Duration,
    pub max_per_day: u64,
}

pub struct Faucet {
    cfg: FaucetConfig,
    sk: SigningKey,
    state: Mutex<FaucetState>,
}

#[derive(Default)]
struct FaucetState {
    by_addr: HashMap<PubKey, Instant>,
    by_ip: HashMap<String, Instant>,
    day: u64,
    sent_today: u64,
}

const PAGE: &str = "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>\
<title>VEARL faucet</title><style>body{font:15px system-ui;max-width:560px;margin:40px auto;padding:0 16px}\
input{width:100%;padding:10px;margin:8px 0;box-sizing:border-box}button{padding:10px 16px}</style>\
<h1>VEARL faucet (testnet)</h1><p>Test tokens with no value. One request per address and per hour.</p>\
<input id=a placeholder='vrl1...' autocomplete=off><button id=b>Receive</button><pre id=o></pre>\
<script>document.getElementById('b').onclick=async()=>{const r=await fetch('/drip',{method:'POST',body:JSON.stringify({address:document.getElementById('a').value.trim()})});\
document.getElementById('o').textContent=JSON.stringify(await r.json(),null,2)}</script>";

impl Faucet {
    pub fn new(cfg: FaucetConfig, sk: SigningKey) -> Faucet {
        Faucet { cfg, sk, state: Mutex::new(FaucetState::default()) }
    }

    pub fn address(&self) -> String {
        address_of(&pubkey(&self.sk))
    }

    pub fn handle(&self, req: Request) -> Response {
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/") => Response::html(PAGE),
            ("GET", "/health") => Response::json(200, &json!({ "ok": true, "faucet": self.address() })),
            ("POST", "/drip") => self.drip(&req),
            ("GET", _) | ("POST", _) => Response::error(404, "unknown route"),
            _ => Response::error(405, "method not allowed"),
        }
    }

    fn drip(&self, req: &Request) -> Response {
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) else { return Response::error(400, "JSON body expected: {\"address\":\"vrl1...\"}") };
        let Some(addr) = v.get("address").and_then(|a| a.as_str()) else { return Response::error(400, "field 'address' required") };
        let to = match parse_address(addr) {
            Ok(p) => p,
            Err(e) => return Response::error(400, &e),
        };
        if to == pubkey(&self.sk) {
            return Response::error(400, "the faucet does not send anything to itself");
        }
        let now = Instant::now();
        let day = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() / 86_400;
        // Critical section: checks the limits AND signs/sends under the same lock (sequential nonces, no race).
        let mut st = self.state.lock().unwrap();
        if st.day != day {
            st.day = day;
            st.sent_today = 0;
        }
        if st.sent_today + self.cfg.amount > self.cfg.max_per_day {
            return Response::error(429, "daily faucet cap reached");
        }
        for (label, last) in [("this address", st.by_addr.get(&to).copied()), ("this IP", st.by_ip.get(&req.peer).copied())] {
            if let Some(t) = last {
                let wait = self.cfg.cooldown.saturating_sub(now.duration_since(t));
                if !wait.is_zero() {
                    return Response::json(429, &json!({ "error": format!("{label} has already received tokens"), "retry_in_secs": wait.as_secs() + 1 }));
                }
            }
        }
        let from = pubkey(&self.sk);
        let (balance, next_nonce) = match request(&self.cfg.node, &Msg::GetAccount(from)) {
            Ok(Msg::Account { balance, next_nonce, .. }) => (balance, next_nonce),
            _ => return Response::error(503, "node unreachable"),
        };
        if balance < self.cfg.amount {
            return Response::error(503, "faucet empty");
        }
        let tx = SignedTx::new(Tx { from, nonce: next_nonce, fee: 0, kind: TxKind::Transfer { to, amount: self.cfg.amount } }, &self.sk);
        let id = tx.id();
        match request(&self.cfg.node, &Msg::NewTx(tx)) {
            Ok(Msg::Ack { ok: true, .. }) => {}
            Ok(Msg::Ack { ok: false, info }) => return Response::error(503, &format!("transaction refused: {info}")),
            _ => return Response::error(503, "node unreachable"),
        }
        st.by_addr.insert(to, now);
        st.by_ip.insert(req.peer.clone(), now);
        st.sent_today += self.cfg.amount;
        Response::json(200, &json!({ "ok": true, "amount": fmt_amount(self.cfg.amount), "to": address_of(&to), "tx": id.iter().map(|b| format!("{b:02x}")).collect::<String>() }))
    }
}
