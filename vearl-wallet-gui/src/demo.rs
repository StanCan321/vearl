//! Design tool (feature `devshots`, NOT in release builds): fills every screen with demo data and saves a PNG of each one.
//!   set VEARL_GUI_SHOTS=C:\some\folder   then start the program: it walks through the screens, writes the pictures and exits.
use crate::app::*;
use crate::core::*;
use crate::theme::Tone;
use eframe::egui;
use std::path::PathBuf;
use std::time::Instant;
use vearl_core::crypto::pubkey;
use vearl_core::state::COIN;
use vearl_core::wallet::{address_of, derive_account, derive_pq_account, new_mnemonic, pq_address_of};

pub struct Shots {
    dir: PathBuf,
    step: usize,
    frames: u32,
    asked: bool,
}

const NAMES: [&str; 20] = [
    "01-start-empty", "02-start-wallets", "03-create-name", "04-create-hidden", "05-create-words", "06-create-check", "07-create-done", "08-restore", "09-overview", "10-send",
    "11-receive", "12-activity", "13-rewards", "14-tools", "15-settings", "16-modal-password", "17-modal-send", "18-modal-phrase", "19-modal-sent", "20-overview-offline",
];

pub fn init(app: &mut App) {
    if let Some(dir) = std::env::var_os("VEARL_GUI_SHOTS") {
        let dir = PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        app.demo = true;
        if let Some((w, h)) = std::env::var("VEARL_GUI_SIZE").ok().and_then(|v| v.split_once('x').and_then(|(w, h)| Some((w.parse::<f32>().ok()?, h.parse::<f32>().ok()?)))) {
            app.ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(w, h)));
        }
        app.shots = Some(Shots { dir, step: 0, frames: 0, asked: false });
    }
}

fn demo_entry(name: &str, seed: u8) -> WalletEntry {
    let m = bip39::Mnemonic::from_entropy_in(bip39::Language::English, &[seed; 32]).unwrap();
    WalletEntry { path: PathBuf::from(format!("C:\\Users\\you\\AppData\\Roaming\\Vearl\\wallets\\{name}.json")), name: name.into(), address: address_of(&pubkey(&derive_account(&m, 0))) }
}

fn demo_rows(me: &str) -> Vec<TxRow> {
    let other = demo_entry("x", 9).address;
    let other2 = demo_entry("y", 3).address;
    let _ = me;
    vec![
        TxRow { id: "a1".repeat(32), direction: "in".into(), label: "Transfer".into(), counterparty: Some(other.clone()), amount: Some(250 * COIN), fee: 0, pending: true, height: None, confirmations: None },
        TxRow { id: "b2".repeat(32), direction: "out".into(), label: "Transfer".into(), counterparty: Some(other2), amount: Some(12 * COIN + 50_000_000), fee: 1_000_000, pending: false, height: Some(20812), confirmations: Some(5) },
        TxRow { id: "c3".repeat(32), direction: "in".into(), label: "Transfer".into(), counterparty: Some(other), amount: Some(1000 * COIN), fee: 0, pending: false, height: Some(20790), confirmations: Some(27) },
        TxRow { id: "d4".repeat(32), direction: "out".into(), label: "Submit job".into(), counterparty: None, amount: Some(5 * COIN), fee: 0, pending: false, height: Some(20400), confirmations: Some(417) },
    ]
}

fn demo_session(app: &mut App, page: Page) {
    let entry = demo_entry("main", 7);
    app.open_session(entry.clone());
    app.cmds.clear();
    let m = bip39::Mnemonic::from_entropy_in(bip39::Language::English, &[7u8; 32]).unwrap();
    let pq = pq_address_of(&derive_pq_account(&m, 0).pubkey());
    let Some(s) = app.session_mut() else { return };
    s.page = page;
    s.status = Some(Ok(NodeStatus { node: "seed.vearl.org:9000".into(), height: 20817, peers: 6, mempool: 1 }));
    s.account = Some(Account { balance: 1_234 * COIN + 56_789_000, nonce: 3, next_nonce: 4 });
    s.pq_address = Some(pq);
    s.pq_account = Some(Account { balance: 42 * COIN + 500_000, nonce: 0, next_nonce: 0 });
    s.activity = Some(Ok(demo_rows(&entry.address)));
    s.registered = Some(Ok(false));
    s.standing = Some(Ok(Some(Standing { rank: 12, blocks: 41, useful_blocks: 6, points: 58.75, active_days: 2, eligible: false, reasons: vec!["active on 2 of the 3 required days".into(), "not registered".into()] })));
    s.last_refresh = Some(Instant::now());
}

fn setup(app: &mut App, step: usize) {
    let other = demo_entry("x", 9).address;
    match step {
        0 => {
            app.wallets.clear();
            app.screen = Screen::Start(StartState::default());
        }
        1 => {
            app.wallets = vec![demo_entry("main", 7), demo_entry("mining-rig", 5)];
            app.screen = Screen::Start(StartState::default());
        }
        2 | 3 | 4 | 5 | 6 => {
            let mut c = Create::new();
            c.name = "my-wallet".into();
            c.password = "Correct-Horse-Battery-9".into();
            c.confirm = c.password.clone();
            let m = bip39::Mnemonic::from_entropy_in(bip39::Language::English, &[11u8; 32]).unwrap();
            let e = m.to_entropy();
            c.check_idx = [(e[0] % 8) as usize, 8 + (e[1] % 8) as usize, 16 + (e[2] % 8) as usize];
            c.phrase = Some(m);
            c.step = match step {
                2 => 0,
                3 | 4 => 1,
                5 => 2,
                _ => 3,
            };
            c.revealed = step >= 4;
            c.written = step >= 4;
            if step == 5 {
                c.check_in[0] = "abandon".into();
            }
            if step == 6 {
                c.done = Some(demo_entry("my-wallet", 11));
            }
            app.screen = Screen::Create(Box::new(c));
        }
        7 => {
            let m = new_mnemonic();
            app.screen = Screen::Restore(Box::new(Restore {
                name: "restored-wallet".into(),
                phrase: m.words().take(17).collect::<Vec<_>>().join(" "),
                password: "a-long-enough-password".into(),
                confirm: String::new(),
                error: None,
                busy: false,
                done: None,
            }));
        }
        8 => demo_session(app, Page::Overview),
        9 => {
            demo_session(app, Page::Send);
            if let Some(s) = app.session_mut() {
                s.send.to = other;
                s.send.amount = "12.5".into();
                s.send.fee = "0.01".into();
            }
        }
        10 => demo_session(app, Page::Receive),
        11 => demo_session(app, Page::Activity),
        12 => {
            demo_session(app, Page::Rewards);
            if let Some(s) = app.session_mut() {
                s.rewards.contact = "jane@example.org".into();
                s.rewards.lines = Some(RegLines {
                    address: s.entry.address.clone(),
                    message: format!("Vearl testnet rewards registration. Address: {}. Contact: jane@example.org. I accept the terms in docs/19.", s.entry.address),
                    signature: "9f".repeat(64),
                });
            }
        }
        13 => {
            demo_session(app, Page::Tools);
            if let Some(s) = app.session_mut() {
                s.tools.message = "I own this address".into();
                s.tools.signed = Some(RegLines { address: s.entry.address.clone(), message: "I own this address".into(), signature: "3c".repeat(64) });
                s.tools.verdict = Some(Ok(true));
            }
        }
        14 => {
            demo_session(app, Page::Settings);
            if let Some(s) = app.session_mut() {
                s.settings.node_test = Some(Ok(NodeStatus { node: "seed.vearl.org:9000".into(), height: 20817, peers: 6, mempool: 1 }));
            }
        }
        15 => {
            demo_session(app, Page::Overview);
            if let Some(s) = app.session_mut() {
                s.modal = Some(Modal::Password(PwPrompt { action: Action::RevealPq, password: "••••••••••".into(), error: Some("wrong password or modified file".into()), busy: false }));
            }
        }
        16 => {
            demo_session(app, Page::Send);
            if let Some(s) = app.session_mut() {
                s.modal = Some(Modal::Password(PwPrompt { action: Action::Send { to: other, amount: 12 * COIN + 50_000_000, fee: 1_000_000, pq: false }, password: String::new(), error: None, busy: false }));
            }
        }
        17 => {
            demo_session(app, Page::Settings);
            let m = bip39::Mnemonic::from_entropy_in(bip39::Language::English, &[7u8; 32]).unwrap();
            if let Some(s) = app.session_mut() {
                s.modal = Some(Modal::Phrase(m.words().map(str::to_string).collect()));
            }
        }
        18 => {
            demo_session(app, Page::Send);
            let from = demo_entry("main", 7).address;
            if let Some(s) = app.session_mut() {
                s.modal = Some(Modal::Sent { txid: "e5".repeat(32), from, to: other, amount: 12 * COIN + 50_000_000, fee: 1_000_000 });
            }
        }
        _ => {
            demo_session(app, Page::Overview);
            if let Some(s) = app.session_mut() {
                s.status = Some(Err("no node answers (seed.vearl.org:9000: connection refused)".into()));
                s.account = None;
                s.activity = Some(Err("vearl.org answered 502".into()));
                s.toast = Some((Tone::Warn, "Demo of the offline state.".into(), Instant::now()));
            }
        }
    }
}

/// Called every frame: sets the screen up, waits a few frames for the layout to settle, asks for a screenshot, saves it, goes on.
pub fn drive(app: &mut App, ctx: &egui::Context) {
    let Some(mut sh) = app.shots.take() else { return };
    if sh.step >= NAMES.len() {
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        app.shots = Some(sh);
        return;
    }
    // a screenshot that came back?
    let shot = ctx.input(|i| {
        i.events.iter().find_map(|e| match e {
            egui::Event::Screenshot { image, .. } => Some(image.clone()),
            _ => None,
        })
    });
    if let Some(img) = shot {
        let (w, h) = (img.size[0] as u32, img.size[1] as u32);
        let mut raw = Vec::with_capacity((w * h * 4) as usize);
        for p in &img.pixels {
            raw.extend_from_slice(&[p.r(), p.g(), p.b(), 255]);
        }
        let path = sh.dir.join(format!("{}.png", NAMES[sh.step]));
        if let Ok(f) = std::fs::File::create(&path) {
            let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            if let Ok(mut wr) = enc.write_header() {
                let _ = wr.write_image_data(&raw);
            }
        }
        sh.step += 1;
        sh.frames = 0;
        sh.asked = false;
        app.shots = Some(sh);
        ctx.request_repaint();
        return;
    }
    if sh.frames == 0 {
        setup(app, sh.step);
    }
    sh.frames += 1;
    if sh.frames > 12 && !sh.asked {
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        sh.asked = true;
    }
    ctx.request_repaint();
    app.shots = Some(sh);
}
