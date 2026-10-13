//! Design tool (feature `devshots`, NOT in release builds): fills every screen with demo data and saves a PNG of each one, some of them
//! at a chosen moment of their animation.
//!   set VEARL_GUI_SHOTS=C:\some\folder   then start the program: it walks through the screens, writes the pictures and exits.
//!   VEARL_GUI_SIZE=1440x900 sets the size of the window. The mean time of a frame is printed at the end.
use crate::app::*;
use crate::core::*;
use crate::demodata as dd;
use crate::theme::Tone;
use eframe::egui::{self, pos2};
use std::path::PathBuf;
use std::time::Instant;
use vearl_core::state::COIN;

pub struct Shots {
    dir: PathBuf,
    step: usize,
    frames: u32,
    t_setup: f64,
    asked: bool,
    frame_times: Vec<f32>,
}

struct Spec {
    name: &'static str,
    /// Seconds to wait after the setup before the picture is taken (the animations run in real time).
    wait: f64,
    setup: fn(&mut App, f64),
}

fn specs() -> Vec<Spec> {
    vec![
        Spec { name: "01-start-empty", wait: 2.00, setup: |a, _| { a.wallets.clear(); go_start(a) } },
        Spec { name: "02-start-wallets", wait: 2.00, setup: |a, _| { a.wallets = dd::entries(); go_start(a) } },
        Spec { name: "03-start-opening", wait: 0.23, setup: |a, _| { a.wallets = dd::entries(); go_start(a) } },
        Spec { name: "04-create-name", wait: 0.67, setup: |a, n| create(a, n, 0, |_, _| {}) },
        Spec { name: "05-create-entropy-start", wait: 0.40, setup: |a, n| create(a, n, 1, |_, _| {}) },
        Spec { name: "06-create-entropy-70", wait: 0.33, setup: |a, n| create(a, n, 1, |c, n| spiral(c, n, 112)) },
        Spec { name: "07-create-entropy-done", wait: 1.00, setup: |a, n| create(a, n, 1, |c, n| spiral(c, n, 170)) },
        Spec { name: "08-create-words-hidden", wait: 0.67, setup: |a, n| create(a, n, 2, |_, _| {}) },
        Spec { name: "09-create-words-decoding", wait: 0.23, setup: |a, n| create(a, n, 2, |c, n| c.reveal_t0 = Some(n)) },
        Spec { name: "10-create-words", wait: 1.50, setup: |a, n| create(a, n, 2, |c, n| { c.reveal_t0 = Some(n - 3.0); c.written = true }) },
        Spec { name: "11-create-verify", wait: 1.00, setup: |a, n| create(a, n, 3, |c, _| { let w = c.words(); c.check_in[0] = w[c.check_idx[0]].clone(); c.check_in[1] = "ab".into() }) },
        Spec { name: "12-create-securing", wait: 0.13, setup: |a, n| create(a, n, 4, |c, n| { c.secure_t0 = n - 1.9; c.t_step = n - 1.9 }) },
        Spec { name: "13-ready-burst", wait: 0.13, setup: |a, n| create(a, n, 5, |c, n| { c.done = Some(dd::entry("my-wallet", 11)); c.t_step = n - 1.0 }) },
        Spec { name: "14-ready", wait: 0.13, setup: |a, n| create(a, n, 5, |c, n| { c.done = Some(dd::entry("my-wallet", 11)); c.t_step = n - 3.4 }) },
        Spec { name: "15-restore", wait: 1.00, setup: |a, n| {
            a.screen = Screen::Restore(Box::new(Restore::new(n)));
            if let Screen::Restore(r) = &mut a.screen {
                r.phrase = dd::mnemonic(5).words().take(17).collect::<Vec<_>>().join(" ");
                r.password = "a-long-enough-password".into();
            }
        } },
        Spec { name: "16-overview", wait: 1.50, setup: |a, _| session(a, Page::Overview, |_| {}) },
        Spec { name: "17-overview-opening", wait: 0.08, setup: |a, _| session(a, Page::Overview, |_| {}) },
        Spec { name: "18-send", wait: 1.00, setup: |a, _| session(a, Page::Send, |s| { s.send.to = dd::entry("x", 9).address; s.send.amount = "12.5".into(); s.send.fee = "0.01".into(); }) },
        Spec { name: "19-receive", wait: 1.00, setup: |a, _| session(a, Page::Receive, |_| {}) },
        Spec { name: "20-activity", wait: 1.00, setup: |a, _| session(a, Page::Activity, |_| {}) },
        Spec { name: "21-rewards", wait: 1.00, setup: |a, _| session(a, Page::Rewards, |s| {
            s.rewards.contact = "jane@example.org".into();
            s.rewards.lines = Some(RegLines { address: s.entry.address.clone(), message: format!("Vearl testnet rewards registration. Address: {}. Contact: jane@example.org. I accept the terms in docs/19.", s.entry.address), signature: "9f".repeat(64) });
        }) },
        Spec { name: "22-tools", wait: 1.00, setup: |a, _| session(a, Page::Tools, |s| {
            s.tools.message = "I own this address".into();
            s.tools.signed = Some(RegLines { address: s.entry.address.clone(), message: "I own this address".into(), signature: "3c".repeat(64) });
            s.tools.verdict = Some(Ok(true));
        }) },
        Spec { name: "23-settings", wait: 1.00, setup: |a, _| session(a, Page::Settings, |s| s.settings.node_test = Some(Ok(dd::status()))) },
        Spec { name: "24-modal-password", wait: 0.67, setup: |a, _| session(a, Page::Overview, |s| { let mut p = PwPrompt::new(Action::RevealPq); p.error = Some("wrong password or modified file".into()); p.failures = 1; s.modal = Some(Modal::Password(p)) }) },
        Spec { name: "25-modal-send", wait: 0.67, setup: |a, _| session(a, Page::Send, |s| s.modal = Some(Modal::Password(PwPrompt::new(Action::Send { to: dd::entry("x", 9).address, amount: 12 * COIN + 50_000_000, fee: 1_000_000, pq: false })))) },
        Spec { name: "26-modal-phrase", wait: 1.00, setup: |a, _| session(a, Page::Settings, |s| s.modal = Some(Modal::Phrase(dd::mnemonic(7).words().map(str::to_string).collect()))) },
        Spec { name: "27-modal-sent", wait: 0.67, setup: |a, _| session(a, Page::Send, |s| s.modal = Some(Modal::Sent { txid: "e5".repeat(32), from: dd::entry("main", 7).address, to: dd::entry("x", 9).address, amount: 12 * COIN + 50_000_000, fee: 1_000_000 })) },
        Spec { name: "28-offline", wait: 0.67, setup: |a, _| session(a, Page::Overview, |s| {
            s.status = Some(Err("no node answers (seed.vearl.org:9000: connection refused)".into()));
            s.account = None;
            s.activity = Some(Err("vearl.org answered 502".into()));
            s.toast = Some((Tone::Warn, "Demo of the offline state.".into(), Instant::now()));
        }) },
        Spec { name: "29-loading", wait: 0.50, setup: |a, _| session(a, Page::Overview, |s| { s.status = None; s.account = None; s.activity = None; s.pq_address = None; }) },
        Spec { name: "30-locked", wait: 0.67, setup: |a, n| { a.wallets = dd::entries(); a.screen = Screen::Start(StartState::new(n - 5.0, Some((Tone::Info, "Locked after 10 minutes without activity.".into())))); } },
    ]
}

pub fn init(app: &mut App) {
    if let Some(dir) = std::env::var_os("VEARL_GUI_SHOTS") {
        let dir = PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        app.demo = true;
        if let Some((w, h)) = std::env::var("VEARL_GUI_SIZE").ok().and_then(|v| v.split_once('x').and_then(|(w, h)| Some((w.parse::<f32>().ok()?, h.parse::<f32>().ok()?)))) {
            app.ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(w, h)));
        }
        app.shots = Some(Shots { dir, step: 0, frames: 0, t_setup: 0.0, asked: false, frame_times: vec![] });
    }
}

fn go_start(a: &mut App) {
    let n = a.now();
    a.screen = Screen::Start(StartState::new(n, None));
}

fn create(a: &mut App, now: f64, step: u8, f: impl FnOnce(&mut Create, f64)) {
    let mut c = Create::new(now);
    c.name = "my-wallet".into();
    c.password = "Correct-Horse-Battery-9".into();
    c.confirm = c.password.clone();
    let m = dd::creation_mnemonic();
    let e = m.to_entropy();
    c.check_idx = [(e[0] % 8) as usize, 8 + (e[1] % 8) as usize, 16 + (e[2] % 8) as usize];
    if step >= 2 {
        c.phrase = Some(m);
    }
    c.step = step;
    c.t_step = now;
    c.secure_t0 = now;
    f(&mut c, now);
    a.screen = Screen::Create(Box::new(c));
}

/// A spiral of pointer positions around the middle of the stage, `n` of them.
fn spiral(c: &mut Create, now: f64, n: u32) {
    for i in 0..n {
        let a = i as f32 * 0.37;
        let r = 20.0 + i as f32 * 1.1;
        c.entropy.add(pos2(705.0 + a.cos() * r, 340.0 + a.sin() * r), now - (n - i) as f64 * 0.012);
    }
}

fn session(a: &mut App, page: Page, f: impl FnOnce(&mut Session)) {
    let entry = dd::entry("main", 7);
    a.open_session(entry);
    a.cmds.clear();
    let n = a.now();
    let Some(s) = a.session_mut() else { return };
    s.page = page;
    s.shown_page = page;
    s.page_t0 = n;
    dd::fill_session(s);
    f(s);
    // the dialogs open with their own animation
    s.modal_t0 = n;
    s.modal_was_open = s.modal.is_some();
}

/// Called every frame: sets the screen up, waits for it to settle, asks for a screenshot, saves it, goes on.
pub fn drive(app: &mut App, ctx: &egui::Context) {
    let Some(mut sh) = app.shots.take() else { return };
    let only = std::env::var("VEARL_GUI_ONLY").ok();
    let list: Vec<Spec> = specs().into_iter().filter(|s| only.as_deref().map_or(true, |o| s.name.starts_with(o))).collect();
    sh.frame_times.push(ctx.input(|i| i.unstable_dt));
    if sh.step >= list.len() {
        let n = sh.frame_times.len().max(1) as f32;
        let mean = sh.frame_times.iter().sum::<f32>() / n;
        let worst = sh.frame_times.iter().cloned().fold(0.0, f32::max);
        println!("frames: {} mean interval {:.1} ms (worst {:.0} ms)", sh.frame_times.len(), mean * 1000.0, worst * 1000.0);
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        app.shots = Some(sh);
        return;
    }
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
        let path = sh.dir.join(format!("{}.png", list[sh.step].name));
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
        let now = app.now();
        sh.t_setup = now;
        (list[sh.step].setup)(app, now);
    }
    sh.frames += 1;
    if sh.frames > 2 && app.now() - sh.t_setup >= list[sh.step].wait && !sh.asked {
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        sh.asked = true;
    }
    ctx.request_repaint();
    app.shots = Some(sh);
}
