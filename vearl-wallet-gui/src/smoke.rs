//! Draws every screen headlessly, at many window sizes and moments of the animations, with and without motion, and checks that
//! nothing panics and that something is painted. It cannot judge the looks (the picture tool does), but it catches layout panics,
//! divisions by zero, slicing mistakes and unwrap()s that only show up at an odd size or at a given instant.
use crate::app::*;
use crate::core::*;
use crate::demodata as dd;
use crate::theme::Tone;
use eframe::egui::{self, pos2, vec2, Pos2, Rect};
use std::time::Instant;
use vearl_core::state::COIN;

fn frame(ctx: &egui::Context, app: &mut App, size: egui::Vec2, time: f64, pointer: Option<Pos2>) -> usize {
    let mut raw = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, size)), time: Some(time), ..Default::default() };
    if let Some(p) = pointer {
        raw.events.push(egui::Event::PointerMoved(p));
    }
    let out = ctx.run_ui(raw, |ui| crate::draw(app, ui));
    out.shapes.len()
}

fn session(app: &mut App, page: Page, modal: Option<Modal>) {
    app.open_session(dd::entry("main", 7));
    app.cmds.clear();
    let n = app.now();
    let s = app.session_mut().unwrap();
    s.page = page;
    s.shown_page = page;
    s.page_t0 = n;
    dd::fill_session(s);
    s.modal = modal;
    s.modal_t0 = n;
}

fn create(app: &mut App, step: u8) {
    let n = app.now();
    let mut c = Create::new(n);
    c.name = "wallet".into();
    c.password = "Correct-Horse-Battery-9".into();
    c.confirm = c.password.clone();
    let m = dd::mnemonic(11);
    c.phrase = Some(m);
    c.step = step;
    c.reveal_t0 = if step == 2 { Some(n) } else { None };
    c.done = Some(dd::entry("wallet", 11));
    c.result = Some(Ok(dd::entry("wallet", 11)));
    app.screen = Screen::Create(Box::new(c));
}

#[test]
fn every_screen_draws_at_every_size_and_time() {
    let _lock = crate::core::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = std::env::temp_dir().join(format!("vearl-gui-smoke-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::env::set_var("VEARL_WALLET_HOME", &tmp);

    let sizes = [vec2(900.0, 700.0), vec2(1120.0, 760.0), vec2(1920.0, 1080.0), vec2(3840.0, 2160.0), vec2(640.0, 480.0), vec2(300.0, 200.0), vec2(1.0, 1.0)];
    let times = [0.0, 0.05, 0.4, 1.3, 3.0, 12.0, 400.0];
    let ctx = egui::Context::default();
    crate::theme::install(&ctx);
    let logo = ctx.load_texture("t", egui::ColorImage::example(), Default::default());
    let mut app = App::new(ctx.clone(), logo, true);
    let mut painted = 0usize;

    for reduce in [false, true] {
        app.settings.reduce_motion = reduce;
        crate::anim::set_reduce_motion(reduce);
        for size in sizes {
            let mut scenes: Vec<(&str, Box<dyn Fn(&mut App)>)> = vec![
                ("start", Box::new(|a| {
                    a.wallets = dd::entries();
                    let n = a.now();
                    a.screen = Screen::Start(StartState::new(n, Some((Tone::Info, "Locked after 10 minutes without activity.".into()))));
                })),
                ("start-empty", Box::new(|a| {
                    a.wallets.clear();
                    let n = a.now();
                    let mut st = StartState::new(n, None);
                    st.show_open = true;
                    st.error = Some("not a wallet".into());
                    a.screen = Screen::Start(st);
                })),
                ("restore-0", Box::new(|a| {
                    let n = a.now();
                    let mut r = Restore::new(n);
                    r.phrase = "abandon abandon zzzz".into();
                    r.error = Some("A recovery phrase has 24 words; you typed 3.".into());
                    a.screen = Screen::Restore(Box::new(r));
                })),
                ("restore-1", Box::new(|a| {
                    let n = a.now();
                    let mut r = Restore::new(n);
                    r.step = 1;
                    r.secure_t0 = n;
                    a.screen = Screen::Restore(Box::new(r));
                })),
                ("restore-1-error", Box::new(|a| {
                    let n = a.now();
                    let mut r = Restore::new(n);
                    r.step = 1;
                    r.secure_t0 = n - 10.0;
                    r.result = Some(Err("disk full".into()));
                    a.screen = Screen::Restore(Box::new(r));
                })),
                ("restore-2", Box::new(|a| {
                    let n = a.now();
                    let mut r = Restore::new(n);
                    r.step = 2;
                    r.done = Some(dd::entry("r", 5));
                    a.screen = Screen::Restore(Box::new(r));
                })),
            ];
            for step in 0..6u8 {
                scenes.push(("create", Box::new(move |a| create(a, step))));
            }
            for (page, _, _) in Page::ALL {
                scenes.push(("page", Box::new(move |a| session(a, page, None))));
            }
            scenes.push(("modal-password", Box::new(|a| session(a, Page::Send, Some(Modal::Password(PwPrompt::new(Action::Send { to: dd::entry("x", 9).address, amount: 5 * COIN, fee: 0, pq: true })))))));
            scenes.push(("modal-locked", Box::new(|a| {
                let mut p = PwPrompt::new(Action::ShowPhrase);
                p.failures = 5;
                p.locked_until = a.now() + 30.0;
                p.error = Some("wrong password or modified file".into());
                session(a, Page::Settings, Some(Modal::Password(p)));
            })));
            scenes.push(("modal-busy", Box::new(|a| {
                let mut p = PwPrompt::new(Action::RevealPq);
                p.busy = true;
                session(a, Page::Overview, Some(Modal::Password(p)));
            })));
            scenes.push(("modal-phrase", Box::new(|a| session(a, Page::Settings, Some(Modal::Phrase(dd::mnemonic(7).words().map(str::to_string).collect()))))));
            scenes.push(("modal-sent", Box::new(|a| session(a, Page::Send, Some(Modal::Sent { txid: "e5".repeat(32), from: dd::entry("main", 7).address, to: dd::entry("x", 9).address, amount: COIN, fee: 0 })))));
            scenes.push(("offline", Box::new(|a| {
                session(a, Page::Overview, None);
                let s = a.session_mut().unwrap();
                s.status = Some(Err("no node answers".into()));
                s.account = None;
                s.activity = Some(Err("explorer down".into()));
                s.registered = Some(Err("down".into()));
                s.standing = Some(Err("down".into()));
            })));
            scenes.push(("loading", Box::new(|a| {
                session(a, Page::Activity, None);
                let s = a.session_mut().unwrap();
                s.status = None;
                s.account = None;
                s.activity = None;
                s.pq_address = None;
                s.registered = None;
                s.standing = None;
            })));
            scenes.push(("other-network", Box::new(|a| {
                session(a, Page::Overview, None);
                let s = a.session_mut().unwrap();
                s.status = Some(Ok(NodeStatus { genesis_ok: Some(false), ..dd::status() }));
            })));
            scenes.push(("empty-activity", Box::new(|a| {
                session(a, Page::Activity, None);
                a.session_mut().unwrap().activity = Some(Ok(vec![]));
            })));
            scenes.push(("huge-balance", Box::new(|a| {
                session(a, Page::Overview, None);
                a.session_mut().unwrap().account = Some(Account { balance: u64::MAX, nonce: 0, next_nonce: 0 });
            })));
            scenes.push(("long-text", Box::new(|a| {
                session(a, Page::Rewards, None);
                let s = a.session_mut().unwrap();
                s.rewards.contact = "x".repeat(400);
                s.rewards.error = Some("é".repeat(600));
                s.send.to = "ü".repeat(300);
                s.send.error = Some("😀".repeat(100));
                s.tools.message = "line\n".repeat(200);
                s.toast = Some((Tone::Warn, "w".repeat(500), Instant::now()));
            })));

            for (name, build) in &scenes {
                build(&mut app);
                for (i, t) in times.iter().enumerate() {
                    let ptr = if i % 2 == 0 { Some(pos2(size.x * 0.6, size.y * 0.4)) } else { None };
                    let n = frame(&ctx, &mut app, size, *t, ptr);
                    painted += n;
                    if size.x >= 640.0 {
                        assert!(n > 0, "{name} at {size:?}, t={t}: nothing was painted");
                    }
                    // the commands of the drawing code (clicks) must be runnable too
                    app.run_cmds();
                }
            }
        }
    }
    assert!(painted > 1000);
    crate::anim::set_reduce_motion(false);
    std::env::remove_var("VEARL_WALLET_HOME");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Regression: after a wallet page had been open, the start screen came back without its "Your wallets" card: the stagger of the wallet
/// pages (a start time and a counter that grows with every card drawn) was left behind, so a card drawn afterwards waited longer and
/// longer and never finished appearing. The screens outside the wallet must not inherit it.
#[test]
fn start_screen_after_a_lock_has_no_stale_stagger() {
    let _lock = crate::core::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = std::env::temp_dir().join(format!("vearl-gui-smoke-lock-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::env::set_var("VEARL_WALLET_HOME", &tmp);
    crate::anim::set_reduce_motion(false);
    let size = vec2(1120.0, 760.0);
    let ctx = egui::Context::default();
    crate::theme::install(&ctx);
    let logo = ctx.load_texture("t", egui::ColorImage::example(), Default::default());
    let mut app = App::new(ctx.clone(), logo, true);

    session(&mut app, Page::Overview, None);
    for k in 0..40 {
        frame(&ctx, &mut app, size, 20.0 + k as f64 * 0.015, None);
    }
    let (t0, _) = crate::anim::scene_next(&ctx);
    assert!(t0 >= 0.0, "a wallet page should have started a scene");
    app.cmds.push(Cmd::GoStart);
    app.run_cmds();
    for k in 0..40 {
        frame(&ctx, &mut app, size, 21.0 + k as f64 * 0.015, None);
    }
    let (t0, _) = crate::anim::scene_next(&ctx);
    assert!(t0 < 0.0, "the start screen inherited the stagger of the wallet page (start {t0})");
    std::env::remove_var("VEARL_WALLET_HOME");
    let _ = std::fs::remove_dir_all(&tmp);
}
