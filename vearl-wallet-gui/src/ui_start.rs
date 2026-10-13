//! The screens before a wallet is open: start, create (a six-step ceremony), restore.
use crate::anim;
use crate::app::*;
use crate::core::*;
use crate::fx;
use crate::theme::{self as t, Kind, Tone};
use eframe::egui::{self, pos2, vec2, Align2, Id, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Ui, UiBuilder};

/// A centred column of at most `max_w` pixels.
pub fn column<R>(ui: &mut Ui, max_w: f32, add: impl FnOnce(&mut Ui) -> R) -> R {
    let avail = ui.available_rect_before_wrap();
    let w = avail.width().min(max_w);
    let x = avail.left() + (avail.width() - w) / 2.0;
    let rect = Rect::from_min_max(pos2(x, avail.top()), pos2(x + w, avail.bottom()));
    ui.scope_builder(UiBuilder::new().max_rect(rect), |ui| {
        ui.set_width(w);
        add(ui)
    })
    .inner
}

pub fn brand(ui: &mut Ui, logo: &egui::TextureHandle, size: f32) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 11.0;
        ui.add(egui::Image::new(egui::load::SizedTexture::new(logo.id(), vec2(size, size))));
        ui.label(RichText::new("Vearl").font(t::sans_semibold(size * 0.78)).color(t::INK));
    });
}

/// The bottom strip of the screens without a sidebar.
pub fn footer(ui: &mut Ui) {
    let r = ui.max_rect();
    ui.painter().line_segment([r.left_top(), r.right_top()], Stroke::new(1.0, t::LINE));
    let p = ui.painter();
    p.text(pos2(r.left() + 28.0, r.center().y), Align2::LEFT_CENTER, "TESTNET WALLET  ·  TOKENS HAVE NO VALUE  ·  NOT AUDITED", t::mono(10.5), t::DIM);
    p.text(pos2(r.right() - 28.0, r.center().y), Align2::RIGHT_CENTER, format!("v{}", env!("CARGO_PKG_VERSION")), t::mono(10.5), t::DIM);
}

fn reveal<R>(ui: &mut Ui, t0: f64, delay: f32, add: impl FnOnce(&mut Ui) -> R) -> R {
    anim::reveal(ui, t0, delay, 16.0, add)
}

// ------------------------------------------------------------------------------------------------ start

pub fn start(ui: &mut Ui, a: &mut App) {
    let ctx = ui.ctx().clone();
    let (fx_now, pointer) = (anim::fx_time(&ctx), ctx.pointer_latest_pos());
    let demo = a.demo;
    let App { logo, wallets, cmds, screen, .. } = a;
    let Screen::Start(st) = screen else { return };
    // a wallet file dropped on the window is opened
    if let Some(path) = ctx.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf())) {
        cmds.push(Cmd::OpenPath(path.display().to_string()));
    }
    let t0 = st.t0;
    ui.add_space(26.0);
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        column(ui, 1080.0, |ui| {
            ui.add_space(6.0);
            reveal(ui, t0, 0.0, |ui| {
                ui.horizontal(|ui| {
                    brand(ui, logo, 36.0);
                    if demo {
                        ui.add_space(8.0);
                        t::chip(ui, "Demo data", t::ROSE);
                    }
                });
            });
            ui.add_space(34.0);
            let total = ui.available_width();
            let gap = 56.0;
            let wide = total > 820.0;
            let (lw, rw) = if wide { ((total - gap) * 0.52, (total - gap) * 0.48) } else { (total, total) };
            let top = ui.cursor().min;
            let left_rect = Rect::from_min_size(top, vec2(lw, 560.0));
            let mut used_l = 0.0;
            // the living mark of the site, behind the card
            let right_rect = if wide { Rect::from_min_size(pos2(top.x + lw + gap, top.y + 140.0), vec2(rw, 560.0)) } else { Rect::from_min_size(pos2(top.x, top.y + 640.0), vec2(rw, 560.0)) };
            if wide {
                let k = anim::ease_out(anim::progress(&ctx, t0, 0.1, 1.6));
                fx::sphere(ui.painter(), pos2(right_rect.center().x + 6.0, top.y + 72.0), 218.0 * (0.86 + 0.14 * k), 11.5, 0.95 * k, fx_now, pointer);
                if !anim::reduced() {
                    ctx.request_repaint();
                }
            }
            ui.scope_builder(UiBuilder::new().max_rect(left_rect), |ui| {
                ui.set_width(lw);
                if let Some((tone, msg)) = st.notice.clone() {
                    t::notice(ui, tone, &msg);
                    ui.add_space(14.0);
                }
                reveal(ui, t0, 0.10, |ui| t::eyebrow(ui, None, "Vearl wallet"));
                ui.add_space(14.0);
                reveal(ui, t0, 0.18, |ui| t::headline(ui, "Your keys,\n", "on your machine.", 52.0));
                ui.add_space(18.0);
                reveal(ui, t0, 0.30, |ui| {
                    ui.add(egui::Label::new(RichText::new("A wallet for the Vearl testnet. Your 24 recovery words never leave this computer, and every payment is signed here, on your side, before it is sent.").font(t::sans(16.5)).color(t::INK2)).wrap());
                });
                ui.add_space(26.0);
                reveal(ui, t0, 0.42, |ui| {
                    ui.horizontal(|ui| {
                        if t::primary(ui, "Create a wallet", true).clicked() {
                            cmds.push(Cmd::StartCreate);
                        }
                        if t::ghost(ui, "Restore from 24 words").clicked() {
                            cmds.push(Cmd::StartRestore);
                        }
                    });
                });
                ui.add_space(26.0);
                for (i, (n, s)) in [("Post-quantum", "Accounts can also sign with SLH-DSA (NIST FIPS 205), which quantum computers cannot forge."), ("Verified", "The wallet is open source and uses the same code as the command-line wallet."), ("Yours alone", "Nothing is sent anywhere except the payments and the registration you confirm.")].into_iter().enumerate() {
                    reveal(ui, t0, 0.55 + i as f32 * 0.1, |ui| {
                        ui.horizontal_top(|ui| {
                            t::dot(ui, t::MINT);
                            ui.vertical(|ui| {
                                ui.label(RichText::new(n).font(t::sans_semibold(14.5)).color(t::INK));
                                ui.add(egui::Label::new(RichText::new(s).font(t::sans(13.5)).color(t::MUTE)).wrap());
                            });
                        });
                    });
                    ui.add_space(4.0);
                }
                used_l = ui.min_rect().height();
            });
            let mut used_r = 0.0;
            ui.scope_builder(UiBuilder::new().max_rect(right_rect), |ui| {
                ui.set_width(rw);
                reveal(ui, t0, 0.5, |ui| {
                    t::hero_card(ui, |ui| {
                        t::eyebrow(ui, Some("01"), "Your wallets");
                        ui.add_space(14.0);
                        if wallets.is_empty() {
                            ui.add_space(6.0);
                            t::body(ui, "No wallet on this computer yet.");
                            ui.add_space(2.0);
                            t::muted(ui, "Create one in a minute, or restore one with its 24 words. If you already have a wallet file (from vearl-wallet or the miner), open it below.");
                        }
                        let mut forget: Option<String> = None;
                        for w in wallets.iter() {
                            let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 68.0), Sense::click());
                            let h = anim::fade(&ctx, resp.id.with("h"), resp.hovered(), 0.18);
                            ui.painter().rect(rect, 0.0, t::alpha(t::MINT, (12.0 * h) as u8), Stroke::new(1.0, anim::mix(t::LINE2, t::MINT, h)), StrokeKind::Inside);
                            ui.painter().text(rect.left_top() + vec2(18.0 + 3.0 * h, 16.0), Align2::LEFT_TOP, &w.name, t::sans_semibold(16.5), t::INK);
                            ui.painter().text(rect.left_top() + vec2(18.0 + 3.0 * h, 41.0), Align2::LEFT_TOP, short_addr(&w.address), t::mono(12.5), t::MUTE);
                            t::draw_arrow(ui.painter(), pos2(rect.right() - 24.0 + 5.0 * h, rect.center().y), anim::mix(t::MUTE, t::MINT, h));
                            if resp.hovered() {
                                ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
                            }
                            if resp.clicked() {
                                cmds.push(Cmd::OpenWallet(w.clone()));
                            }
                            resp.context_menu(|ui| {
                                if ui.button("Remove from this list (the file stays)").clicked() {
                                    forget = Some(w.path.display().to_string());
                                    ui.close();
                                }
                            });
                            ui.add_space(2.0);
                        }
                        if let Some(f) = forget {
                            cmds.push(Cmd::Forget(f));
                        }
                        ui.add_space(10.0);
                        if st.show_open {
                            t::field(ui, "Path of a wallet file", &mut st.open_path, "C:\\Users\\you\\wallet.json", false, true);
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                if t::button_ex(ui, "Open this file", Kind::Ghost, !st.open_path.trim().is_empty(), false, 0.0).clicked() {
                                    st.error = None;
                                    cmds.push(Cmd::OpenPath(st.open_path.clone()));
                                }
                                if t::ghost(ui, "Cancel").clicked() {
                                    st.show_open = false;
                                    st.error = None;
                                }
                            });
                        } else {
                            if t::link(ui, "Open a wallet file...").clicked() {
                                st.show_open = true;
                            }
                            ui.add_space(2.0);
                            ui.label(RichText::new("You can also drop a wallet file on this window.").font(t::sans(12.5)).color(t::DIM));
                        }
                        if let Some(e) = &st.error {
                            ui.add_space(6.0);
                            t::notice(ui, Tone::Bad, e);
                        }
                    });
                });
                used_r = ui.min_rect().height();
            });
            let used = if wide { used_l.max(used_r + 140.0) } else { used_l + 30.0 + used_r };
            ui.allocate_space(vec2(total, used.max(100.0) + 60.0));
        });
    });
}

// ------------------------------------------------------------------------------------------------ the left rail of the ceremony

const CREATE_STEPS: [&str; 6] = ["Name and password", "Randomness", "Recovery words", "Verification", "Securing", "Ready"];
const RESTORE_STEPS: [&str; 3] = ["Recovery words", "Securing", "Ready"];

fn rail(ui: &mut Ui, rect: Rect, logo: &egui::TextureHandle, steps: &[&str], cur: usize, demo: bool) {
    let ctx = ui.ctx().clone();
    let real = anim::time(&ctx);
    ui.painter().rect_filled(rect, 0.0, t::BG2);
    ui.painter().line_segment([rect.right_top(), rect.right_bottom()], Stroke::new(1.0, t::LINE));
    ui.scope_builder(UiBuilder::new().max_rect(Rect::from_min_size(rect.min + vec2(30.0, 30.0), vec2(rect.width() - 50.0, 40.0))), |ui| {
        ui.horizontal(|ui| {
            brand(ui, logo, 30.0);
            if demo {
                t::chip(ui, "Demo", t::ROSE);
            }
        });
    });
    let (y0, gap, cx) = (rect.top() + 150.0, 64.0, rect.left() + 42.0);
    let p = ui.painter();
    for (i, name) in steps.iter().enumerate() {
        let cy = y0 + i as f32 * gap;
        let c = pos2(cx, cy);
        if i + 1 < steps.len() {
            let (a, b) = (pos2(cx, cy + 16.0), pos2(cx, cy + gap - 16.0));
            p.line_segment([a, b], Stroke::new(1.0, t::LINE2));
            let k = anim::fade(&ctx, Id::new(("rail-line", i)), i < cur, 0.7);
            if k > 0.0 {
                p.line_segment([a, a + (b - a) * k], Stroke::new(1.5, fx::nacre(i as f32 / steps.len() as f32)));
            }
        }
        let done = anim::fade(&ctx, Id::new(("rail-done", i)), i < cur, 0.5);
        let now_on = anim::fade(&ctx, Id::new(("rail-cur", i)), i == cur, 0.4);
        p.circle_stroke(c, 11.0, Stroke::new(1.5, anim::mix(t::LINE2, t::MINT, done.max(now_on))));
        if done > 0.0 {
            p.circle_filled(c, 11.0 * anim::ease_out_back(done).min(1.12), t::MINT);
            fx::check(p, c, 10.0, done, t::ACCENT_INK, 2.0);
        }
        if now_on > 0.0 {
            let ph = ((real as f32) * 0.9) % 1.0;
            p.circle_filled(c, 4.0 * now_on, t::MINT);
            p.circle_stroke(c, 11.0 + 7.0 * ph, Stroke::new(1.5, t::alpha(t::MINT, (90.0 * (1.0 - ph) * now_on) as u8)));
            if !anim::reduced() {
                ctx.request_repaint();
            }
        }
        let col = anim::mix(anim::mix(t::DIM, t::INK, now_on), t::MUTE, done * (1.0 - now_on));
        p.text(pos2(cx + 30.0, cy - 11.0), Align2::LEFT_TOP, format!("STEP 0{}", i + 1), t::mono(9.5), anim::mix(t::DIM, t::MINT, now_on));
        p.text(pos2(cx + 30.0, cy + 2.0), Align2::LEFT_TOP, *name, t::sans_semibold(15.0), col);
    }
    p.text(pos2(rect.left() + 30.0, rect.bottom() - 70.0), Align2::LEFT_TOP, "Nothing leaves this computer.", t::sans(12.5), t::MUTE);
    p.text(pos2(rect.left() + 30.0, rect.bottom() - 50.0), Align2::LEFT_TOP, "No account. No server. No tracking.", t::sans(12.5), t::DIM);
}

fn step_title(ui: &mut Ui, t0: f64, n: usize, name: &str, plain: &str, accent: &str) {
    reveal(ui, t0, 0.0, |ui| t::eyebrow(ui, Some(&format!("0{n}")), name));
    ui.add_space(10.0);
    reveal(ui, t0, 0.08, |ui| t::headline(ui, plain, accent, 40.0));
    ui.add_space(14.0);
}

// ------------------------------------------------------------------------------------------------ create

pub fn create(ui: &mut Ui, a: &mut App) {
    let ctx = ui.ctx().clone();
    let (real, fx_now, pointer) = (anim::time(&ctx), anim::fx_time(&ctx), ctx.pointer_latest_pos());
    let demo = a.demo;
    let App { logo, cmds, screen, .. } = a;
    let Screen::Create(c) = screen else { return };
    let full = ui.max_rect();
    let rail_w = 290.0;
    let rail_rect = Rect::from_min_max(full.min, pos2(full.left() + rail_w, full.bottom()));
    let stage = Rect::from_min_max(pos2(full.left() + rail_w, full.top()), full.max);
    rail(ui, rail_rect, logo, &CREATE_STEPS, c.step as usize, demo);
    // the faint mark behind the stage (not on the steps that have their own)
    if c.step != 1 && c.step != 4 && c.step != 5 {
        fx::sphere(ui.painter(), pos2(stage.right() - 150.0, stage.bottom() - 120.0), 230.0, 14.0, 0.16, fx_now, pointer);
        if !anim::reduced() {
            ctx.request_repaint();
        }
    }
    ui.scope_builder(UiBuilder::new().max_rect(stage), |ui| {
        ui.add_space(22.0);
        ui.horizontal(|ui| {
            ui.add_space(stage.width() - 120.0);
            if c.step < 4 && t::link(ui, "Cancel").clicked() {
                cmds.push(Cmd::GoStart);
            }
        });
        let t0 = c.t_step;
        match c.step {
            0 => stage_scroll(ui, 0, |ui| step_name(ui, c, cmds, t0)),
            1 => step_entropy(ui, c, cmds, stage, real, fx_now, pointer),
            2 => stage_scroll(ui, 2, |ui| step_words(ui, c, cmds, t0, real)),
            3 => stage_scroll(ui, 3, |ui| step_verify(ui, c, cmds, t0)),
            4 => step_securing(ui, c, stage, real, fx_now),
            _ => step_ready(ui, c.done.clone(), c.t_step, stage, real, false, cmds),
        }
    });
}

fn stage_scroll(ui: &mut Ui, step: u8, add: impl FnOnce(&mut Ui)) {
    // one scroll position per step: a new step starts at the top
    egui::ScrollArea::vertical().id_salt(("stage", step)).auto_shrink([false, false]).show(ui, |ui| {
        column(ui, 620.0, |ui| {
            ui.add_space(30.0);
            add(ui);
            ui.add_space(60.0);
        });
    });
}

fn step_name(ui: &mut Ui, c: &mut Create, cmds: &mut Vec<Cmd>, t0: f64) {
    step_title(ui, t0, 1, "Name and password", "Create your ", "wallet.");
    reveal(ui, t0, 0.14, |ui| {
        t::body(ui, "Choose a name for the file and a password. The password encrypts the wallet file on this computer; you will need it to send, sign or register.");
    });
    ui.add_space(18.0);
    reveal(ui, t0, 0.2, |ui| {
        t::field(ui, "Wallet name", &mut c.name, "my-wallet", false, false);
        ui.add_space(8.0);
        t::field(ui, &format!("Password ({MIN_PASSWORD} characters or more)"), &mut c.password, "", true, false);
        let (lvl, txt) = password_strength(&c.password);
        ui.add_space(2.0);
        t::strength_meter(ui, lvl, if c.password.is_empty() { "" } else { txt });
        ui.add_space(8.0);
        let again = t::field(ui, "Password again", &mut c.confirm, "", true, false);
        if again.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            cmds.push(Cmd::CreateNext);
        }
        if !c.confirm.is_empty() {
            ui.add_space(2.0);
            if c.confirm == c.password {
                ui.label(RichText::new("The two passwords match.").font(t::sans(13.0)).color(t::OK));
            } else {
                ui.label(RichText::new("The two passwords are different.").font(t::sans(13.0)).color(t::BAD));
            }
        }
    });
    ui.add_space(16.0);
    reveal(ui, t0, 0.3, |ui| {
        t::notice(ui, Tone::Warn, "Nobody can recover this password: not us, not anyone. If you lose it, the 24 words you are about to write down are the only way back in.");
    });
    ui.add_space(14.0);
    reveal(ui, t0, 0.38, |ui| {
        t::card(ui, |ui| {
            t::eyebrow(ui, None, "What protects the file");
            ui.add_space(8.0);
            for l in ["Argon2id key derivation: 64 MiB of memory, 3 passes, slows down every guess", "ChaCha20-Poly1305: authenticated encryption, any change to the file is detected", "24 words (256 bits) from your system's random generator and your own movements"] {
                ui.horizontal_top(|ui| {
                    t::dot(ui, t::MINT);
                    ui.add(egui::Label::new(RichText::new(l).font(t::sans(13.5)).color(t::INK2)).wrap());
                });
            }
        });
    });
    if let Some(e) = &c.error {
        ui.add_space(10.0);
        t::notice(ui, Tone::Bad, e);
    }
    ui.add_space(20.0);
    reveal(ui, t0, 0.46, |ui| {
        if t::primary(ui, "Continue", true).clicked() {
            cmds.push(Cmd::CreateNext);
        }
    });
}

// ---- step 2: randomness

fn step_entropy(ui: &mut Ui, c: &mut Create, cmds: &mut Vec<Cmd>, stage: Rect, real: f64, fx_now: f64, pointer: Option<Pos2>) {
    let ctx = ui.ctx().clone();
    let t0 = c.t_step;
    // gather the movements of the pointer over the stage
    if let Some(p) = pointer {
        if stage.contains(p) && c.entropy.progress() < 1.0 && c.entropy.add(p, real) {
            ctx.request_repaint();
        }
    }
    let prog = ctx.animate_value_with_time(Id::new("entropy-prog"), c.entropy.progress(), if anim::reduced() { 0.0 } else { 0.35 });
    let done = c.entropy.progress() >= 1.0;
    let center = pos2(stage.center().x, stage.top() + 388.0);
    let r = 100.0;
    let p = ui.painter().clone();
    fx::halo(&p, center, r + 80.0, 0.25 + 0.5 * prog, fx_now);
    fx::ring(&p, center, r + 26.0, 3.0, prog);
    let k = anim::ease_out(anim::progress(&ctx, t0, 0.0, 0.8));
    fx::sphere(&p, center, r * (0.8 + 0.2 * k), 8.6, (0.10 + 0.90 * prog) * k, fx_now, pointer);
    // the trail of the pointer
    for (pos, born, hue) in c.entropy.dots.iter() {
        let age = ((real - born) / 1.6) as f32;
        if (0.0..1.0).contains(&age) {
            p.circle_filled(*pos, 1.0 + 3.6 * (1.0 - age), t::alpha(fx::nacre(*hue), (230.0 * (1.0 - age)) as u8));
        }
    }
    if !anim::reduced() {
        ctx.request_repaint();
    }
    if done {
        let ck = anim::fade(&ctx, Id::new("entropy-done"), true, 0.7);
        p.circle_filled(center, 38.0 * ck, t::alpha(t::MINT, (240.0 * ck) as u8));
        fx::check(&p, center, 40.0, ck, t::ACCENT_INK, 4.5);
    }
    // the words above it
    ui.scope_builder(UiBuilder::new().max_rect(Rect::from_min_max(pos2(stage.center().x - 310.0, stage.top() + 56.0), pos2(stage.center().x + 310.0, stage.bottom()))), |ui| {
        ui.set_width(620.0);
        step_title(ui, t0, 2, "Randomness", "Add your own ", "randomness.");
        reveal(ui, t0, 0.14, |ui| {
            t::body(ui, "Move the mouse (or your finger) around the window. Your movements are mixed with your computer's secure random generator to build the key: an extra, never a replacement.");
        });
    });
    let below = center.y + r + 40.0;
    p.text(
        pos2(center.x, below),
        Align2::CENTER_TOP,
        if done { "RANDOMNESS COLLECTED".to_string() } else { format!("{}%", (prog * 100.0).floor() as u32) },
        if done { t::mono_semibold(13.0) } else { t::mono_semibold(22.0) },
        if done { t::OK } else { t::INK },
    );
    ui.scope_builder(UiBuilder::new().max_rect(Rect::from_min_max(pos2(stage.center().x - 310.0, below + 36.0), pos2(stage.center().x + 310.0, stage.bottom()))), |ui| {
        ui.set_width(620.0);
        ui.horizontal(|ui| {
            if t::ghost(ui, "Back").clicked() {
                cmds.push(Cmd::CreateBack);
            }
            if t::primary(ui, "Generate my 24 words", done).clicked() {
                cmds.push(Cmd::CreateNext);
            }
            if !done && t::link(ui, "Skip: system generator only").clicked() {
                cmds.push(Cmd::CreateNext);
            }
        });
    });
}

// ---- step 3: the words

/// The 24 words as numbered cards. `reveal_t0`: when they were revealed (they decode one after the other); `None`: hidden.
pub fn word_cards(ui: &mut Ui, words: &[String], reveal_t0: Option<f64>, mark: Option<usize>) {
    let ctx = ui.ctx().clone();
    let (real, w) = (anim::time(&ctx), ui.available_width());
    let gap = 8.0;
    let cw = (w - gap * 3.0) / 4.0;
    let ch = 46.0;
    let (rect, _) = ui.allocate_exact_size(vec2(w, ch * 6.0 + gap * 5.0), Sense::hover());
    for i in 0..24 {
        let (col, row) = (i / 6, i % 6);
        let r = Rect::from_min_size(rect.min + vec2(col as f32 * (cw + gap), row as f32 * (ch + gap)), vec2(cw, ch));
        let (shown, flash) = match reveal_t0 {
            Some(t0) => {
                let p = (((real - t0) as f32) - i as f32 * 0.045) / 0.55;
                if p < 1.0 && !anim::reduced() {
                    ctx.request_repaint();
                }
                let p = if anim::reduced() { 1.0 } else { p };
                (p.clamp(0.0, 1.0), (1.0 - p.clamp(0.0, 1.0)).max(0.0))
            }
            None => (-1.0, 0.0),
        };
        let hot = mark == Some(i);
        ui.painter().rect(r, 0.0, t::PANEL, Stroke::new(1.0, anim::mix(if hot { t::MINT } else { t::LINE2 }, t::MINT, flash)), StrokeKind::Inside);
        ui.painter().text(r.left_top() + vec2(12.0, 6.0), Align2::LEFT_TOP, format!("{:02}", i + 1), t::mono(9.5), if hot { t::MINT } else { t::DIM });
        let text = if shown < 0.0 { "••••••".to_string() } else { anim::scramble(words.get(i).map_or("", |s| s.as_str()), shown, i as u64 + 1, real) };
        ui.painter().text(r.left_bottom() + vec2(12.0, -7.0), Align2::LEFT_BOTTOM, text, t::mono_medium(16.0), if shown < 0.0 { t::DIM } else if hot { t::MINT } else { t::INK });
    }
}

fn step_words(ui: &mut Ui, c: &mut Create, cmds: &mut Vec<Cmd>, t0: f64, real: f64) {
    step_title(ui, t0, 3, "Recovery words", "Your ", "24 recovery words.");
    reveal(ui, t0, 0.14, |ui| {
        t::body(ui, "Write them on paper, in this order. They are the master key of your wallet: whoever has them has your funds, and with them you can restore the wallet on any computer.");
    });
    ui.add_space(14.0);
    let words = c.words();
    // they hide again by themselves
    if let Some(t) = c.reveal_t0 {
        if real - t > WORDS_VISIBLE_SECS {
            c.reveal_t0 = None;
        }
    }
    reveal(ui, t0, 0.2, |ui| {
        word_cards(ui, &words, c.reveal_t0, None);
    });
    ui.add_space(10.0);
    match c.reveal_t0 {
        None => {
            reveal(ui, t0, 0.3, |ui| {
                ui.horizontal(|ui| {
                    if t::primary(ui, "Show my 24 words", true).clicked() {
                        c.reveal_t0 = Some(real);
                    }
                    ui.label(RichText::new("Make sure nobody is looking at your screen.").font(t::sans(13.5)).color(t::MUTE));
                });
            });
        }
        Some(t) => {
            let left = (WORDS_VISIBLE_SECS - (real - t)).max(0.0);
            ui.horizontal(|ui| {
                if t::ghost(ui, "Hide them").clicked() {
                    c.reveal_t0 = None;
                }
                ui.label(RichText::new(format!("They hide again in {} s.", left.ceil() as u32)).font(t::sans(13.0)).color(t::MUTE));
            });
            let (bar, _) = ui.allocate_exact_size(vec2(ui.available_width(), 2.0), Sense::hover());
            ui.painter().rect_filled(bar, 0.0, t::LINE);
            ui.painter().rect_filled(Rect::from_min_size(bar.min, vec2(bar.width() * (left / WORDS_VISIBLE_SECS) as f32, 2.0)), 0.0, t::MINT);
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
        }
    }
    ui.add_space(14.0);
    reveal(ui, t0, 0.36, |ui| {
        t::notice(ui, Tone::Warn, "Do not photograph them, do not type them anywhere else, do not send them to anyone. Nobody from Vearl will ever ask for them.");
    });
    ui.add_space(14.0);
    t::checkbox(ui, &mut c.written, "I have written the 24 words on paper and I keep that paper safe.");
    if let Some(e) = &c.error {
        ui.add_space(10.0);
        t::notice(ui, Tone::Bad, e);
    }
    ui.add_space(20.0);
    if t::primary(ui, "Continue", c.reveal_t0.is_some() && c.written).clicked() {
        cmds.push(Cmd::CreateNext);
    }
}

// ---- step 4: verification

fn step_verify(ui: &mut Ui, c: &mut Create, cmds: &mut Vec<Cmd>, t0: f64) {
    step_title(ui, t0, 4, "Verification", "Check your ", "paper.");
    reveal(ui, t0, 0.14, |ui| {
        t::body(ui, "Type the three words below, from your paper. This proves that you copied them correctly: a mistake here would mean losing the wallet later.");
    });
    ui.add_space(14.0);
    let words = c.words();
    let idx = c.check_idx;
    let ctx = ui.ctx().clone();
    let mut all_ok = true;
    let all_words_right = (0..3).all(|k| words.get(idx[k]).is_some_and(|w| c.check_in[k].trim().to_lowercase() == *w));
    for k in 0..3 {
        let right = words.get(idx[k]).is_some_and(|w| c.check_in[k].trim().to_lowercase() == *w);
        all_ok &= right;
        reveal(ui, t0, 0.2 + k as f32 * 0.08, |ui| {
            let r = t::field(ui, &format!("Word number {}", idx[k] + 1), &mut c.check_in[k], "", false, true);
            if k == 2 && r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && all_words_right {
                cmds.push(Cmd::CreateFinish);
            }
            let ok = anim::fade(&ctx, Id::new(("verify-ok", k)), right, 0.35);
            if ok > 0.0 {
                ui.painter().rect_stroke(r.rect, 0.0, Stroke::new(1.0, t::alpha(t::OK, (255.0 * ok) as u8)), StrokeKind::Inside);
                fx::check(ui.painter(), pos2(r.rect.right() - 24.0, r.rect.center().y), 14.0, ok, t::OK, 2.4);
            }
        });
        ui.add_space(6.0);
    }
    if let Some(e) = &c.error {
        ui.add_space(6.0);
        t::notice(ui, Tone::Bad, e);
    }
    ui.add_space(18.0);
    ui.horizontal(|ui| {
        if t::ghost(ui, "Back").clicked() {
            cmds.push(Cmd::CreateBack);
        }
        if t::primary(ui, "Create the wallet", all_ok).clicked() {
            cmds.push(Cmd::CreateFinish);
        }
    });
}

// ---- step 5: securing

struct Stage {
    at: f32,
    title: &'static str,
    sub: &'static str,
}

const SECURING: [Stage; 4] = [
    Stage { at: 0.0, title: "Keys derived from your 24 words", sub: "Ed25519 and SLH-DSA accounts, on this computer" },
    Stage { at: 0.8, title: "Encryption key derived from your password", sub: "Argon2id, 64 MiB of memory, 3 passes" },
    Stage { at: 1.7, title: "Recovery phrase encrypted", sub: "ChaCha20-Poly1305, authenticated" },
    Stage { at: 2.5, title: "File written, read back and unlocked once", sub: "A wallet that cannot be opened is never announced" },
];
const SECURING_MIN: f32 = 3.3;

fn step_securing(ui: &mut Ui, c: &mut Create, stage: Rect, real: f64, fx_now: f64) {
    let (t0, secure_t0, result) = (c.t_step, c.secure_t0, c.result.clone());
    let elapsed = (real - secure_t0) as f32;
    securing_view(ui, stage, t0, elapsed, real, fx_now, "Securing your ", "wallet.", result.as_ref().and_then(|r| r.as_ref().err().cloned()));
    match result {
        Some(Ok(entry)) if elapsed >= SECURING_MIN => {
            // the wallet is on disk: nothing secret needs to stay in memory
            zeroize::Zeroize::zeroize(&mut c.password);
            zeroize::Zeroize::zeroize(&mut c.confirm);
            for w in c.check_in.iter_mut() {
                zeroize::Zeroize::zeroize(w);
            }
            c.phrase = None;
            wipe_text_state(ui.ctx());
            c.done = Some(entry);
            c.step = 5;
            c.t_step = real;
            ui.ctx().request_repaint();
        }
        Some(Err(_)) => {
            ui.scope_builder(UiBuilder::new().max_rect(Rect::from_min_max(pos2(stage.center().x - 310.0, stage.bottom() - 120.0), pos2(stage.center().x + 310.0, stage.bottom() - 40.0))), |ui| {
                if t::ghost(ui, "Back").clicked() {
                    c.step = 3;
                    c.result = None;
                    c.t_step = real;
                }
            });
        }
        _ => {}
    }
}

/// The ring and the checklist shown while the file is made (restore uses it too).
fn securing_view(ui: &mut Ui, stage: Rect, t0: f64, elapsed: f32, real: f64, fx_now: f64, plain: &str, accent: &str, error: Option<String>) {
    let ctx = ui.ctx().clone();
    let p = ui.painter().clone();
    let center = pos2(stage.center().x, stage.top() + 190.0);
    let k = anim::ease_out(anim::progress(&ctx, t0, 0.0, 0.7));
    fx::halo(&p, center, 150.0, 0.6 * k, fx_now);
    let frac = (elapsed / SECURING_MIN).clamp(0.0, 1.0);
    let shown = ctx.animate_value_with_time(Id::new("securing-frac"), frac, if anim::reduced() { 0.0 } else { 0.2 });
    fx::ring(&p, center, 92.0, 3.5, shown);
    fx::sphere(&p, center, 78.0 * (0.8 + 0.2 * k), 7.0, (0.25 + 0.75 * shown) * k, fx_now, None);
    if error.is_none() {
        fx::spinner_at(&p, center, 92.0, 3.4, real);
    }
    if !anim::reduced() {
        ctx.request_repaint();
    }
    ui.scope_builder(UiBuilder::new().max_rect(Rect::from_min_max(pos2(stage.center().x - 290.0, stage.top() + 330.0), pos2(stage.center().x + 290.0, stage.bottom()))), |ui| {
        ui.set_width(580.0);
        t::headline(ui, plain, accent, 34.0);
        ui.add_space(14.0);
        for (i, s) in SECURING.iter().enumerate() {
            let on = anim::fade(&ctx, Id::new(("securing-row", i)), elapsed >= s.at, 0.4);
            let done = elapsed >= SECURING.get(i + 1).map_or(SECURING_MIN, |n| n.at);
            let dn = anim::fade(&ctx, Id::new(("securing-done", i)), done, 0.35);
            ui.scope(|ui| {
                ui.set_opacity(on);
                let (row, _) = ui.allocate_exact_size(vec2(580.0, 46.0), Sense::hover());
                let c = pos2(row.left() + 12.0, row.top() + 14.0);
                let pp = ui.painter();
                pp.circle_stroke(c, 8.0, Stroke::new(1.5, anim::mix(t::LINE2, t::OK, dn)));
                if dn > 0.0 {
                    pp.circle_filled(c, 8.0 * dn, t::OK);
                    fx::check(pp, c, 8.0, dn, t::ACCENT_INK, 1.8);
                } else if on > 0.5 {
                    fx::spinner_at(pp, c, 8.0, 2.0, real);
                }
                pp.text(row.left_top() + vec2(34.0, 0.0), Align2::LEFT_TOP, s.title, t::sans_medium(15.0), t::INK);
                pp.text(row.left_top() + vec2(34.0, 21.0), Align2::LEFT_TOP, s.sub, t::mono(11.0), t::MUTE);
            });
        }
        if let Some(e) = error {
            ui.add_space(10.0);
            t::notice(ui, Tone::Bad, &e);
        }
    });
}

// ---- step 6: ready

fn step_ready(ui: &mut Ui, entry: Option<WalletEntry>, t0: f64, stage: Rect, real: f64, restored: bool, cmds: &mut Vec<Cmd>) {
    let Some(e) = entry else { return };
    let ctx = ui.ctx().clone();
    let el = (real - t0) as f32;
    let p = ui.painter().clone();
    let center = pos2(stage.center().x, stage.top() + 130.0);
    let fx_now = anim::fx_time(&ctx);
    fx::halo(&p, center, 150.0, 0.8 * anim::ease_out(el / 1.2), fx_now);
    let ring_k = anim::ease_out((el / 0.9).clamp(0.0, 1.0));
    fx::ring(&p, center, 58.0, 4.0, ring_k);
    let pop = anim::ease_out_back(((el - 0.5) / 0.45).clamp(0.0, 1.0));
    p.circle_filled(center, 46.0 * pop, t::MINT);
    fx::check(&p, center, 42.0, anim::ease_in_out(((el - 0.7) / 0.45).clamp(0.0, 1.0)), t::ACCENT_INK, 5.0);
    if !anim::reduced() {
        fx::burst(&p, center, el - 0.95);
        if el < 3.0 {
            ctx.request_repaint();
        }
    }
    ui.scope_builder(UiBuilder::new().max_rect(Rect::from_min_max(pos2(stage.center().x - 310.0, stage.top() + 230.0), pos2(stage.center().x + 310.0, stage.bottom()))), |ui| {
        ui.set_width(620.0);
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.set_width(620.0);
            reveal(ui, t0, 0.5, |ui| {
                ui.vertical_centered(|ui| {
                    t::tracked(ui, if restored { "RESTORED" } else { "READY" }, t::mono_medium(11.5), t::MINT, 1.8);
                    ui.add_space(8.0);
                    t::headline(ui, "Your wallet is ", if restored { "back." } else { "ready." }, 40.0);
                });
            });
            ui.add_space(18.0);
            reveal(ui, t0, 0.7, |ui| {
                t::hero_card(ui, |ui| {
                    t::label(ui, "Your address");
                    ui.add_space(4.0);
                    let shown = anim::typed(&e.address, el - 1.0, 70.0);
                    let cursor = if shown.chars().count() < e.address.chars().count() { "|" } else { "" };
                    ui.add(egui::Label::new(RichText::new(format!("{shown}{cursor}")).font(t::mono(14.0)).color(t::INK)).wrap());
                    ui.add_space(8.0);
                    t::copy_button(ui, "ready-address", &e.address);
                });
            });
            ui.add_space(14.0);
            reveal(ui, t0, 0.9, |ui| {
                ui.columns(3, |cols| {
                    for (i, (a, b)) in [("24 words", if restored { "restored" } else { "written and checked" }), ("Argon2id", "+ ChaCha20-Poly1305"), ("On this computer", "never sent anywhere")].into_iter().enumerate() {
                        cols[i].vertical(|ui| {
                            t::card(ui, |ui| {
                                ui.label(RichText::new(a).font(t::sans_semibold(15.0)).color(t::INK));
                                ui.label(RichText::new(b).font(t::sans(12.5)).color(t::MUTE));
                            });
                        });
                    }
                });
            });
            ui.add_space(10.0);
            reveal(ui, t0, 1.0, |ui| t::muted(ui, &format!("Saved in {}", e.path.display())));
            ui.add_space(16.0);
            reveal(ui, t0, 1.1, |ui| {
                t::notice(ui, Tone::Info, "To receive tokens, give this address to the sender. Testnet tokens have no value. When the public testnet is open, mine with this wallet from the miner, or ask for tokens on the Discord.");
            });
            ui.add_space(20.0);
            reveal(ui, t0, 1.2, |ui| {
                if t::primary(ui, "Open my wallet", true).clicked() {
                    cmds.push(Cmd::OpenWallet(e.clone()));
                }
            });
            ui.add_space(40.0);
        });
    });
}

// ------------------------------------------------------------------------------------------------ restore

pub fn restore(ui: &mut Ui, a: &mut App) {
    let ctx = ui.ctx().clone();
    let (real, fx_now, pointer) = (anim::time(&ctx), anim::fx_time(&ctx), ctx.pointer_latest_pos());
    let demo = a.demo;
    let App { logo, cmds, screen, .. } = a;
    let Screen::Restore(r) = screen else { return };
    let full = ui.max_rect();
    let rail_rect = Rect::from_min_max(full.min, pos2(full.left() + 290.0, full.bottom()));
    let stage = Rect::from_min_max(pos2(full.left() + 290.0, full.top()), full.max);
    rail(ui, rail_rect, logo, &RESTORE_STEPS, r.step as usize, demo);
    if r.step == 0 {
        fx::sphere(ui.painter(), pos2(stage.right() - 150.0, stage.bottom() - 120.0), 230.0, 14.0, 0.16, fx_now, pointer);
        if !anim::reduced() {
            ctx.request_repaint();
        }
    }
    ui.scope_builder(UiBuilder::new().max_rect(stage), |ui| {
        ui.add_space(22.0);
        ui.horizontal(|ui| {
            ui.add_space(stage.width() - 120.0);
            if r.step == 0 && t::link(ui, "Cancel").clicked() {
                cmds.push(Cmd::GoStart);
            }
        });
        let t0 = r.t_step;
        match r.step {
            0 => stage_scroll(ui, 10, |ui| {
                step_title(ui, t0, 1, "Recovery words", "Restore from ", "24 words.");
                reveal(ui, t0, 0.14, |ui| {
                    t::body(ui, "Type or paste your 24 recovery words, separated by spaces. A new password will protect the wallet file on this computer (it can be different from the old one).");
                });
                ui.add_space(16.0);
                reveal(ui, t0, 0.2, |ui| {
                    t::field(ui, "Wallet name", &mut r.name, "restored-wallet", false, false);
                    ui.add_space(8.0);
                    t::field_multiline(ui, "Recovery phrase", &mut r.phrase, "word1 word2 word3 ...", 4, true);
                });
                let mut enter = false;
                let cleaned = clean_phrase(&r.phrase);
                let n = cleaned.split_whitespace().count();
                let unknown = unknown_words(&cleaned);
                ui.add_space(2.0);
                if n > 0 {
                    let (txt, col) = if !unknown.is_empty() {
                        (format!("{n} of 24 words. Not in the word list: {}", unknown.iter().take(4).cloned().collect::<Vec<_>>().join(", ")), t::BAD)
                    } else if n == 24 {
                        ("24 words.".to_string(), t::OK)
                    } else {
                        (format!("{n} of 24 words."), t::MUTE)
                    };
                    ui.label(RichText::new(txt).font(t::sans(13.0)).color(col));
                    // the progress towards 24
                    let (bar, _) = ui.allocate_exact_size(vec2(ui.available_width(), 2.0), Sense::hover());
                    ui.painter().rect_filled(bar, 0.0, t::LINE);
                    let k = ctx.animate_value_with_time(Id::new("restore-words"), (n as f32 / 24.0).min(1.0), 0.25);
                    ui.painter().rect_filled(Rect::from_min_size(bar.min, vec2(bar.width() * k, 2.0)), 0.0, if unknown.is_empty() { t::MINT } else { t::BAD });
                }
                ui.add_space(8.0);
                reveal(ui, t0, 0.28, |ui| {
                    t::field(ui, &format!("New password ({MIN_PASSWORD} characters or more)"), &mut r.password, "", true, false);
                    let (lvl, txt) = password_strength(&r.password);
                    ui.add_space(2.0);
                    t::strength_meter(ui, lvl, if r.password.is_empty() { "" } else { txt });
                    ui.add_space(8.0);
                    let again = t::field(ui, "Password again", &mut r.confirm, "", true, false);
                    if again.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        enter = true;
                    }
                });
                if let Some(e) = &r.error {
                    ui.add_space(12.0);
                    t::notice(ui, Tone::Bad, e);
                }
                ui.add_space(20.0);
                let ready = n == 24 && unknown.is_empty() && !r.password.is_empty();
                if t::primary(ui, "Restore the wallet", ready).clicked() || (enter && ready) {
                    cmds.push(Cmd::RestoreFinish);
                }
            }),
            1 => {
                let elapsed = (real - r.secure_t0) as f32;
                let err = r.result.as_ref().and_then(|x| x.as_ref().err().cloned());
                securing_view(ui, stage, t0, elapsed, real, fx_now, "Restoring your ", "wallet.", err);
                match r.result.clone() {
                    Some(Ok(entry)) if elapsed >= SECURING_MIN => {
                        zeroize::Zeroize::zeroize(&mut r.phrase);
                        zeroize::Zeroize::zeroize(&mut r.password);
                        zeroize::Zeroize::zeroize(&mut r.confirm);
                        wipe_text_state(&ctx);
                        r.done = Some(entry);
                        r.step = 2;
                        r.t_step = real;
                        ctx.request_repaint();
                    }
                    Some(Err(_)) => {
                        ui.scope_builder(UiBuilder::new().max_rect(Rect::from_min_max(pos2(stage.center().x - 310.0, stage.bottom() - 120.0), pos2(stage.center().x + 310.0, stage.bottom() - 40.0))), |ui| {
                            if t::ghost(ui, "Back").clicked() {
                                r.step = 0;
                                r.result = None;
                                r.t_step = real;
                            }
                        });
                    }
                    _ => {}
                }
            }
            _ => step_ready(ui, r.done.clone(), r.t_step, stage, real, true, cmds),
        }
    });
}
