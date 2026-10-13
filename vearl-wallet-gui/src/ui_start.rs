//! The screens before a wallet is open: start, create (4 steps), restore.
use crate::app::*;
use crate::core::*;
use crate::theme::{self as t, Kind, Tone};
use eframe::egui::{self, pos2, vec2, Align, Color32, Layout, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Ui, UiBuilder};

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

/// A half-tone sphere of dots lit from the top left, tinted with the nacre gradient: the mark of the site, enlarged.
pub fn halftone(painter: &egui::Painter, center: Pos2, radius: f32, step: f32, opacity: f32) {
    let n = (radius / step).ceil() as i32;
    for gy in -n..=n {
        for gx in -n..=n {
            let off = vec2(gx as f32 * step, gy as f32 * step);
            let d = off.length() / radius;
            if d >= 1.0 {
                continue;
            }
            let nz = (1.0 - d * d).sqrt();
            let nrm = vec2(off.x / radius, off.y / radius);
            let light = (-nrm.x * 0.55 - nrm.y * 0.55 + nz * 0.62).clamp(0.0, 1.0);
            let r = step * 0.5 * (0.12 + 0.95 * light);
            if r < 0.5 {
                continue;
            }
            let u = ((off.x + off.y) / (2.0 * radius) * 0.5 + 0.5).clamp(0.0, 1.0);
            let c = if u < 0.5 { lerp(t::MINT, t::LAV, u * 2.0) } else { lerp(t::LAV, t::ROSE, (u - 0.5) * 2.0) };
            painter.circle_filled(center + off, r, t::alpha(c, (opacity * 255.0 * (0.35 + 0.65 * light)) as u8));
        }
    }
}

fn lerp(a: Color32, b: Color32, k: f32) -> Color32 {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * k) as u8;
    Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
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
    p.text(pos2(r.left() + 28.0, r.center().y), egui::Align2::LEFT_CENTER, "TESTNET WALLET  ·  TOKENS HAVE NO VALUE  ·  NOT AUDITED", t::mono(10.5), t::DIM);
    p.text(pos2(r.right() - 28.0, r.center().y), egui::Align2::RIGHT_CENTER, format!("v{}", env!("CARGO_PKG_VERSION")), t::mono(10.5), t::DIM);
}

// ------------------------------------------------------------------------------------------------ start

pub fn start(ui: &mut Ui, a: &mut App) {
    let App { logo, wallets, cmds, screen, .. } = a;
    let Screen::Start(st) = screen else { return };
    // a wallet file dropped on the window is opened
    if let Some(path) = ui.ctx().input(|i| i.raw.dropped_files.iter().next().map(|f| f.path().to_path_buf())) {
        cmds.push(Cmd::OpenPath(path.display().to_string()));
    }
    ui.add_space(26.0);
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        column(ui, 1080.0, |ui| {
            ui.add_space(6.0);
            brand(ui, logo, 36.0);
            ui.add_space(34.0);
            let total = ui.available_width();
            let gap = 56.0;
            let (lw, rw) = if total > 820.0 { ((total - gap) * 0.52, (total - gap) * 0.48) } else { (total, total) };
            let top = ui.cursor().min;
            let left_rect = Rect::from_min_size(top, vec2(lw, 560.0));
            let mut used_l = 0.0;
            ui.scope_builder(UiBuilder::new().max_rect(left_rect), |ui| {
                ui.set_width(lw);
                t::eyebrow(ui, None, "Vearl wallet");
                ui.add_space(14.0);
                t::headline(ui, "Your keys,\n", "on your machine.", 52.0);
                ui.add_space(18.0);
                ui.add(egui::Label::new(RichText::new("A wallet for the Vearl testnet. Your 24 recovery words never leave this computer, and every payment is signed here, on your side, before it is sent.").font(t::sans(16.5)).color(t::INK2)).wrap());
                ui.add_space(26.0);
                ui.horizontal(|ui| {
                    if t::primary(ui, "Create a wallet", true).clicked() {
                        cmds.push(Cmd::StartCreate);
                    }
                    if t::ghost(ui, "Restore from 24 words").clicked() {
                        cmds.push(Cmd::StartRestore);
                    }
                });
                ui.add_space(26.0);
                for (n, s) in [("Post-quantum", "Accounts can also sign with SLH-DSA (NIST FIPS 205), which quantum computers cannot forge."), ("Verified", "The wallet is open source and uses the same code as the command-line wallet."), ("Yours alone", "Nothing is sent anywhere except the payments and the registration you confirm.")] {
                    ui.horizontal_top(|ui| {
                        t::dot(ui, t::MINT);
                        ui.vertical(|ui| {
                            ui.label(RichText::new(n).font(t::sans_semibold(14.5)).color(t::INK));
                            ui.add(egui::Label::new(RichText::new(s).font(t::sans(13.5)).color(t::MUTE)).wrap());
                        });
                    });
                    ui.add_space(4.0);
                }
                used_l = ui.min_rect().height();
            });
            let wide = total > 820.0;
            let right_rect = if wide { Rect::from_min_size(pos2(top.x + lw + gap, top.y + 130.0), vec2(rw, 560.0)) } else { Rect::from_min_size(pos2(top.x, top.y + used_l + 30.0), vec2(rw, 560.0)) };
            let mut used_r = 0.0;
            if wide {
                // the mark of the site, enlarged, behind the top of the card
                halftone(ui.painter(), pos2(right_rect.center().x + 10.0, top.y + 70.0), 215.0, 11.0, 0.95);
            }
            ui.scope_builder(UiBuilder::new().max_rect(right_rect), |ui| {
                ui.set_width(rw);
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
                        let hover = resp.hovered();
                        ui.painter().rect(rect, 0.0, if hover { t::alpha(t::MINT, 10) } else { Color32::TRANSPARENT }, Stroke::new(1.0, if hover { t::MINT } else { t::LINE2 }), StrokeKind::Inside);
                        ui.painter().text(rect.left_top() + vec2(18.0, 16.0), egui::Align2::LEFT_TOP, &w.name, t::sans_semibold(16.5), t::INK);
                        ui.painter().text(rect.left_top() + vec2(18.0, 41.0), egui::Align2::LEFT_TOP, short_addr(&w.address), t::mono(12.5), t::MUTE);
                        t::draw_arrow(ui.painter(), pos2(rect.right() - 24.0, rect.center().y), if hover { t::MINT } else { t::MUTE });
                        if hover {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
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
                        if let Some(e) = &st.error {
                            ui.add_space(6.0);
                            t::notice(ui, Tone::Bad, e);
                        }
                    } else {
                        if t::link(ui, "Open a wallet file...").clicked() {
                            st.show_open = true;
                        }
                        ui.add_space(2.0);
                        ui.label(RichText::new("You can also drop a wallet file on this window.").font(t::sans(12.5)).color(t::DIM));
                        if let Some(e) = &st.error {
                            ui.add_space(6.0);
                            t::notice(ui, Tone::Bad, e);
                        }
                    }
                });
                used_r = ui.min_rect().height();
            });
            let used = if wide { used_l.max(used_r + 130.0) } else { used_l + 30.0 + used_r };
            ui.allocate_space(vec2(total, used.max(100.0) + 60.0));
        });
    });
}

// ------------------------------------------------------------------------------------------------ create

fn step_header(ui: &mut Ui, step: u8, title_plain: &str, title_accent: &str) {
    let names = ["Name and password", "Write the words down", "Check the words", "Done"];
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        for (i, n) in names.iter().enumerate() {
            let (r, _) = ui.allocate_exact_size(vec2(56.0, 3.0), Sense::hover());
            let on = i as u8 <= step;
            if on {
                t::nacre_rect(ui.painter(), r, 255);
            } else {
                ui.painter().rect_filled(r, 0.0, t::LINE2);
            }
            let _ = n;
        }
    });
    ui.add_space(14.0);
    t::eyebrow(ui, Some(&format!("0{}", step + 1)), names[step as usize]);
    ui.add_space(10.0);
    t::headline(ui, title_plain, title_accent, 38.0);
    ui.add_space(14.0);
}

pub fn create(ui: &mut Ui, a: &mut App) {
    let App { logo, cmds, screen, .. } = a;
    let Screen::Create(c) = screen else { return };
    ui.add_space(22.0);
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        column(ui, 680.0, |ui| {
            ui.horizontal(|ui| {
                brand(ui, logo, 30.0);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if c.step < 3 && !c.busy && t::link(ui, "Cancel").clicked() {
                        cmds.push(Cmd::GoStart);
                    }
                });
            });
            ui.add_space(30.0);
            match c.step {
                0 => {
                    step_header(ui, 0, "Create your ", "wallet.");
                    t::body(ui, "Choose a name for the file and a password. The password encrypts the wallet file on this computer; you will need it to send, sign or register.");
                    ui.add_space(18.0);
                    t::field(ui, "Wallet name", &mut c.name, "my-wallet", false, false);
                    ui.add_space(8.0);
                    t::field(ui, "Password (8 characters or more)", &mut c.password, "", true, false);
                    let (lvl, txt) = password_strength(&c.password);
                    ui.add_space(2.0);
                    t::strength_meter(ui, lvl, if c.password.is_empty() { "" } else { txt });
                    ui.add_space(8.0);
                    t::field(ui, "Password again", &mut c.confirm, "", true, false);
                    if !c.confirm.is_empty() {
                        ui.add_space(2.0);
                        if c.confirm == c.password {
                            ui.label(RichText::new("The two passwords match.").font(t::sans(13.0)).color(t::OK));
                        } else {
                            ui.label(RichText::new("The two passwords are different.").font(t::sans(13.0)).color(t::BAD));
                        }
                    }
                    ui.add_space(16.0);
                    t::notice(ui, Tone::Warn, "Nobody can recover this password: not us, not anyone. If you lose it, the 24 words you are about to write down are the only way back in.");
                    if let Some(e) = &c.error {
                        ui.add_space(10.0);
                        t::notice(ui, Tone::Bad, e);
                    }
                    ui.add_space(20.0);
                    if t::primary(ui, "Continue", true).clicked() {
                        cmds.push(Cmd::CreateNext);
                    }
                }
                1 => {
                    step_header(ui, 1, "Your ", "24 recovery words.");
                    t::body(ui, "Write them on paper, in this order. They are the master key of your wallet: whoever has them has your funds, and with them you can restore the wallet on any computer.");
                    ui.add_space(14.0);
                    let words = c.words();
                    t::card(ui, |ui| {
                        if !c.revealed {
                            ui.add_space(10.0);
                            ui.vertical_centered(|ui| {
                                ui.label(RichText::new("Make sure nobody is looking at your screen.").font(t::sans_medium(15.0)).color(t::INK2));
                                ui.add_space(10.0);
                                if t::primary(ui, "Show my 24 words", true).clicked() {
                                    c.revealed = true;
                                }
                            });
                            ui.add_space(10.0);
                        } else {
                            word_grid(ui, &words, None);
                        }
                    });
                    ui.add_space(14.0);
                    t::notice(ui, Tone::Warn, "Do not photograph them, do not type them anywhere else, do not send them to anyone. Nobody from Vearl will ever ask for them.");
                    ui.add_space(14.0);
                    t::checkbox(ui, &mut c.written, "I have written the 24 words on paper and I keep that paper safe.");
                    if let Some(e) = &c.error {
                        ui.add_space(10.0);
                        t::notice(ui, Tone::Bad, e);
                    }
                    ui.add_space(20.0);
                    ui.horizontal(|ui| {
                        if t::ghost(ui, "Back").clicked() {
                            cmds.push(Cmd::CreateBack);
                        }
                        if t::primary(ui, "Continue", c.revealed).clicked() {
                            cmds.push(Cmd::CreateNext);
                        }
                    });
                }
                2 => {
                    step_header(ui, 2, "Check your ", "paper.");
                    t::body(ui, "Type the three words below, from your paper. This proves that you copied them correctly: a mistake here would mean losing the wallet later.");
                    ui.add_space(14.0);
                    let idx = c.check_idx;
                    for k in 0..3 {
                        t::field(ui, &format!("Word number {}", idx[k] + 1), &mut c.check_in[k], "", false, true);
                        ui.add_space(6.0);
                    }
                    if let Some(e) = &c.error {
                        ui.add_space(6.0);
                        t::notice(ui, Tone::Bad, e);
                    }
                    ui.add_space(18.0);
                    ui.horizontal(|ui| {
                        if t::button_ex(ui, "Back", Kind::Ghost, !c.busy, false, 0.0).clicked() {
                            cmds.push(Cmd::CreateBack);
                        }
                        let ready = !c.busy && c.check_in.iter().all(|w| !w.trim().is_empty());
                        if t::primary(ui, if c.busy { "Encrypting..." } else { "Create the wallet" }, ready).clicked() {
                            cmds.push(Cmd::CreateFinish);
                        }
                        if c.busy {
                            ui.add(egui::Spinner::new().color(t::MINT));
                        }
                    });
                }
                _ => {
                    step_header(ui, 3, "Your wallet ", "is ready.");
                    if let Some(e) = c.done.clone() {
                        t::hero_card(ui, |ui| {
                            t::label(ui, "Your address");
                            ui.add_space(4.0);
                            ui.add(egui::Label::new(RichText::new(&e.address).font(t::mono(14.0)).color(t::INK)).wrap());
                            ui.add_space(8.0);
                            t::copy_button(ui, "new-address", &e.address);
                            ui.add_space(10.0);
                            t::muted(ui, &format!("Saved in {}", e.path.display()));
                        });
                        ui.add_space(14.0);
                        t::notice(ui, Tone::Info, "To receive tokens, give this address to the sender. Testnet tokens have no value. When the public testnet is open, mine with this wallet from the miner, or ask for tokens on the Discord.");
                        ui.add_space(20.0);
                        if t::primary(ui, "Open my wallet", true).clicked() {
                            cmds.push(Cmd::OpenWallet(e));
                        }
                    }
                }
            }
            ui.add_space(60.0);
        });
    });
}

/// The 24 words in four columns of six, numbered like on a sheet of paper. `mark` highlights one number.
pub fn word_grid(ui: &mut Ui, words: &[String], mark: Option<usize>) {
    let w = ui.available_width();
    let col_w = w / 4.0;
    let row_h = 34.0;
    let (rect, _) = ui.allocate_exact_size(vec2(w, row_h * 6.0), Sense::hover());
    for (i, word) in words.iter().enumerate() {
        let (col, row) = (i / 6, i % 6);
        let p = rect.min + vec2(col as f32 * col_w, row as f32 * row_h);
        let hot = mark == Some(i);
        ui.painter().text(p + vec2(0.0, 5.0), egui::Align2::LEFT_TOP, format!("{:>2}", i + 1), t::mono(12.0), if hot { t::MINT } else { t::DIM });
        ui.painter().text(p + vec2(30.0, 2.0), egui::Align2::LEFT_TOP, word, t::mono_medium(16.0), if hot { t::MINT } else { t::INK });
    }
}

// ------------------------------------------------------------------------------------------------ restore

pub fn restore(ui: &mut Ui, a: &mut App) {
    let App { logo, cmds, screen, .. } = a;
    let Screen::Restore(r) = screen else { return };
    ui.add_space(22.0);
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        column(ui, 680.0, |ui| {
            ui.horizontal(|ui| {
                brand(ui, logo, 30.0);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if !r.busy && t::link(ui, if r.done.is_some() { "Back" } else { "Cancel" }).clicked() {
                        cmds.push(Cmd::GoStart);
                    }
                });
            });
            ui.add_space(30.0);
            if let Some(e) = r.done.clone() {
                t::eyebrow(ui, Some("02"), "Done");
                ui.add_space(10.0);
                t::headline(ui, "Wallet ", "restored.", 38.0);
                ui.add_space(14.0);
                t::hero_card(ui, |ui| {
                    t::label(ui, "Your address");
                    ui.add_space(4.0);
                    ui.add(egui::Label::new(RichText::new(&e.address).font(t::mono(14.0)).color(t::INK)).wrap());
                    ui.add_space(8.0);
                    t::copy_button(ui, "restored-address", &e.address);
                });
                ui.add_space(20.0);
                if t::primary(ui, "Open my wallet", true).clicked() {
                    cmds.push(Cmd::OpenWallet(e));
                }
                return;
            }
            t::eyebrow(ui, Some("01"), "Restore");
            ui.add_space(10.0);
            t::headline(ui, "Restore from ", "24 words.", 38.0);
            ui.add_space(14.0);
            t::body(ui, "Type or paste your 24 recovery words, separated by spaces. A new password will protect the wallet file on this computer (it can be different from the old one).");
            ui.add_space(16.0);
            t::field(ui, "Wallet name", &mut r.name, "restored-wallet", false, false);
            ui.add_space(8.0);
            t::field_multiline(ui, "Recovery phrase", &mut r.phrase, "word1 word2 word3 ...", 4, true);
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
            }
            ui.add_space(8.0);
            t::field(ui, "New password (8 characters or more)", &mut r.password, "", true, false);
            let (lvl, txt) = password_strength(&r.password);
            ui.add_space(2.0);
            t::strength_meter(ui, lvl, if r.password.is_empty() { "" } else { txt });
            ui.add_space(8.0);
            t::field(ui, "Password again", &mut r.confirm, "", true, false);
            if let Some(e) = &r.error {
                ui.add_space(12.0);
                t::notice(ui, Tone::Bad, e);
            }
            ui.add_space(20.0);
            ui.horizontal(|ui| {
                if t::primary(ui, if r.busy { "Encrypting..." } else { "Restore the wallet" }, !r.busy && n == 24 && unknown.is_empty() && !r.password.is_empty()).clicked() {
                    cmds.push(Cmd::RestoreFinish);
                }
                if r.busy {
                    ui.add(egui::Spinner::new().color(t::MINT));
                }
            });
            ui.add_space(60.0);
        });
    });
}
