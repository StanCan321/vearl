//! The screens of an open wallet: overview, send, receive, activity, rewards, tools, settings, and the dialogs.
use crate::app::*;
use crate::core::*;
use crate::theme::{self as t, Kind, Tone};
use crate::ui_start::{column, word_grid};
use eframe::egui::{self, pos2, vec2, Align, Color32, Frame, Layout, Margin, Rect, RichText, Sense, Stroke, StrokeKind, Ui, UiBuilder};
use std::time::Instant;
use vearl_core::wallet::fmt_amount;

pub fn wallet(ui: &mut Ui, a: &mut App) {
    let explorer = a.explorer();
    let App { logo, cmds, screen, settings, .. } = a;
    let Screen::Wallet(s) = screen else { return };

    egui::Panel::left("nav").exact_size(236.0).resizable(false).frame(Frame::new().fill(t::BG2).inner_margin(Margin::symmetric(0, 0))).show(ui, |ui| {
        sidebar(ui, logo, s, cmds);
    });
    egui::CentralPanel::default().frame(Frame::new().fill(t::BG).inner_margin(Margin::symmetric(36, 0))).show(ui, |ui| {
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.add_space(34.0);
            column(ui, 880.0, |ui| {
                if let Some((tone, msg, at)) = s.toast.clone() {
                    if at.elapsed().as_secs() < 8 {
                        t::notice(ui, tone, &msg);
                        ui.add_space(14.0);
                        ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
                    } else {
                        s.toast = None;
                    }
                }
                match s.page {
                    Page::Overview => overview(ui, s, cmds, &explorer),
                    Page::Send => send(ui, s, cmds),
                    Page::Receive => receive(ui, s, cmds),
                    Page::Activity => activity(ui, s, cmds, &explorer),
                    Page::Rewards => rewards(ui, s, cmds),
                    Page::Tools => tools(ui, s, cmds),
                    Page::Settings => settings_page(ui, s, cmds, settings, &explorer),
                }
                ui.add_space(70.0);
            });
        });
    });
    modal(ui.ctx(), s, cmds, &explorer);

    // refresh in the background every 15 seconds while the window is open
    let due = s.last_refresh.is_none_or(|t| t.elapsed() >= REFRESH_EVERY);
    if due && !s.refreshing && s.last_refresh.is_some() {
        cmds.push(Cmd::Refresh);
        cmds.push(Cmd::RefreshActivity);
    }
    ui.ctx().request_repaint_after(std::time::Duration::from_secs(5));
}

// ------------------------------------------------------------------------------------------------ sidebar

fn sidebar(ui: &mut Ui, logo: &egui::TextureHandle, s: &mut Session, cmds: &mut Vec<Cmd>) {
    let full = ui.max_rect();
    ui.painter().line_segment([full.right_top(), full.right_bottom()], Stroke::new(1.0, t::LINE));
    ui.add_space(26.0);
    ui.horizontal(|ui| {
        ui.add_space(24.0);
        crate::ui_start::brand(ui, logo, 30.0);
    });
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.add_space(24.0);
        t::chip(ui, "Testnet", t::LAV);
    });
    ui.add_space(22.0);
    ui.spacing_mut().item_spacing.y = 2.0;
    for (page, n, name) in Page::ALL {
        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 42.0), Sense::click());
        let on = s.page == page;
        let hover = resp.hovered();
        if on {
            ui.painter().rect_filled(rect, 0.0, t::alpha(t::MINT, 12));
            ui.painter().rect_filled(Rect::from_min_size(rect.left_top(), vec2(2.0, rect.height())), 0.0, t::MINT);
        } else if hover {
            ui.painter().rect_filled(rect, 0.0, t::alpha(t::INK, 6));
        }
        let col = if on { t::MINT } else if hover { t::INK } else { t::MUTE };
        t::paint_tracked(ui.painter(), pos2(rect.left() + 24.0, rect.center().y - 7.5), n, &t::mono(11.0), if on { t::MINT } else { t::DIM }, 1.2);
        t::paint_tracked(ui.painter(), pos2(rect.left() + 58.0, rect.center().y - 8.0), &name.to_uppercase(), &t::mono_medium(12.0), col, 1.6);
        if hover {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if resp.clicked() {
            s.page = page;
            if page == Page::Rewards && s.registered.is_none() {
                cmds.push(Cmd::CheckRegistered);
            }
            if page == Page::Activity {
                cmds.push(Cmd::RefreshActivity);
            }
        }
    }
    // bottom block: the wallet, the network, lock
    let bottom = full.bottom() - 18.0;
    let block = Rect::from_min_max(pos2(full.left() + 24.0, bottom - 160.0), pos2(full.right() - 20.0, bottom));
    ui.scope_builder(UiBuilder::new().max_rect(block), |ui| {
        ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
            if t::button_ex(ui, "Lock wallet", Kind::Ghost, true, false, ui.available_width()).clicked() {
                cmds.push(Cmd::GoStart);
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                match &s.status {
                    Some(Ok(st)) => {
                        t::dot(ui, t::OK);
                        ui.label(RichText::new(format!("Block {}", fmt_int(st.height))).font(t::mono(12.0)).color(t::INK2));
                    }
                    Some(Err(_)) => {
                        t::dot(ui, t::BAD);
                        ui.label(RichText::new("Offline").font(t::mono(12.0)).color(t::BAD));
                    }
                    None => {
                        t::dot(ui, t::WARN);
                        ui.label(RichText::new("Connecting...").font(t::mono(12.0)).color(t::MUTE));
                    }
                }
            });
            ui.add_space(8.0);
            ui.label(RichText::new(short_addr(&s.entry.address)).font(t::mono(11.5)).color(t::MUTE));
            ui.label(RichText::new(&s.entry.name).font(t::sans_semibold(14.5)).color(t::INK));
        });
    });
}

fn fmt_int(v: u64) -> String {
    let (i, _) = fmt_grouped(v * vearl_core::state::COIN);
    i
}

// ------------------------------------------------------------------------------------------------ pieces

fn page_title(ui: &mut Ui, n: &str, eyebrow: &str, plain: &str, accent: &str) {
    page_title_ex(ui, n, eyebrow, plain, accent, true);
}

/// `rule`: draw the gradient line under the title (not above a card that has its own gradient edge).
fn page_title_ex(ui: &mut Ui, n: &str, eyebrow: &str, plain: &str, accent: &str, rule: bool) {
    t::eyebrow(ui, Some(n), eyebrow);
    ui.add_space(4.0);
    t::headline(ui, plain, accent, 38.0);
    ui.add_space(6.0);
    if rule {
        t::nacre_line(ui);
    }
    ui.add_space(22.0);
}

/// The balance written big: "1,234.56" in serif, the six last decimals small and dim.
fn balance_text(ui: &mut Ui, v: Option<u64>, size: f32) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        match v {
            Some(v) => {
                let (int, dec) = fmt_grouped(v);
                ui.label(RichText::new(format!("{int}.{}", &dec[..2])).font(t::serif(size)).color(t::INK));
                ui.label(RichText::new(&dec[2..]).font(t::mono(size * 0.30)).color(t::DIM));
            }
            None => {
                ui.label(RichText::new("—").font(t::serif(size)).color(t::DIM));
            }
        }
        ui.add_space(12.0);
        ui.label(RichText::new(SYMBOL).font(t::mono_semibold(size * 0.28)).color(t::MINT));
    });
}

fn address_box(ui: &mut Ui, id: &str, title: &str, addr: &str) {
    t::label(ui, title);
    ui.add_space(-2.0);
    Frame::new().fill(t::BG2).stroke(Stroke::new(1.0, t::LINE2)).inner_margin(Margin::symmetric(14, 12)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_wrapped(|ui| {
            ui.add(egui::Label::new(RichText::new(addr).font(t::mono(13.5)).color(t::INK)).wrap().selectable(true));
        });
    });
    ui.add_space(2.0);
    t::copy_button(ui, id, addr);
}

fn explorer_link(explorer: &str, id: &str) -> Cmd {
    Cmd::Link(tx_url(explorer, id))
}

fn activity_row(ui: &mut Ui, r: &TxRow, cmds: &mut Vec<Cmd>, explorer: &str) {
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 66.0), Sense::click());
    let hover = resp.hovered();
    if hover {
        ui.painter().rect_filled(rect, 0.0, t::alpha(t::INK, 6));
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let incoming = r.direction == "in";
    let c = if incoming { t::MINT } else { t::LAV };
    let ic = pos2(rect.left() + 22.0, rect.center().y);
    ui.painter().rect_stroke(Rect::from_center_size(ic, vec2(30.0, 30.0)), 0.0, Stroke::new(1.0, t::alpha(c, 110)), StrokeKind::Inside);
    // arrow down-left (received) or up-right (sent)
    let (d1, d2) = if incoming { (vec2(5.0, -5.0), vec2(-5.0, 5.0)) } else { (vec2(-5.0, 5.0), vec2(5.0, -5.0)) };
    let st = Stroke::new(1.6, c);
    ui.painter().line_segment([ic + d1, ic + d2], st);
    let head = ic + d2;
    let (h1, h2) = if incoming { (vec2(5.0, 0.0), vec2(0.0, -5.0)) } else { (vec2(-5.0, 0.0), vec2(0.0, 5.0)) };
    ui.painter().line_segment([head, head + h1], st);
    ui.painter().line_segment([head, head + h2], st);
    let title = if r.label == "Transfer" { if incoming { "Received".to_string() } else { "Sent".to_string() } } else { r.label.clone() };
    ui.painter().text(pos2(rect.left() + 52.0, rect.top() + 14.0), egui::Align2::LEFT_TOP, title, t::sans_semibold(15.5), t::INK);
    let who = r.counterparty.as_deref().map(|a| format!("{} {}", if incoming { "from" } else { "to" }, short_addr(a))).unwrap_or_else(|| short_addr(&r.id));
    ui.painter().text(pos2(rect.left() + 52.0, rect.top() + 38.0), egui::Align2::LEFT_TOP, who, t::mono(12.0), t::MUTE);
    if let Some(v) = r.amount {
        let (int, dec) = fmt_grouped(v);
        ui.painter().text(pos2(rect.right() - 16.0, rect.top() + 12.0), egui::Align2::RIGHT_TOP, format!("{}{int}.{}", if r.label == "Transfer" { if incoming { "+" } else { "-" } } else { "" }, &dec[..2]), t::mono_medium(17.0), if incoming { t::MINT } else { t::INK });
    }
    let status = if r.pending { "PENDING".to_string() } else { r.confirmations.map_or("CONFIRMED".to_string(), |c| format!("{} CONFIRMATION{}", c, if c == 1 { "" } else { "S" })) };
    ui.painter().text(pos2(rect.right() - 16.0, rect.top() + 40.0), egui::Align2::RIGHT_TOP, status, t::mono(10.0), if r.pending { t::WARN } else { t::DIM });
    ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, t::LINE));
    if resp.clicked() && !r.id.is_empty() {
        cmds.push(explorer_link(explorer, &r.id));
    }
}

// ------------------------------------------------------------------------------------------------ overview

fn overview(ui: &mut Ui, s: &mut Session, cmds: &mut Vec<Cmd>, explorer: &str) {
    page_title_ex(ui, "01", "Overview", "Your ", "balance.", false);
    t::hero_card(ui, |ui| {
        t::label(ui, "Available");
        ui.add_space(6.0);
        balance_text(ui, s.account.as_ref().map(|a| a.balance), 66.0);
        let problem = match &s.status {
            Some(Err(e)) => Some(e.clone()),
            _ => s.account_error.clone(),
        };
        if let Some(e) = problem {
            ui.add_space(8.0);
            t::notice(ui, Tone::Warn, &format!("The balance could not be read. {e}"));
            ui.add_space(6.0);
            if t::ghost(ui, "Try again").clicked() {
                cmds.push(Cmd::Refresh);
            }
        }
        ui.add_space(16.0);
        ui.horizontal(|ui| {
            if t::primary(ui, "Send", true).clicked() {
                s.page = Page::Send;
            }
            if t::ghost(ui, "Receive").clicked() {
                s.page = Page::Receive;
            }
        });
    });
    ui.add_space(16.0);
    t::card(ui, |ui| {
        address_box(ui, "ov-address", "Your address", &s.entry.address.clone());
    });
    ui.add_space(16.0);
    t::card(ui, |ui| {
        t::eyebrow(ui, None, "Post-quantum account");
        ui.add_space(8.0);
        t::muted(ui, "Besides the usual account, your recovery words also give a second account whose signatures (SLH-DSA, NIST FIPS 205) cannot be forged by a quantum computer. Its payments are larger and take about two seconds to sign.");
        ui.add_space(10.0);
        match s.pq_address.clone() {
            Some(a) => {
                balance_text(ui, s.pq_account.as_ref().map(|x| x.balance), 34.0);
                ui.add_space(8.0);
                address_box(ui, "ov-pq", "Post-quantum address", &a);
            }
            None => {
                if t::ghost(ui, "Show it (password)").clicked() {
                    cmds.push(Cmd::Ask(Action::RevealPq));
                }
            }
        }
    });
    ui.add_space(22.0);
    ui.horizontal(|ui| {
        t::eyebrow(ui, None, "Recent activity");
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if t::link(ui, "See all").clicked() {
                s.page = Page::Activity;
            }
        });
    });
    ui.add_space(6.0);
    match &s.activity {
        Some(Ok(rows)) if !rows.is_empty() => {
            for r in rows.iter().take(5) {
                activity_row(ui, r, cmds, explorer);
            }
        }
        Some(Ok(_)) => empty_state(ui, "No transaction yet.", "Payments you send or receive appear here."),
        Some(Err(e)) => empty_state(ui, "The activity list is not available.", e),
        None => {
            ui.add_space(14.0);
            ui.add(egui::Spinner::new().color(t::MINT));
        }
    }
}

fn empty_state(ui: &mut Ui, title: &str, sub: &str) {
    t::card(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(10.0);
            ui.label(RichText::new(title).font(t::sans_medium(15.5)).color(t::INK2));
            ui.add(egui::Label::new(RichText::new(sub).font(t::sans(13.0)).color(t::MUTE)).wrap());
            ui.add_space(10.0);
        });
    });
}

// ------------------------------------------------------------------------------------------------ send

fn send(ui: &mut Ui, s: &mut Session, cmds: &mut Vec<Cmd>) {
    page_title(ui, "02", "Send", "Send ", "tokens.");
    let balance = if s.send.pq { s.pq_account.as_ref().map(|a| a.balance) } else { s.account.as_ref().map(|a| a.balance) };
    t::card(ui, |ui| {
        t::field(ui, "To (address)", &mut s.send.to, "vrl1...", false, true);
        ui.add_space(10.0);
        ui.columns(2, |cols| {
            cols[0].vertical(|ui| {
                t::field(ui, &format!("Amount ({SYMBOL})"), &mut s.send.amount, "0.00", false, true);
                ui.add_space(4.0);
                if let Some(b) = balance {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("Available: {} {SYMBOL}", fmt_amount(b))).font(t::sans(12.5)).color(t::MUTE));
                        if t::link(ui, "Max").clicked() {
                            let fee = vearl_core::wallet::parse_amount(s.send.fee.trim()).unwrap_or(0);
                            s.send.amount = fmt_amount(b.saturating_sub(fee)).trim_end_matches('0').trim_end_matches('.').to_string();
                        }
                    });
                }
            });
            cols[1].vertical(|ui| {
                t::field(ui, &format!("Fee ({SYMBOL}, paid to the miner)"), &mut s.send.fee, "0.01", false, true);
            });
        });
        ui.add_space(10.0);
        t::checkbox(ui, &mut s.send.pq, "Sign with my post-quantum key (pays from the post-quantum account, larger transaction, about 2 seconds)");
        if s.send.pq && s.pq_address.is_none() {
            ui.add_space(4.0);
            ui.label(RichText::new("The post-quantum balance is shown once its address is unlocked (Overview, then 'Show it').").font(t::sans(12.5)).color(t::MUTE));
        }
        if let Some(e) = &s.send.error {
            ui.add_space(12.0);
            t::notice(ui, Tone::Bad, e);
        }
        ui.add_space(18.0);
        if t::primary(ui, "Review the payment", true).clicked() {
            match crate::app::validate_send(&s.send.to, &s.send.amount, &s.send.fee, balance) {
                Ok((to, amount, fee)) => {
                    s.send.error = None;
                    cmds.push(Cmd::Ask(Action::Send { to, amount, fee, pq: s.send.pq }));
                }
                Err(e) => s.send.error = Some(e),
            }
        }
    });
    ui.add_space(14.0);
    t::muted(ui, "Testnet tokens have no value. Check the address carefully: a payment cannot be taken back. You will see a summary and be asked for your password before anything is sent.");
}

// ------------------------------------------------------------------------------------------------ receive

fn qr_matrix(data: &str) -> Option<(usize, Vec<bool>)> {
    let code = qrcode::QrCode::new(data.as_bytes()).ok()?;
    let w = code.width();
    Some((w, code.to_colors().into_iter().map(|c| c == qrcode::Color::Dark).collect()))
}

fn paint_qr(ui: &mut Ui, data: &str, size: f32) {
    let (rect, _) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, Color32::WHITE);
    if let Some((w, cells)) = qr_matrix(data) {
        let quiet = 2.0;
        let cell = size / (w as f32 + 2.0 * quiet);
        for y in 0..w {
            for x in 0..w {
                if cells[y * w + x] {
                    let p = rect.min + vec2((x as f32 + quiet) * cell, (y as f32 + quiet) * cell);
                    ui.painter().rect_filled(Rect::from_min_size(p, vec2(cell + 0.4, cell + 0.4)), 0.0, Color32::from_rgb(0x07, 0x08, 0x0b));
                }
            }
        }
    }
}

fn receive(ui: &mut Ui, s: &mut Session, cmds: &mut Vec<Cmd>) {
    page_title_ex(ui, "03", "Receive", "Get paid ", "here.", false);
    t::hero_card(ui, |ui| {
        ui.horizontal_top(|ui| {
            paint_qr(ui, &s.entry.address, 220.0);
            ui.add_space(22.0);
            ui.vertical(|ui| {
                ui.set_width(ui.available_width());
                address_box(ui, "rx-address", "Your address", &s.entry.address.clone());
                ui.add_space(14.0);
                t::muted(ui, "Give this address, or let the sender scan the code. It can be shared safely: it only lets people send you tokens.");
            });
        });
    });
    ui.add_space(16.0);
    t::card(ui, |ui| {
        t::eyebrow(ui, None, "Post-quantum address");
        ui.add_space(8.0);
        match s.pq_address.clone() {
            Some(a) => {
                ui.horizontal_top(|ui| {
                    paint_qr(ui, &a, 140.0);
                    ui.add_space(18.0);
                    ui.vertical(|ui| {
                        ui.set_width(ui.available_width());
                        address_box(ui, "rx-pq", "Post-quantum address", &a);
                    });
                });
            }
            None => {
                t::muted(ui, "A second address, protected by signatures that a quantum computer cannot forge. Showing it needs your password once.");
                ui.add_space(8.0);
                if t::ghost(ui, "Show it (password)").clicked() {
                    cmds.push(Cmd::Ask(Action::RevealPq));
                }
            }
        }
    });
}

// ------------------------------------------------------------------------------------------------ activity

fn activity(ui: &mut Ui, s: &mut Session, cmds: &mut Vec<Cmd>, explorer: &str) {
    page_title(ui, "04", "Activity", "What ", "happened.");
    ui.horizontal(|ui| {
        if t::ghost(ui, "Refresh").clicked() {
            cmds.push(Cmd::RefreshActivity);
            cmds.push(Cmd::Refresh);
        }
        if t::ghost(ui, "Open the explorer").clicked() {
            cmds.push(Cmd::Link(format!("{}/explorer/#/address/{}", explorer.trim_end_matches('/'), s.entry.address)));
        }
    });
    ui.add_space(12.0);
    match &s.activity {
        Some(Ok(rows)) if !rows.is_empty() => {
            for r in rows {
                activity_row(ui, r, cmds, explorer);
            }
            ui.add_space(8.0);
            t::muted(ui, "The list shows your payments and transactions. Mining rewards are visible in the explorer.");
        }
        Some(Ok(_)) => empty_state(ui, "No transaction yet.", "Payments you send or receive appear here."),
        Some(Err(e)) => {
            t::notice(ui, Tone::Warn, &format!("The activity list could not be loaded. {e}"));
        }
        None => {
            ui.add(egui::Spinner::new().color(t::MINT));
        }
    }
}

// ------------------------------------------------------------------------------------------------ rewards

fn rewards(ui: &mut Ui, s: &mut Session, cmds: &mut Vec<Cmd>) {
    page_title(ui, "05", "Rewards", "Testnet ", "contributors.");
    t::card(ui, |ui| {
        ui.horizontal(|ui| {
            t::eyebrow(ui, None, "Status");
            match &s.registered {
                Some(Ok(true)) => {
                    t::chip(ui, "Registered", t::OK);
                }
                Some(Ok(false)) => {
                    t::chip(ui, "Not registered yet", t::WARN);
                }
                Some(Err(_)) => {
                    t::chip(ui, "Could not check", t::MUTE);
                }
                None => {
                    ui.add(egui::Spinner::new().color(t::MINT));
                }
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if t::link(ui, "Check again").clicked() {
                    cmds.push(Cmd::CheckRegistered);
                }
            });
        });
        ui.add_space(10.0);
        t::body(ui, "Miners who help secure the public testnet are planned to receive Vearl tokens on mainnet as a thank-you. This is a plan, not a guarantee, and testnet tokens have no value. The rules are in docs/19 of the repository.");
        ui.add_space(8.0);
        if t::link(ui, "Read the rules").clicked() {
            cmds.push(Cmd::Link("https://github.com/vearl-verified-compute/vearl/blob/main/docs/19-testnet-rewards.md".into()));
        }
    });
    ui.add_space(16.0);
    t::card(ui, |ui| {
        t::eyebrow(ui, None, "Your standing");
        ui.add_space(10.0);
        match &s.standing {
            None => {
                ui.add(egui::Spinner::new().color(t::MINT));
            }
            Some(Err(e)) => {
                t::muted(ui, &format!("The leaderboard could not be read: {e}"));
            }
            Some(Ok(None)) => {
                t::body(ui, "This address has not mined a block yet. Points come only from the blocks that your address mines once the public testnet is open.");
            }
            Some(Ok(Some(st))) => {
                let kv = |ui: &mut Ui, k: &str, v: String| {
                    ui.vertical(|ui| {
                        ui.set_min_width(120.0);
                        t::label(ui, k);
                        ui.add_space(-2.0);
                        ui.label(RichText::new(v).font(t::serif(30.0)).color(t::INK));
                    });
                };
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 34.0;
                    kv(ui, "Rank", format!("#{}", st.rank));
                    kv(ui, "Points", format!("{:.2}", st.points));
                    kv(ui, "Blocks", format!("{} ({} with paid work)", st.blocks, st.useful_blocks));
                    kv(ui, "Active days", st.active_days.to_string());
                });
                ui.add_space(10.0);
                if st.eligible {
                    t::chip(ui, "Eligible", t::OK);
                } else {
                    ui.horizontal(|ui| {
                        t::chip(ui, "Not eligible yet", t::WARN);
                    });
                    for r in &st.reasons {
                        ui.label(RichText::new(format!("- {r}")).font(t::sans(13.5)).color(t::MUTE));
                    }
                }
            }
        }
    });
    ui.add_space(16.0);
    if matches!(s.registered, Some(Ok(true))) {
        t::notice(ui, Tone::Ok, "Your address is in the public list of registered addresses.");
        return;
    }
    t::card(ui, |ui| {
        t::eyebrow(ui, Some("1"), "Register this address");
        ui.add_space(10.0);
        t::body(ui, "Enter your Discord name or an e-mail address. The wallet signs a short message with your key (a proof that you own the address, no secret is sent) and shows you exactly what will be e-mailed before it leaves.");
        ui.add_space(12.0);
        t::field(ui, "Contact (Discord name or e-mail)", &mut s.rewards.contact, "name or name@example.org", false, false);
        if let Some(e) = &s.rewards.error {
            ui.add_space(8.0);
            t::notice(ui, Tone::Bad, e);
        }
        ui.add_space(14.0);
        if t::primary(ui, "Sign the registration", true).clicked() {
            match vearl_core::regmail::check_contact(s.rewards.contact.trim()) {
                Ok(()) => {
                    s.rewards.error = None;
                    cmds.push(Cmd::Ask(Action::Register { contact: s.rewards.contact.trim().to_string() }));
                }
                Err(e) => s.rewards.error = Some(e),
            }
        }
    });
    if let Some(l) = s.rewards.lines.clone() {
        ui.add_space(16.0);
        t::hero_card(ui, |ui| {
            t::eyebrow(ui, Some("2"), "Check, then send");
            ui.add_space(10.0);
            for (k, v) in [("Address", &l.address), ("Message", &l.message), ("Signature", &l.signature)] {
                t::label(ui, k);
                ui.add_space(-2.0);
                Frame::new().fill(t::BG2).stroke(Stroke::new(1.0, t::LINE2)).inner_margin(Margin::symmetric(12, 9)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.add(egui::Label::new(RichText::new(v.as_str()).font(t::mono(12.0)).color(t::INK2)).wrap().selectable(true));
                });
                ui.add_space(6.0);
            }
            t::muted(ui, &format!("Sending writes ONE e-mail from {} to {} with these three lines. Nothing secret is in it. The server sees your IP address, like any website.", vearl_core::regmail::SENDER, vearl_core::regmail::RECIPIENT));
            ui.add_space(12.0);
            match &s.rewards.sent {
                Some(Ok(())) => {
                    t::notice(ui, Tone::Ok, "Sent. It is processed within a few minutes: your address then appears in the public list (press 'Check again' above).");
                }
                Some(Err(e)) => {
                    t::notice(ui, Tone::Bad, &format!("The e-mail could not be sent: {e}. Some networks block outgoing mail. You can send the three lines yourself."));
                }
                None => {}
            }
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                let busy = s.rewards.sending;
                if t::primary(ui, if busy { "Sending..." } else { "Send the registration" }, !busy && !matches!(s.rewards.sent, Some(Ok(()))) ).clicked() {
                    cmds.push(Cmd::SendMail);
                }
                if busy {
                    ui.add(egui::Spinner::new().color(t::MINT));
                }
            });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                t::copy_button(ui, "reg-lines", &format!("{}\n{}\n{}", l.address, l.message, l.signature));
                ui.label(RichText::new(format!("Copy the three lines to e-mail them yourself to {}", vearl_core::regmail::RECIPIENT)).font(t::sans(12.5)).color(t::MUTE));
            });
        });
    }
}

// ------------------------------------------------------------------------------------------------ tools

fn tools(ui: &mut Ui, s: &mut Session, cmds: &mut Vec<Cmd>) {
    page_title(ui, "06", "Tools", "Sign and ", "verify.");
    t::card(ui, |ui| {
        t::eyebrow(ui, Some("1"), "Sign a message");
        ui.add_space(8.0);
        t::muted(ui, "Prove that you own this address by signing a text with your key. Nothing is sent anywhere.");
        ui.add_space(10.0);
        t::field_multiline(ui, "Message", &mut s.tools.message, "Any text", 3, false);
        ui.add_space(12.0);
        if t::button_ex(ui, "Sign (password)", Kind::Primary, !s.tools.message.is_empty(), false, 0.0).clicked() {
            cmds.push(Cmd::Ask(Action::Sign { message: s.tools.message.clone() }));
        }
        if let Some(l) = s.tools.signed.clone() {
            ui.add_space(14.0);
            for (k, v) in [("Address", &l.address), ("Message", &l.message), ("Signature", &l.signature)] {
                t::label(ui, k);
                ui.add_space(-2.0);
                Frame::new().fill(t::BG2).stroke(Stroke::new(1.0, t::LINE2)).inner_margin(Margin::symmetric(12, 9)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.add(egui::Label::new(RichText::new(v.as_str()).font(t::mono(12.0)).color(t::INK2)).wrap().selectable(true));
                });
                ui.add_space(6.0);
            }
            t::copy_button(ui, "sig-all", &format!("address:   {}\nmessage:   {}\nsignature: {}", l.address, l.message, l.signature));
        }
    });
    ui.add_space(16.0);
    t::card(ui, |ui| {
        t::eyebrow(ui, Some("2"), "Verify a signature");
        ui.add_space(8.0);
        t::muted(ui, "Check that a signature was made by the owner of an address.");
        ui.add_space(10.0);
        t::field(ui, "Address", &mut s.tools.v_address, "vrl1...", false, true);
        ui.add_space(6.0);
        t::field_multiline(ui, "Message", &mut s.tools.v_message, "The signed text", 2, false);
        ui.add_space(6.0);
        t::field(ui, "Signature (hexadecimal)", &mut s.tools.v_signature, "", false, true);
        ui.add_space(12.0);
        if t::ghost(ui, "Verify").clicked() {
            s.tools.verdict = Some(verify_text(&s.tools.v_address, &s.tools.v_message, &s.tools.v_signature));
        }
        match &s.tools.verdict {
            Some(Ok(true)) => {
                ui.add_space(10.0);
                t::notice(ui, Tone::Ok, "Valid: this signature was made by the owner of that address.");
            }
            Some(Ok(false)) => {
                ui.add_space(10.0);
                t::notice(ui, Tone::Bad, "INVALID signature.");
            }
            Some(Err(e)) => {
                ui.add_space(10.0);
                t::notice(ui, Tone::Bad, e);
            }
            None => {}
        }
    });
    let _ = cmds;
}

// ------------------------------------------------------------------------------------------------ settings

fn settings_page(ui: &mut Ui, s: &mut Session, cmds: &mut Vec<Cmd>, app_settings: &Settings, explorer: &str) {
    page_title(ui, "07", "Settings", "Your ", "wallet.");
    t::card(ui, |ui| {
        t::eyebrow(ui, Some("1"), "Network");
        ui.add_space(8.0);
        t::muted(ui, "Leave the node empty to use the seed nodes of your release (network.json). Change it only to use your own node.");
        ui.add_space(10.0);
        t::field(ui, "Node (HOST:PORT)", &mut s.settings.node, "seed.vearl.org:9000", false, true);
        ui.add_space(6.0);
        t::field(ui, "Block explorer (for the activity list)", &mut s.settings.explorer, "https://vearl.org", false, true);
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            if t::primary(ui, "Save", true).clicked() {
                cmds.push(Cmd::SaveSettings);
            }
            if t::ghost(ui, "Test the node").clicked() {
                cmds.push(Cmd::TestNode);
            }
            if s.settings.testing {
                ui.add(egui::Spinner::new().color(t::MINT));
            }
        });
        if s.settings.saved && s.settings.node.trim() == app_settings.node && s.settings.explorer.trim() == app_settings.explorer {
            ui.add_space(8.0);
            ui.label(RichText::new("Saved.").font(t::sans(13.0)).color(t::OK));
        }
        match &s.settings.node_test {
            Some(Ok(st)) => {
                ui.add_space(8.0);
                t::notice(ui, Tone::Ok, &format!("Connected to {}: block {}, {} peer(s), {} pending transaction(s).", st.node, st.height, st.peers, st.mempool));
            }
            Some(Err(e)) => {
                ui.add_space(8.0);
                t::notice(ui, Tone::Bad, e);
            }
            None => {}
        }
        let _ = explorer;
    });
    ui.add_space(16.0);
    t::card(ui, |ui| {
        t::eyebrow(ui, Some("2"), "Recovery words");
        ui.add_space(8.0);
        t::muted(ui, "Show the 24 words again (for example to check your paper). Make sure nobody is looking at the screen.");
        ui.add_space(10.0);
        if t::ghost(ui, "Show my 24 words (password)").clicked() {
            cmds.push(Cmd::Ask(Action::ShowPhrase));
        }
    });
    ui.add_space(16.0);
    t::card(ui, |ui| {
        t::eyebrow(ui, Some("3"), "Password");
        ui.add_space(8.0);
        t::muted(ui, "Changing the password re-encrypts the wallet file. The 24 words and the address stay the same.");
        ui.add_space(10.0);
        t::field(ui, "New password (8 characters or more)", &mut s.settings.new_password, "", true, false);
        let (lvl, txt) = password_strength(&s.settings.new_password);
        ui.add_space(2.0);
        t::strength_meter(ui, lvl, if s.settings.new_password.is_empty() { "" } else { txt });
        ui.add_space(6.0);
        t::field(ui, "New password again", &mut s.settings.new_confirm, "", true, false);
        if let Some(e) = &s.settings.pw_error {
            ui.add_space(8.0);
            t::notice(ui, Tone::Bad, e);
        }
        ui.add_space(12.0);
        if t::ghost(ui, "Change the password").clicked() {
            if s.settings.new_password.chars().count() < 8 {
                s.settings.pw_error = Some("The new password needs at least 8 characters.".into());
            } else if s.settings.new_password != s.settings.new_confirm {
                s.settings.pw_error = Some("The two passwords are not the same.".into());
            } else {
                s.settings.pw_error = None;
                cmds.push(Cmd::Ask(Action::ChangePassword { new: s.settings.new_password.clone() }));
            }
        }
    });
    ui.add_space(16.0);
    t::card(ui, |ui| {
        t::eyebrow(ui, Some("4"), "Wallet file");
        ui.add_space(8.0);
        ui.add(egui::Label::new(RichText::new(s.entry.path.display().to_string()).font(t::mono(12.5)).color(t::INK2)).wrap().selectable(true));
        ui.add_space(6.0);
        t::muted(ui, "Back this file up as well as the paper. It is encrypted: without the password it is useless to a thief, and with the 24 words you do not need it.");
        ui.add_space(10.0);
        if t::ghost(ui, "Show in the folder").clicked() {
            cmds.push(Cmd::Reveal(s.entry.path.clone()));
        }
    });
    ui.add_space(16.0);
    t::card(ui, |ui| {
        t::eyebrow(ui, Some("5"), "About");
        ui.add_space(8.0);
        t::body(ui, &format!("Vearl Wallet {}. Testnet software, not audited: tokens have no value.", env!("CARGO_PKG_VERSION")));
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            for (name, url) in [("Website", "https://vearl.org"), ("Source code", "https://github.com/vearl-verified-compute/vearl"), ("Discord", "https://discord.gg/jya8meKnvM")] {
                if t::link(ui, name).clicked() {
                    cmds.push(Cmd::Link(url.into()));
                }
                ui.add_space(10.0);
            }
        });
    });
}

// ------------------------------------------------------------------------------------------------ dialogs

fn modal_frame() -> Frame {
    Frame::new().fill(t::PANEL).stroke(Stroke::new(1.0, t::LINE2)).inner_margin(Margin::same(28))
}

fn modal(ctx: &egui::Context, s: &mut Session, cmds: &mut Vec<Cmd>, explorer: &str) {
    let Some(m) = s.modal.as_mut() else { return };
    let backdrop = Color32::from_black_alpha(190);
    match m {
        Modal::Password(p) => {
            let mut submit = false;
            let mut cancel = false;
            egui::Modal::new(egui::Id::new("pw-modal")).frame(modal_frame()).backdrop_color(backdrop).show(ctx, |ui| {
                ui.set_width(460.0);
                let (title, accent, detail) = prompt_texts(&p.action);
                t::eyebrow(ui, None, "Password needed");
                ui.add_space(8.0);
                t::headline(ui, title, accent, 28.0);
                ui.add_space(12.0);
                if let Action::Send { to, amount, fee, pq } = &p.action {
                    summary_row(ui, "To", to, true);
                    summary_row(ui, "Amount", &format!("{} {SYMBOL}", fmt_amount(*amount)), false);
                    summary_row(ui, "Fee", &format!("{} {SYMBOL}", fmt_amount(*fee)), false);
                    summary_row(ui, "Total", &format!("{} {SYMBOL}", fmt_amount(amount + fee)), false);
                    summary_row(ui, "Signature", if *pq { "Post-quantum (SLH-DSA)" } else { "Standard (Ed25519)" }, false);
                    ui.add_space(10.0);
                } else if !detail.is_empty() {
                    t::muted(ui, detail);
                    ui.add_space(10.0);
                }
                let r = t::field(ui, "Wallet password", &mut p.password, "", true, false);
                if !p.busy && !r.has_focus() && p.password.is_empty() {
                    r.request_focus();
                }
                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    submit = true;
                }
                if let Some(e) = &p.error {
                    ui.add_space(10.0);
                    t::notice(ui, Tone::Bad, e);
                }
                ui.add_space(18.0);
                ui.horizontal(|ui| {
                    if t::button_ex(ui, "Cancel", Kind::Ghost, !p.busy, false, 0.0).clicked() {
                        cancel = true;
                    }
                    if t::button_ex(ui, if p.busy { "Working..." } else { confirm_label(&p.action) }, Kind::Primary, !p.busy, !p.busy, 0.0).clicked() {
                        submit = true;
                    }
                    if p.busy {
                        ui.add(egui::Spinner::new().color(t::MINT));
                    }
                });
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) && !p.busy {
                    cancel = true;
                }
            });
            if submit {
                cmds.push(Cmd::SubmitPassword);
            }
            if cancel {
                cmds.push(Cmd::CancelModal);
            }
        }
        Modal::Phrase(words) => {
            let mut close = false;
            egui::Modal::new(egui::Id::new("phrase-modal")).frame(modal_frame()).backdrop_color(backdrop).show(ctx, |ui| {
                ui.set_width(600.0);
                t::eyebrow(ui, None, "Recovery words");
                ui.add_space(8.0);
                t::headline(ui, "Your ", "24 words.", 30.0);
                ui.add_space(14.0);
                t::card(ui, |ui| word_grid(ui, words, None));
                ui.add_space(12.0);
                t::notice(ui, Tone::Warn, "Whoever sees these words controls your funds. Nobody from Vearl will ever ask for them.");
                ui.add_space(16.0);
                if t::primary(ui, "Hide them", true).clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    close = true;
                }
            });
            if close {
                for w in words.iter_mut() {
                    zeroize::Zeroize::zeroize(w);
                }
                s.modal = None;
            }
        }
        Modal::Sent { txid, from, to, amount, fee } => {
            let mut close = false;
            let (txid, from, to, amount, fee) = (txid.clone(), from.clone(), to.clone(), *amount, *fee);
            egui::Modal::new(egui::Id::new("sent-modal")).frame(modal_frame()).backdrop_color(backdrop).show(ctx, |ui| {
                ui.set_width(520.0);
                t::eyebrow(ui, None, "Accepted by the network");
                ui.add_space(8.0);
                t::headline(ui, "Payment ", "sent.", 30.0);
                ui.add_space(12.0);
                summary_row(ui, "From", &from, true);
                summary_row(ui, "To", &to, true);
                summary_row(ui, "Amount", &format!("{} {SYMBOL}", fmt_amount(amount)), false);
                summary_row(ui, "Fee", &format!("{} {SYMBOL}", fmt_amount(fee)), false);
                summary_row(ui, "Transaction", &txid, true);
                ui.add_space(8.0);
                t::muted(ui, "It becomes final once a miner includes it in a block (about 30 seconds).");
                ui.add_space(16.0);
                ui.horizontal(|ui| {
                    if t::primary(ui, "Done", true).clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        close = true;
                    }
                    if t::ghost(ui, "View in the explorer").clicked() {
                        cmds.push(explorer_link(explorer, &txid));
                    }
                });
            });
            if close {
                s.modal = None;
            }
        }
    }
    let _ = Instant::now();
}

fn summary_row(ui: &mut Ui, k: &str, v: &str, mono: bool) {
    ui.horizontal_top(|ui| {
        let (r, _) = ui.allocate_exact_size(vec2(96.0, 20.0), Sense::hover());
        t::paint_tracked(ui.painter(), r.min + vec2(0.0, 3.0), &k.to_uppercase(), &t::mono(10.5), t::MUTE, 1.4);
        ui.add(egui::Label::new(RichText::new(v).font(if mono { t::mono(12.5) } else { t::sans_medium(15.0) }).color(t::INK)).wrap().selectable(true));
    });
    ui.add_space(2.0);
}

fn prompt_texts(a: &Action) -> (&'static str, &'static str, &'static str) {
    match a {
        Action::RevealPq => ("Show the ", "post-quantum address.", "Your password unlocks the key for a moment to derive the second address. Nothing is sent."),
        Action::ShowPhrase => ("Show the ", "24 words.", "Make sure nobody is looking at your screen."),
        Action::Send { .. } => ("Confirm the ", "payment.", ""),
        Action::Register { .. } => ("Sign the ", "registration.", "The wallet signs a short message. Nothing is sent yet: you will see the three lines first."),
        Action::Sign { .. } => ("Sign the ", "message.", "The signature proves that you own the address. Nothing is sent."),
        Action::ChangePassword { .. } => ("Enter your ", "current password.", "The wallet file is re-encrypted with the new password you typed."),
    }
}

fn confirm_label(a: &Action) -> &'static str {
    match a {
        Action::Send { .. } => "Send now",
        Action::ChangePassword { .. } => "Change password",
        Action::ShowPhrase | Action::RevealPq => "Show",
        _ => "Sign",
    }
}
