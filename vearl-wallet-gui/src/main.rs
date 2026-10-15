//! Vearl Wallet: the wallet of the Vearl testnet with a window (Windows and Linux). It uses the same keystore and the same code as
//! the command-line wallet `vearl-wallet`: a wallet created in one opens in the other.
//!
//!   vearl-wallet-gui            the wallet
//!   vearl-wallet-gui --demo     for recordings: made-up figures, no network, nothing is sent (files go to a temporary folder)
#![cfg_attr(all(windows, not(test)), windows_subsystem = "windows")]

mod anim;
mod app;
mod core;
mod demodata;
#[cfg(feature = "devshots")]
mod demo;
#[cfg(feature = "devshots")]
mod record;
#[cfg(test)]
mod e2e;
#[cfg(test)]
mod smoke;
mod fx;
mod httpget;
mod platform;
mod theme;
mod ui_start;
mod ui_wallet;

use app::{App, Screen};
use eframe::egui;

struct Shell {
    app: App,
    styled: bool,
    capture_hidden: bool,
    /// When the last frame ended: frames are never closer than about 15 ms (a graphics driver that ignores the display refresh would
    /// otherwise make the window redraw as fast as it can while something moves).
    last_frame: std::time::Instant,
    #[cfg(feature = "devshots")]
    rec: Option<record::Recorder>,
}

impl eframe::App for Shell {
    #[cfg(feature = "devshots")]
    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw: &mut egui::RawInput) {
        if let Some(r) = &mut self.rec {
            r.hook(raw);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if !self.styled {
            platform::style_window(frame);
            self.styled = true;
        }
        self.app.poll();
        self.app.tick(ui.ctx());
        // the window disappears from screenshots and recordings while the 24 words are on screen
        let hide = self.app.settings.protect_capture && self.app.words_on_screen();
        if hide != self.capture_hidden {
            platform::capture_protection(frame, hide);
            self.capture_hidden = hide;
        }
        // Ctrl+L: lock the wallet now
        if matches!(self.app.screen, Screen::Wallet(_)) && ui.ctx().input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::L)) {
            self.app.cmds.push(app::Cmd::GoStart);
        }
        #[cfg(feature = "devshots")]
        demo::drive(&mut self.app, ui.ctx());
        #[cfg(feature = "devshots")]
        if let Some(r) = &mut self.rec {
            r.drive(&mut self.app, ui.ctx());
        }
        draw(&mut self.app, ui);
        let min = std::time::Duration::from_millis(std::env::var("VEARL_GUI_FRAME_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(15));
        let el = self.last_frame.elapsed();
        if el < min {
            std::thread::sleep(min - el);
        }
        self.last_frame = std::time::Instant::now();
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.027, 0.031, 0.043, 1.0]
    }
}

/// Draws the current screen (everything below the platform handling of the window).
pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    if !matches!(app.screen, Screen::Wallet(_)) {
        // the cards of these screens are staged by their own reveals: forget the stagger a wallet page left behind (its counter
        // grows with every frame, so a card drawn afterwards would never finish appearing)
        anim::scene_begin(ui.ctx(), -100.0);
        egui::Panel::bottom("footer").exact_size(46.0).resizable(false).frame(egui::Frame::new().fill(theme::BG)).show(ui, ui_start::footer);
    }
    let plain = egui::Frame::new().fill(theme::BG);
    match &app.screen {
        Screen::Wallet(_) => ui_wallet::wallet(ui, app),
        Screen::Start(_) => {
            egui::CentralPanel::default().frame(plain.inner_margin(egui::Margin::symmetric(32, 0))).show(ui, |ui| ui_start::start(ui, app));
        }
        Screen::Create(_) => {
            egui::CentralPanel::default().frame(plain).show(ui, |ui| ui_start::create(ui, app));
        }
        Screen::Restore(_) => {
            egui::CentralPanel::default().frame(plain).show(ui, |ui| ui_start::restore(ui, app));
        }
    }
    app.run_cmds();
}

fn decode_png(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().expect("logo png");
    let mut buf = vec![0; reader.output_buffer_size().expect("logo size")];
    let info = reader.next_frame(&mut buf).expect("logo frame");
    buf.truncate(info.buffer_size());
    (buf, info.width, info.height)
}

fn main() -> eframe::Result {
    let demo = std::env::args().any(|a| a == "--demo") || std::env::var_os("VEARL_GUI_DEMO").is_some();
    if demo {
        // everything the demo writes (wallets it creates, settings) goes to a temporary folder
        std::env::set_var("VEARL_WALLET_HOME", std::env::temp_dir().join("vearl-wallet-demo"));
    }
    platform::harden_process();
    let (icon_rgba, iw, ih) = decode_png(include_bytes!("../assets/logo-256.png"));
    let viewport = egui::ViewportBuilder::default()
        .with_title("Vearl Wallet")
        .with_inner_size([1120.0, 760.0])
        .with_min_inner_size([900.0, 700.0])
        .with_icon(egui::IconData { rgba: icon_rgba, width: iw, height: ih });
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native(
        "Vearl Wallet",
        options,
        Box::new(move |cc| {
            theme::install(&cc.egui_ctx);
            let (rgba, w, h) = decode_png(include_bytes!("../assets/logo-256.png"));
            let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
            let logo = cc.egui_ctx.load_texture("logo", image, egui::TextureOptions::LINEAR);
            #[allow(unused_mut)]
            let mut app = App::new(cc.egui_ctx.clone(), logo, demo);
            if demo && std::env::var_os("VEARL_GUI_OPEN").is_some() {
                // recordings: start inside the demo wallet
                if let Some(w) = app.wallets.first().cloned() {
                    app.cmds.push(app::Cmd::OpenWallet(w));
                }
            }
            #[cfg(feature = "devshots")]
            demo::init(&mut app);
            Ok(Box::new(Shell {
                app,
                styled: false,
                capture_hidden: false,
                last_frame: std::time::Instant::now(),
                #[cfg(feature = "devshots")]
                rec: record::Recorder::from_env(),
            }))
        }),
    )
}
