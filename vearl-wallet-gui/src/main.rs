//! Vearl Wallet: the wallet of the Vearl testnet with a window (Windows and Linux). It uses the same keystore and the same code as
//! the command-line wallet `vearl-wallet`: a wallet created in one opens in the other.
#![cfg_attr(all(windows, not(test)), windows_subsystem = "windows")]

mod app;
mod core;
#[cfg(test)]
mod e2e;
#[cfg(feature = "devshots")]
mod demo;
mod httpget;
mod theme;
mod ui_start;
mod ui_wallet;

use app::{App, Screen};
use eframe::egui;

struct Shell {
    app: App,
}

impl eframe::App for Shell {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.app.poll();
        #[cfg(feature = "devshots")]
        demo::drive(&mut self.app, ui.ctx());
        let bg = egui::Frame::new().fill(theme::BG).inner_margin(egui::Margin::symmetric(32, 0));
        if !matches!(self.app.screen, Screen::Wallet(_)) {
            egui::Panel::bottom("footer").exact_size(46.0).resizable(false).frame(egui::Frame::new().fill(theme::BG)).show(ui, ui_start::footer);
        }
        match &self.app.screen {
            Screen::Wallet(_) => ui_wallet::wallet(ui, &mut self.app),
            Screen::Start(_) => {
                egui::CentralPanel::default().frame(bg).show(ui, |ui| ui_start::start(ui, &mut self.app));
            }
            Screen::Create(_) => {
                egui::CentralPanel::default().frame(bg).show(ui, |ui| ui_start::create(ui, &mut self.app));
            }
            Screen::Restore(_) => {
                egui::CentralPanel::default().frame(bg).show(ui, |ui| ui_start::restore(ui, &mut self.app));
            }
        }
        self.app.run_cmds();
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.027, 0.031, 0.043, 1.0]
    }
}

fn decode_png(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().expect("logo png");
    let mut buf = vec![0; reader.output_buffer_size().expect("logo size")];
    let info = reader.next_frame(&mut buf).expect("logo frame");
    buf.truncate(info.buffer_size());
    (buf, info.width, info.height)
}

fn main() -> eframe::Result {
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
        Box::new(|cc| {
            theme::install(&cc.egui_ctx);
            let (rgba, w, h) = decode_png(include_bytes!("../assets/logo-256.png"));
            let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
            let logo = cc.egui_ctx.load_texture("logo", image, egui::TextureOptions::LINEAR);
            #[allow(unused_mut)]
            let mut app = App::new(cc.egui_ctx.clone(), logo);
            #[cfg(feature = "devshots")]
            demo::init(&mut app);
            Ok(Box::new(Shell { app }))
        }),
    )
}
