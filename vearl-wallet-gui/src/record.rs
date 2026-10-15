//! Clip recorder (feature `devshots`, NOT in release builds): plays a scene of the real window on a virtual clock and saves every
//! frame, for the video material. Every animation of the window reads the time of egui's input, so replacing that time with an
//! exact clock (1/60 s per frame) makes the clips smooth and repeatable whatever the speed of the machine.
//!   set VEARL_GUI_CLIPS=C:\some\folder        (VEARL_GUI_CLIP=name records one clip only, VEARL_GUI_SCALE=2 sets the pixel density)
//! The pointer of the clip is scripted; the real mouse is ignored while recording. The demo chip is hidden (the video says "demo data").
use crate::app::*;
use crate::core::Account;
use crate::demodata as dd;
use eframe::egui::{self, pos2, Pos2};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use vearl_core::state::COIN;

pub static RECORDING: AtomicBool = AtomicBool::new(false);

const FPS: f64 = 60.0;
const LOGICAL: (f32, f32) = (1120.0, 760.0);

struct Clip {
    name: &'static str,
    secs: f64,
    /// Puts the window in the state of the first frame (`now` is the virtual time of that frame).
    setup: fn(&mut App, f64),
    /// Called before every frame with the time since the start of the clip: changes of state along the clip.
    step: fn(&mut App, f64, f64),
    /// Where the pointer is at that time (None: outside the window).
    pointer: fn(f64) -> Option<Pos2>,
}

fn none(_: f64) -> Option<Pos2> {
    None
}
fn nothing(_: &mut App, _: f64, _: f64) {}

fn smooth(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn clips() -> Vec<Clip> {
    vec![
        // the start screen opening, then the pointer glides over the sphere (its dots swell towards it)
        Clip {
            name: "start",
            secs: 4.0,
            setup: |a, n| {
                a.wallets = dd::entries();
                a.screen = Screen::Start(StartState::new(n, None));
            },
            step: nothing,
            pointer: |t| {
                if t < 1.4 {
                    return None;
                }
                let k = smooth((t - 1.4) / 2.6);
                let (x0, y0, x1, y1) = (1060.0, 560.0, 700.0, 170.0);
                let bend = (k * std::f64::consts::PI).sin() * 90.0;
                Some(pos2((x0 + (x1 - x0) * k + bend) as f32, (y0 + (y1 - y0) * k + bend * 0.4) as f32))
            },
        },
        // randomness: a hand-drawn spiral over the stage fills the ring, then the step says it is done
        Clip {
            name: "entropy",
            secs: 5.0,
            setup: |a, n| demo_create(a, n, 1),
            step: nothing,
            pointer: |t| {
                if t < 0.5 {
                    return None;
                }
                let u = (t - 0.5).min(3.2);
                let a = u * 8.0;
                let r = 105.0 + 65.0 * smooth(u / 2.7) + 14.0 * (u * 5.0).sin();
                Some(pos2((705.0 + a.cos() * r) as f32, (388.0 + a.sin() * r * 0.8) as f32))
            },
        },
        // the 24 words decode one after the other
        Clip {
            name: "words",
            secs: 4.5,
            setup: |a, n| {
                demo_create(a, n, 2);
                if let Screen::Create(c) = &mut a.screen {
                    c.phrase = Some(dd::creation_mnemonic());
                }
            },
            step: |a, t, now| {
                if let Screen::Create(c) = &mut a.screen {
                    if t >= 0.6 && c.reveal_t0.is_none() {
                        c.reveal_t0 = Some(now);
                    }
                }
            },
            pointer: none,
        },
        // three words typed back from the paper
        Clip {
            name: "verify",
            secs: 3.0,
            setup: |a, n| {
                demo_create(a, n, 3);
                if let Screen::Create(c) = &mut a.screen {
                    c.phrase = Some(dd::creation_mnemonic());
                    let e = dd::creation_mnemonic().to_entropy();
                    c.check_idx = [(e[0] % 8) as usize, 8 + (e[1] % 8) as usize, 16 + (e[2] % 8) as usize];
                }
            },
            step: |a, t, _| {
                if let Screen::Create(c) = &mut a.screen {
                    let words = c.words();
                    for k in 0..3 {
                        let w = &words[c.check_idx[k]];
                        let start = 0.5 + k as f64 * 0.75;
                        let n = (((t - start) / 0.09).floor().max(0.0) as usize).min(w.len());
                        c.check_in[k] = w[..n].to_string();
                    }
                }
            },
            pointer: none,
        },
        // securing (the real checklist, paced), then the ready burst and the address typing itself
        Clip {
            name: "securing",
            secs: 7.5,
            setup: |a, n| {
                demo_create(a, n, 4);
                if let Screen::Create(c) = &mut a.screen {
                    c.secure_t0 = n;
                }
            },
            step: |a, t, _| {
                if let Screen::Create(c) = &mut a.screen {
                    if t >= 0.6 && c.result.is_none() && c.step == 4 {
                        c.result = Some(Ok(dd::entry("my-wallet", 11)));
                    }
                }
            },
            pointer: none,
        },
        // the wallet opens and the balance counts up
        Clip {
            name: "overview",
            secs: 4.0,
            setup: |a, n| {
                demo_session(a, n, Page::Overview);
                if let Some(s) = a.session_mut() {
                    s.account = Some(Account { balance: 0, nonce: 3, next_nonce: 4 });
                    s.pq_account = Some(Account { balance: 0, nonce: 0, next_nonce: 0 });
                }
            },
            step: |a, t, _| {
                if t >= 0.55 {
                    if let Some(s) = a.session_mut() {
                        s.account = Some(Account { balance: 1_234 * COIN + 56_789_000, nonce: 3, next_nonce: 4 });
                        s.pq_account = Some(Account { balance: 42 * COIN + 500_000, nonce: 0, next_nonce: 0 });
                    }
                }
            },
            pointer: none,
        },
        // a payment: the form, then its summary with the post-quantum signature
        Clip {
            name: "send",
            secs: 4.0,
            setup: |a, n| {
                demo_session(a, n, Page::Send);
                if let Some(s) = a.session_mut() {
                    s.send.to = dd::entry("x", 9).address;
                    s.send.amount = "12.5".into();
                    s.send.fee = "0.01".into();
                    s.send.pq = true;
                }
            },
            step: |a, t, now| {
                if t >= 1.3 {
                    if let Some(s) = a.session_mut() {
                        if s.modal.is_none() {
                            s.modal = Some(Modal::Password(PwPrompt::new(Action::Send { to: dd::entry("x", 9).address, amount: 12 * COIN + 50_000_000, fee: 1_000_000, pq: true })));
                            s.modal_t0 = now;
                            s.modal_was_open = true;
                        }
                    }
                }
            },
            pointer: none,
        },
        // the payment accepted by the node
        Clip {
            name: "sent",
            secs: 2.5,
            setup: |a, n| {
                demo_session(a, n, Page::Send);
                if let Some(s) = a.session_mut() {
                    s.modal = Some(Modal::Sent { txid: "e5".repeat(32), from: dd::entry("main", 7).address, to: dd::entry("x", 9).address, amount: 12 * COIN + 50_000_000, fee: 1_000_000 });
                    s.modal_t0 = n;
                    s.modal_was_open = true;
                }
            },
            step: nothing,
            pointer: none,
        },
        // receive: the address and its QR code
        Clip {
            name: "receive",
            secs: 3.0,
            setup: |a, n| demo_session(a, n, Page::Receive),
            step: nothing,
            pointer: none,
        },
        // the rewards page with a signed registration
        Clip {
            name: "rewards",
            secs: 3.0,
            setup: |a, n| {
                demo_session(a, n, Page::Rewards);
                if let Some(s) = a.session_mut() {
                    s.rewards.contact = "jane@example.org".into();
                    s.rewards.lines = Some(RegLines { address: s.entry.address.clone(), message: format!("Vearl testnet rewards registration. Address: {}. Contact: jane@example.org. I accept the terms in docs/19.", s.entry.address), signature: "9f".repeat(64) });
                }
            },
            step: nothing,
            pointer: none,
        },
    ]
}

fn demo_create(a: &mut App, now: f64, step: u8) {
    let mut c = Create::new(now);
    c.name = "my-wallet".into();
    c.password = "Correct-Horse-Battery-9".into();
    c.confirm = c.password.clone();
    c.step = step;
    c.t_step = now;
    c.secure_t0 = now;
    if step >= 2 {
        c.phrase = Some(dd::creation_mnemonic());
    }
    a.screen = Screen::Create(Box::new(c));
}

fn demo_session(a: &mut App, now: f64, page: Page) {
    a.open_session(dd::entry("main", 7));
    a.cmds.clear();
    if let Some(s) = a.session_mut() {
        s.page = page;
        s.shown_page = page;
        s.page_t0 = now;
        dd::fill_session(s);
        s.modal = None;
        s.modal_was_open = false;
    }
}

pub struct Recorder {
    dir: PathBuf,
    list: Vec<Clip>,
    scale: f32,
    clip: usize,
    /// Frame index of the current clip that the next pass draws (`None` while the window is being sized).
    frame: Option<u32>,
    /// Pictures asked and saved, and passes since the last frame was asked.
    expected: u32,
    saved: u32,
    tail: u32,
    /// The virtual clock.
    vt: f64,
    clip_t0: f64,
    warm: u32,
    tries: u32,
}

impl Recorder {
    pub fn from_env() -> Option<Recorder> {
        let dir = PathBuf::from(std::env::var_os("VEARL_GUI_CLIPS")?);
        let only = std::env::var("VEARL_GUI_CLIP").ok();
        let list: Vec<Clip> = clips().into_iter().filter(|c| only.as_deref().map_or(true, |o| o.split(',').any(|x| x == c.name))).collect();
        let scale = std::env::var("VEARL_GUI_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0);
        RECORDING.store(true, Ordering::Relaxed);
        Some(Recorder { dir, list, scale, clip: 0, frame: None, expected: 0, saved: 0, tail: 0, vt: 1000.0, clip_t0: 1000.0, warm: 0, tries: 0 })
    }

    /// Before each pass: the virtual time and the scripted pointer replace the real ones.
    pub fn hook(&mut self, raw: &mut egui::RawInput) {
        if let Some(f) = self.frame {
            self.vt = self.clip_t0 + f as f64 / FPS;
        } else {
            self.vt += 1.0 / FPS;
        }
        raw.time = Some(self.vt);
        raw.predicted_dt = (1.0 / FPS) as f32;
        raw.events.retain(|e| !matches!(e, egui::Event::PointerMoved(_) | egui::Event::PointerButton { .. } | egui::Event::MouseWheel { .. } | egui::Event::PointerGone | egui::Event::Key { .. } | egui::Event::Text(_)));
        let pos = match (self.frame, self.list.get(self.clip)) {
            (Some(f), Some(c)) => (c.pointer)(f as f64 / FPS),
            _ => None,
        };
        match pos {
            Some(p) => raw.events.push(egui::Event::PointerMoved(p)),
            None => raw.events.push(egui::Event::PointerGone),
        }
    }

    /// During each pass: saves the picture of the last frame, moves the clip on, asks for the picture of this one.
    pub fn drive(&mut self, app: &mut App, ctx: &egui::Context) {
        ctx.request_repaint();
        // every picture carries the clip and the frame it was asked for
        let shots: Vec<((usize, u32), std::sync::Arc<egui::ColorImage>)> = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|e| match e {
                    egui::Event::Screenshot { image, user_data, .. } => user_data.data.as_ref().and_then(|d| d.downcast_ref::<(usize, u32)>()).map(|k| (*k, image.clone())),
                    _ => None,
                })
                .collect()
        });
        for ((c, f), img) in shots {
            save(&self.dir.join(self.list[c].name).join(format!("f{f:04}.png")), &img);
            self.saved += 1;
        }
        if self.clip >= self.list.len() {
            self.tail += 1;
            if self.saved >= self.expected || self.tail > 30 {
                if self.saved < self.expected {
                    println!("missing pictures: {} of {}", self.expected - self.saved, self.expected);
                }
                RECORDING.store(false, Ordering::Relaxed);
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            return;
        }
        // size the window: the logical size of the real window, drawn at `scale` pixels per point
        if self.frame.is_none() {
            ctx.set_zoom_factor(self.scale);
            let native = ctx.native_pixels_per_point().unwrap_or(1.0);
            ctx.send_viewport_cmd(egui::ViewportCommand::MaxInnerSize(egui::vec2(8000.0, 6000.0)));
            // the size is given in points, which already include the zoom
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(LOGICAL.0, LOGICAL.1)));
            let r = ctx.content_rect();
            let ok = (r.width() - LOGICAL.0).abs() < 1.0 && (r.height() - LOGICAL.1).abs() < 1.0;
            self.warm = if ok { self.warm + 1 } else { 0 };
            self.tries += 1;
            if self.tries % 60 == 0 {
                println!("sizing: content {:?}, native ppp {native}, ppp {}", r.size(), ctx.pixels_per_point());
            }
            if self.warm < 20 {
                return;
            }
            let _ = std::fs::create_dir_all(self.dir.join(self.list[self.clip].name));
            self.clip_t0 = clip_start(self.clip);
            self.frame = Some(0);
            return;
        }
        let f = self.frame.unwrap_or(0);
        let c = &self.list[self.clip];
        let t = f as f64 / FPS;
        if f == 0 {
            app.settings.protect_capture = false;
            app.settings.reduce_motion = false;
            (c.setup)(app, self.vt);
        }
        (c.step)(app, t, self.vt);
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new((self.clip, f))));
        self.expected += 1;
        let total = (c.secs * FPS).round() as u32;
        if f + 1 >= total {
            println!("clip {} : {} frames", c.name, total);
            self.clip += 1;
            if self.clip < self.list.len() {
                let _ = std::fs::create_dir_all(self.dir.join(self.list[self.clip].name));
                self.clip_t0 = clip_start(self.clip);
                self.frame = Some(0);
            } else {
                self.frame = Some(f + 1);
            }
        } else {
            self.frame = Some(f + 1);
        }
    }
}

/// The virtual time of the first frame of the clip number `k`: fixed, so that the effects that depend on the absolute time (the light
/// that circles the sphere) can be matched by the video.
pub fn clip_start(k: usize) -> f64 {
    2000.0 + 100.0 * k as f64
}

fn save(path: &std::path::Path, img: &egui::ColorImage) {
    let (w, h) = (img.size[0] as u32, img.size[1] as u32);
    let mut raw = Vec::with_capacity((w * h * 3) as usize);
    for p in &img.pixels {
        raw.extend_from_slice(&[p.r(), p.g(), p.b()]);
    }
    if let Ok(f) = std::fs::File::create(path) {
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(png::Compression::Fast);
        if let Ok(mut wr) = enc.write_header() {
            let _ = wr.write_image_data(&raw);
        }
    }
}
