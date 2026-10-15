//! State of the window and everything that happens in the background. Drawing is in `ui_start.rs` and `ui_wallet.rs`: they only change
//! the state and push [`Cmd`]s, which are carried out here (threads, files, settings), so no drawing code ever blocks or touches the disk.
use crate::anim;
use crate::core::*;
use crate::demodata;
use crate::theme::Tone;
use eframe::egui;
use rand::RngCore;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};
use vearl_core::crypto::PubKey;
use vearl_core::wallet::{derive_pq_account, parse_address, parse_amount, parse_mnemonic, pq_address_of};
use zeroize::Zeroize;

pub const REFRESH_EVERY: Duration = Duration::from_secs(15);
/// Mouse movements gathered as extra randomness when a wallet is created.
pub const ENTROPY_SAMPLES: u32 = 160;
/// The recovery words hide themselves again after this many seconds on screen.
pub const WORDS_VISIBLE_SECS: f64 = 90.0;
/// A new password must have at least this many characters (the wallet file format accepts 8).
pub const MIN_PASSWORD: usize = 10;

/// Whether the "Demo data" tag is drawn: always in demo mode, except while the clip recorder of the design tool runs (the video
/// says it in words).
pub fn show_demo_chip(demo: bool) -> bool {
    #[cfg(feature = "devshots")]
    {
        demo && !crate::record::RECORDING.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "devshots"))]
    {
        demo
    }
}

// ------------------------------------------------------------------------------------------------ screens

pub enum Screen {
    Start(StartState),
    Create(Box<Create>),
    Restore(Box<Restore>),
    Wallet(Box<Session>),
}

pub struct StartState {
    /// When the screen appeared (for the opening animation).
    pub t0: f64,
    pub show_open: bool,
    pub open_path: String,
    pub error: Option<String>,
    pub notice: Option<(Tone, String)>,
}

impl StartState {
    pub fn new(t0: f64, notice: Option<(Tone, String)>) -> StartState {
        StartState { t0, show_open: false, open_path: String::new(), error: None, notice }
    }
}

/// Randomness for a new wallet: the system generator (always) plus the movements of the pointer, hashed together. The pointer can only
/// add: whatever it does, the result is at least as unpredictable as the system generator alone.
pub struct Entropy {
    hasher: blake3::Hasher,
    pub samples: u32,
    pub last: Option<egui::Pos2>,
    /// (position, time, hue) of the dots drawn along the path of the pointer.
    pub dots: Vec<(egui::Pos2, f64, f32)>,
}

impl Entropy {
    pub fn new() -> Entropy {
        let mut os = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut os);
        let mut hasher = blake3::Hasher::new_derive_key("vearl wallet entropy v1");
        hasher.update(&os);
        os.zeroize();
        Entropy { hasher, samples: 0, last: None, dots: vec![] }
    }

    /// One movement of the pointer; counted only if it moved far enough from the last one.
    pub fn add(&mut self, pos: egui::Pos2, now: f64) -> bool {
        if let Some(l) = self.last {
            if (pos - l).length() < 14.0 {
                return false;
            }
        }
        self.last = Some(pos);
        self.hasher.update(&pos.x.to_le_bytes());
        self.hasher.update(&pos.y.to_le_bytes());
        self.hasher.update(&now.to_le_bytes());
        self.hasher.update(&(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())).to_le_bytes());
        self.samples += 1;
        self.dots.push((pos, now, anim::noise(self.samples as u64, 77, 3)));
        if self.dots.len() > 90 {
            self.dots.remove(0);
        }
        true
    }

    pub fn progress(&self) -> f32 {
        (self.samples as f32 / ENTROPY_SAMPLES as f32).min(1.0)
    }

    /// The 24 words: 256 bits drawn from everything gathered, plus fresh system randomness.
    pub fn finish(&self) -> bip39::Mnemonic {
        let mut h = self.hasher.clone();
        let mut fresh = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut fresh);
        h.update(&fresh);
        fresh.zeroize();
        let mut out = [0u8; 32];
        h.finalize_xof().fill(&mut out);
        let m = bip39::Mnemonic::from_entropy_in(bip39::Language::English, &out).expect("32 bytes make 24 words");
        out.zeroize();
        m
    }
}

pub struct Create {
    /// 0 name and password, 1 randomness, 2 write the words down, 3 check them, 4 securing the file, 5 ready
    pub step: u8,
    /// When the current step started (for its animation).
    pub t_step: f64,
    pub name: String,
    pub password: String,
    pub confirm: String,
    pub entropy: Entropy,
    pub phrase: Option<bip39::Mnemonic>,
    /// When the words were revealed (they hide again by themselves).
    pub reveal_t0: Option<f64>,
    pub written: bool,
    pub check_idx: [usize; 3],
    pub check_in: [String; 3],
    pub error: Option<String>,
    pub secure_t0: f64,
    pub result: Option<Result<WalletEntry, String>>,
    pub done: Option<WalletEntry>,
}

impl Create {
    pub fn new(now: f64) -> Create {
        Create {
            step: 0,
            t_step: now,
            name: "my-wallet".into(),
            password: String::new(),
            confirm: String::new(),
            entropy: Entropy::new(),
            phrase: None,
            reveal_t0: None,
            written: false,
            check_idx: [0, 8, 16],
            check_in: Default::default(),
            error: None,
            secure_t0: 0.0,
            result: None,
            done: None,
        }
    }
    pub fn words(&self) -> Vec<String> {
        self.phrase.as_ref().map(|m| m.words().map(str::to_string).collect()).unwrap_or_default()
    }
}

impl Drop for Create {
    fn drop(&mut self) {
        self.password.zeroize();
        self.confirm.zeroize();
        for w in self.check_in.iter_mut() {
            w.zeroize();
        }
    }
}

pub struct Restore {
    /// 0 form, 1 securing the file, 2 ready
    pub step: u8,
    pub t_step: f64,
    pub name: String,
    pub phrase: String,
    pub password: String,
    pub confirm: String,
    pub error: Option<String>,
    pub secure_t0: f64,
    pub result: Option<Result<WalletEntry, String>>,
    pub done: Option<WalletEntry>,
}

impl Restore {
    pub fn new(now: f64) -> Restore {
        Restore { step: 0, t_step: now, name: "restored-wallet".into(), phrase: String::new(), password: String::new(), confirm: String::new(), error: None, secure_t0: 0.0, result: None, done: None }
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        self.phrase.zeroize();
        self.password.zeroize();
        self.confirm.zeroize();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Page {
    Overview,
    Send,
    Receive,
    Activity,
    Rewards,
    Tools,
    Settings,
}

impl Page {
    pub const ALL: [(Page, &'static str, &'static str); 7] = [
        (Page::Overview, "01", "Overview"),
        (Page::Send, "02", "Send"),
        (Page::Receive, "03", "Receive"),
        (Page::Activity, "04", "Activity"),
        (Page::Rewards, "05", "Rewards"),
        (Page::Tools, "06", "Tools"),
        (Page::Settings, "07", "Settings"),
    ];
}

#[derive(Default)]
pub struct SendForm {
    pub to: String,
    pub amount: String,
    pub fee: String,
    pub pq: bool,
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct RegLines {
    pub address: String,
    pub message: String,
    pub signature: String,
}

#[derive(Default)]
pub struct RewardsForm {
    pub contact: String,
    pub lines: Option<RegLines>,
    pub error: Option<String>,
    pub sending: bool,
    pub sent: Option<Result<(), String>>,
}

#[derive(Default)]
pub struct ToolsForm {
    pub message: String,
    pub signed: Option<RegLines>,
    pub v_address: String,
    pub v_message: String,
    pub v_signature: String,
    pub verdict: Option<Result<bool, String>>,
}

#[derive(Default)]
pub struct SettingsForm {
    pub node: String,
    pub explorer: String,
    pub node_test: Option<Result<NodeStatus, String>>,
    pub testing: bool,
    pub new_password: String,
    pub new_confirm: String,
    pub pw_error: Option<String>,
    pub saved: bool,
}

impl Drop for SettingsForm {
    fn drop(&mut self) {
        self.new_password.zeroize();
        self.new_confirm.zeroize();
    }
}

/// What a password is asked for.
#[derive(Clone)]
pub enum Action {
    RevealPq,
    ShowPhrase,
    Send { to: String, amount: u64, fee: u64, pq: bool },
    Register { contact: String },
    Sign { message: String },
    ChangePassword { new: String },
}

impl Action {
    /// A secret travels with the action (the new password): wipe it when the action is done with.
    pub fn wipe(&mut self) {
        if let Action::ChangePassword { new } = self {
            new.zeroize();
        }
    }
}

pub struct PwPrompt {
    pub action: Action,
    pub password: String,
    pub error: Option<String>,
    pub busy: bool,
    /// Wrong passwords in a row, and when the next try is allowed (it slows down after three).
    pub failures: u32,
    pub locked_until: f64,
}

impl PwPrompt {
    pub fn new(action: Action) -> PwPrompt {
        PwPrompt { action, password: String::new(), error: None, busy: false, failures: 0, locked_until: 0.0 }
    }
}

impl Drop for PwPrompt {
    fn drop(&mut self) {
        self.password.zeroize();
        self.action.wipe();
    }
}

pub enum Modal {
    Password(PwPrompt),
    Phrase(Vec<String>),
    Sent { txid: String, from: String, to: String, amount: u64, fee: u64 },
}

impl Drop for Modal {
    fn drop(&mut self) {
        if let Modal::Phrase(words) = self {
            for w in words.iter_mut() {
                w.zeroize();
            }
        }
    }
}

pub struct Session {
    pub sid: u64,
    pub entry: WalletEntry,
    pub pk: PubKey,
    pub page: Page,
    /// The page whose opening animation is running, and when it started.
    pub shown_page: Page,
    pub page_t0: f64,
    pub status: Option<Result<NodeStatus, String>>,
    pub account: Option<Account>,
    pub account_error: Option<String>,
    pub pq_address: Option<String>,
    pub pq_account: Option<Account>,
    pub activity: Option<Result<Vec<TxRow>, String>>,
    pub registered: Option<Result<bool, String>>,
    pub standing: Option<Result<Option<Standing>, String>>,
    pub last_refresh: Option<Instant>,
    pub refreshing: bool,
    pub refresh_started: Option<Instant>,
    pub send: SendForm,
    pub rewards: RewardsForm,
    pub tools: ToolsForm,
    pub settings: SettingsForm,
    pub modal: Option<Modal>,
    pub modal_t0: f64,
    pub modal_was_open: bool,
    pub toast: Option<(Tone, String, Instant)>,
}

impl Session {
    /// Addresses this wallet has already paid (from the activity list): a payment to anyone else is flagged as a new recipient.
    pub fn known_recipient(&self, address: &str) -> bool {
        match &self.activity {
            Some(Ok(rows)) => rows.iter().any(|r| r.direction == "out" && r.counterparty.as_deref() == Some(address)),
            _ => false,
        }
    }
}

// ------------------------------------------------------------------------------------------------ messages

pub enum Done {
    Pq(String),
    Phrase(Vec<String>),
    Sent { txid: String, from: String, to: String, amount: u64, fee: u64 },
    Signed(RegLines),
    Registration(RegLines),
    PasswordChanged,
}

pub enum Ev {
    Refresh { sid: u64, status: Result<NodeStatus, String>, account: Option<Result<Account, String>>, pq: Option<Result<Account, String>> },
    Activity { sid: u64, res: Result<Vec<TxRow>, String> },
    Registered { sid: u64, res: Result<bool, String> },
    Standing { sid: u64, res: Result<Option<Standing>, String> },
    Created(Result<WalletEntry, String>),
    Action { sid: u64, res: Result<Done, String> },
    Mail { sid: u64, res: Result<(), String> },
    NodeTest { sid: u64, res: Result<NodeStatus, String> },
}

pub enum Cmd {
    OpenWallet(WalletEntry),
    OpenPath(String),
    GoStart,
    /// Back to the start screen because nothing was touched for a while.
    LockIdle,
    StartCreate,
    StartRestore,
    CreateNext,
    CreateBack,
    CreateFinish,
    RestoreFinish,
    Refresh,
    RefreshActivity,
    CheckRegistered,
    Ask(Action),
    SubmitPassword,
    CancelModal,
    SendMail,
    SaveSettings,
    /// The program's own settings (lock delay, motion...) changed.
    PersistSettings,
    TestNode,
    Link(String),
    Reveal(PathBuf),
    Forget(String),
}

/// Removes what the text fields remember (their undo history keeps copies of what was typed, passwords included).
pub fn wipe_text_state(ctx: &egui::Context) {
    ctx.data_mut(|d| d.remove_by_type::<egui::text_edit::TextEditState>());
}

// ------------------------------------------------------------------------------------------------ the application

pub struct App {
    pub ctx: egui::Context,
    pub logo: egui::TextureHandle,
    pub settings: Settings,
    pub network: Network,
    pub wallets: Vec<WalletEntry>,
    pub screen: Screen,
    pub demo: bool,
    pub cmds: Vec<Cmd>,
    pub last_activity: Instant,
    tx: Sender<Ev>,
    rx: Receiver<Ev>,
    next_sid: u64,
    #[cfg(feature = "devshots")]
    pub shots: Option<crate::demo::Shots>,
}

impl App {
    pub fn new(ctx: egui::Context, logo: egui::TextureHandle, demo: bool) -> App {
        let settings = Settings::load();
        let network = load_network();
        let wallets = if demo { demodata::entries() } else { list_wallets(&settings) };
        let (tx, rx) = channel();
        let now = anim::time(&ctx);
        anim::set_reduce_motion(settings.reduce_motion);
        App {
            ctx,
            logo,
            settings,
            network,
            wallets,
            screen: Screen::Start(StartState::new(now, None)),
            demo,
            cmds: vec![],
            last_activity: Instant::now(),
            tx,
            rx,
            next_sid: 1,
            #[cfg(feature = "devshots")]
            shots: None,
        }
    }

    pub fn now(&self) -> f64 {
        anim::time(&self.ctx)
    }

    pub fn nodes(&self) -> Vec<String> {
        node_list(&self.settings, &self.network)
    }

    pub fn explorer(&self) -> String {
        let s = self.settings.explorer.trim();
        if s.is_empty() {
            self.network.explorer.clone().unwrap_or_else(|| DEFAULT_EXPLORER.into())
        } else {
            s.to_string()
        }
    }

    fn spawn(&self, f: impl FnOnce() -> Ev + Send + 'static) {
        let (tx, ctx) = (self.tx.clone(), self.ctx.clone());
        std::thread::spawn(move || {
            let ev = f();
            let _ = tx.send(ev);
            ctx.request_repaint();
        });
    }

    pub fn session_mut(&mut self) -> Option<&mut Session> {
        match &mut self.screen {
            Screen::Wallet(s) => Some(s),
            _ => None,
        }
    }

    /// True while the 24 recovery words are on screen (the window can then hide itself from screen capture).
    pub fn words_on_screen(&self) -> bool {
        match &self.screen {
            Screen::Create(c) => c.step == 2 && c.reveal_t0.is_some(),
            Screen::Wallet(s) => matches!(s.modal, Some(Modal::Phrase(_))),
            _ => false,
        }
    }

    pub fn open_session(&mut self, entry: WalletEntry) {
        let Ok(pk) = parse_address(&entry.address) else { return };
        let sid = self.next_sid;
        self.next_sid += 1;
        if !self.demo {
            self.settings.last_wallet = entry.path.display().to_string();
            self.settings.save();
        }
        let now = self.now();
        let mut s = Session {
            sid,
            entry,
            pk,
            page: Page::Overview,
            shown_page: Page::Overview,
            page_t0: now,
            status: None,
            account: None,
            account_error: None,
            pq_address: None,
            pq_account: None,
            activity: None,
            registered: None,
            standing: None,
            last_refresh: None,
            refreshing: false,
            refresh_started: None,
            send: SendForm { fee: FEE_DEFAULT.into(), ..Default::default() },
            rewards: RewardsForm::default(),
            tools: ToolsForm::default(),
            settings: {
                let mut f = SettingsForm::default();
                f.node = self.settings.node.clone();
                f.explorer = self.settings.explorer.clone();
                f
            },
            modal: None,
            modal_t0: now,
            modal_was_open: false,
            toast: None,
        };
        if self.demo {
            demodata::fill_session(&mut s);
        }
        self.screen = Screen::Wallet(Box::new(s));
        wipe_text_state(&self.ctx);
        self.last_activity = Instant::now();
        self.cmds.push(Cmd::Refresh);
        self.cmds.push(Cmd::RefreshActivity);
    }

    /// Called every frame: idle lock, and the settings that act on the whole program.
    pub fn tick(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| !i.events.is_empty() || i.pointer.delta() != egui::Vec2::ZERO) {
            self.last_activity = Instant::now();
        }
        anim::set_reduce_motion(self.settings.reduce_motion);
        let mins = self.settings.autolock_minutes as u64;
        if mins > 0 && matches!(self.screen, Screen::Wallet(_)) && !self.demo {
            let idle = self.last_activity.elapsed();
            if idle >= Duration::from_secs(mins * 60) {
                self.cmds.push(Cmd::LockIdle);
            } else {
                ctx.request_repaint_after(Duration::from_secs(mins * 60) - idle + Duration::from_millis(50));
            }
        }
    }

    // -------------------------------------------------------------------------------------------- events from the threads

    pub fn poll(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            self.handle(ev);
        }
    }

    fn handle(&mut self, ev: Ev) {
        let now = self.now();
        match ev {
            Ev::Created(res) => {
                match &mut self.screen {
                    Screen::Create(c) => c.result = Some(res),
                    Screen::Restore(r) => r.result = Some(res),
                    _ => {}
                }
                self.wallets = if self.demo { demodata::entries() } else { list_wallets(&self.settings) };
            }
            Ev::Refresh { sid, status, account, pq } => {
                if let Some(s) = self.session_mut().filter(|s| s.sid == sid) {
                    s.refreshing = false;
                    s.last_refresh = Some(Instant::now());
                    match account {
                        Some(Ok(a)) => {
                            s.account = Some(a);
                            s.account_error = None;
                        }
                        Some(Err(e)) => s.account_error = Some(e),
                        None => {}
                    }
                    if let Some(Ok(a)) = pq {
                        s.pq_account = Some(a);
                    }
                    s.status = Some(status);
                }
            }
            Ev::Activity { sid, res } => {
                if let Some(s) = self.session_mut().filter(|s| s.sid == sid) {
                    s.activity = Some(res);
                }
            }
            Ev::Registered { sid, res } => {
                if let Some(s) = self.session_mut().filter(|s| s.sid == sid) {
                    s.registered = Some(res);
                }
            }
            Ev::Standing { sid, res } => {
                if let Some(s) = self.session_mut().filter(|s| s.sid == sid) {
                    s.standing = Some(res);
                }
            }
            Ev::NodeTest { sid, res } => {
                if let Some(s) = self.session_mut().filter(|s| s.sid == sid) {
                    s.settings.testing = false;
                    s.settings.node_test = Some(res);
                }
            }
            Ev::Mail { sid, res } => {
                if let Some(s) = self.session_mut().filter(|s| s.sid == sid) {
                    s.rewards.sending = false;
                    s.rewards.sent = Some(res);
                }
                self.cmds.push(Cmd::CheckRegistered);
            }
            Ev::Action { sid, res } => {
                let Some(s) = self.session_mut().filter(|s| s.sid == sid) else { return };
                match res {
                    Err(e) => {
                        if let Some(Modal::Password(p)) = &mut s.modal {
                            p.busy = false;
                            if e.starts_with("wrong password") {
                                p.failures += 1;
                                if p.failures >= 3 {
                                    p.locked_until = now + (2u64.pow((p.failures - 3).min(4)) as f64) * 2.0;
                                }
                            }
                            p.error = Some(e);
                            p.password.zeroize();
                        }
                    }
                    Ok(done) => {
                        s.modal = None;
                        match done {
                            Done::Pq(a) => {
                                s.pq_address = Some(a);
                                s.toast = Some((Tone::Ok, "Post-quantum address unlocked.".into(), Instant::now()));
                                self.cmds.push(Cmd::Refresh);
                            }
                            Done::Phrase(w) => s.modal = Some(Modal::Phrase(w)),
                            Done::Sent { txid, from, to, amount, fee } => {
                                s.send = SendForm { fee: FEE_DEFAULT.into(), ..Default::default() };
                                s.modal = Some(Modal::Sent { txid, from, to, amount, fee });
                                self.cmds.push(Cmd::Refresh);
                                self.cmds.push(Cmd::RefreshActivity);
                            }
                            Done::Signed(l) => s.tools.signed = Some(l),
                            Done::Registration(l) => {
                                s.rewards.lines = Some(l);
                                s.rewards.sent = None;
                                s.rewards.error = None;
                            }
                            Done::PasswordChanged => {
                                s.settings.new_password.zeroize();
                                s.settings.new_confirm.zeroize();
                                s.settings.pw_error = None;
                                s.toast = Some((Tone::Ok, "Password changed. The wallet file was re-encrypted.".into(), Instant::now()));
                            }
                        }
                        wipe_text_state(&self.ctx);
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------------------------- commands from the drawing code

    pub fn run_cmds(&mut self) {
        let cmds = std::mem::take(&mut self.cmds);
        for c in cmds {
            self.run(c);
        }
    }

    fn go_start(&mut self, notice: Option<(Tone, String)>) {
        self.wallets = if self.demo { demodata::entries() } else { list_wallets(&self.settings) };
        let now = self.now();
        self.screen = Screen::Start(StartState::new(now, notice));
        wipe_text_state(&self.ctx);
    }

    fn run(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::OpenWallet(e) => self.open_session(e),
            Cmd::OpenPath(p) => {
                let path = PathBuf::from(p.trim().trim_matches('"'));
                match entry_of(&path) {
                    Ok(entry) => {
                        let s = path.display().to_string();
                        if !self.settings.extra_wallets.contains(&s) && !self.wallets.iter().any(|w| w.path == path) {
                            self.settings.extra_wallets.push(s);
                            self.settings.save();
                        }
                        self.wallets = list_wallets(&self.settings);
                        self.open_session(entry);
                    }
                    Err(e) => {
                        if let Screen::Start(st) = &mut self.screen {
                            st.error = Some(e);
                        }
                    }
                }
            }
            Cmd::Forget(p) => {
                self.settings.extra_wallets.retain(|x| *x != p);
                self.settings.save();
                self.wallets = list_wallets(&self.settings);
            }
            Cmd::GoStart => self.go_start(None),
            Cmd::LockIdle => {
                let m = self.settings.autolock_minutes;
                self.go_start(Some((Tone::Info, format!("Locked after {m} minute{} without activity.", if m == 1 { "" } else { "s" }))));
            }
            Cmd::StartCreate => {
                let now = self.now();
                self.screen = Screen::Create(Box::new(Create::new(now)));
            }
            Cmd::StartRestore => {
                let now = self.now();
                self.screen = Screen::Restore(Box::new(Restore::new(now)));
            }
            Cmd::CreateNext => self.create_next(),
            Cmd::CreateBack => {
                let now = self.now();
                if let Screen::Create(c) = &mut self.screen {
                    c.error = None;
                    match c.step {
                        1 => {
                            c.step = 0;
                            c.t_step = now;
                        }
                        3 => {
                            c.step = 2;
                            c.t_step = now;
                            c.reveal_t0 = None;
                        }
                        4 => {
                            c.step = 3;
                            c.t_step = now;
                            c.result = None;
                        }
                        _ => {}
                    }
                }
            }
            Cmd::CreateFinish => self.create_finish(),
            Cmd::RestoreFinish => self.restore_finish(),
            Cmd::Refresh => self.refresh(),
            Cmd::RefreshActivity => self.refresh_activity(),
            Cmd::CheckRegistered => {
                if self.demo {
                    if let Some(s) = self.session_mut() {
                        s.registered = Some(Ok(false));
                        s.standing = Some(Ok(Some(demodata::standing())));
                    }
                    return;
                }
                if let Some((sid, addr)) = self.session_mut().map(|s| (s.sid, s.entry.address.clone())) {
                    let addr2 = addr.clone();
                    self.spawn(move || Ev::Registered { sid, res: is_registered(&addr) });
                    self.spawn(move || Ev::Standing { sid, res: fetch_standing(&addr2) });
                }
            }
            Cmd::Ask(action) => {
                let now = self.now();
                if let Some(s) = self.session_mut() {
                    s.modal = Some(Modal::Password(PwPrompt::new(action)));
                    s.modal_t0 = now;
                }
            }
            Cmd::CancelModal => {
                if let Some(s) = self.session_mut() {
                    s.modal = None;
                }
                wipe_text_state(&self.ctx);
            }
            Cmd::SubmitPassword => self.submit_password(),
            Cmd::SendMail => self.send_mail(),
            Cmd::SaveSettings => {
                if let Some(s) = self.session_mut() {
                    let (n, e) = (s.settings.node.trim().to_string(), s.settings.explorer.trim().to_string());
                    s.settings.saved = true;
                    self.settings.node = n;
                    self.settings.explorer = if e.is_empty() { DEFAULT_EXPLORER.into() } else { e };
                    self.settings.save();
                    self.cmds.push(Cmd::Refresh);
                    self.cmds.push(Cmd::RefreshActivity);
                }
            }
            Cmd::PersistSettings => {
                if !self.demo {
                    self.settings.save();
                }
            }
            Cmd::TestNode => {
                let demo = self.demo;
                let nodes: Vec<String> = match self.session_mut() {
                    Some(s) => {
                        s.settings.testing = true;
                        s.settings.node_test = None;
                        let own = s.settings.node.trim().to_string();
                        if own.is_empty() {
                            vec![]
                        } else {
                            vec![own]
                        }
                    }
                    None => return,
                };
                let nodes = if nodes.is_empty() { self.nodes() } else { nodes };
                let sid = self.session_mut().map_or(0, |s| s.sid);
                self.spawn(move || {
                    if demo {
                        std::thread::sleep(Duration::from_millis(900));
                        return Ev::NodeTest { sid, res: Ok(demodata::status()) };
                    }
                    Ev::NodeTest { sid, res: node_status(&nodes) }
                });
            }
            Cmd::Link(u) => {
                if !self.demo {
                    open_link(&u)
                }
            }
            Cmd::Reveal(p) => reveal(&p),
        }
    }

    fn refresh(&mut self) {
        if self.demo {
            return;
        }
        let nodes = self.nodes();
        let Some(s) = self.session_mut() else { return };
        // a request that never came back (a node that accepts and stays silent) must not block the refresh for ever
        if s.refreshing && s.refresh_started.is_some_and(|t| t.elapsed() < Duration::from_secs(60)) {
            return;
        }
        s.refreshing = true;
        s.refresh_started = Some(Instant::now());
        let (sid, pk) = (s.sid, s.pk);
        let pq_pk: Option<PubKey> = s.pq_address.as_ref().and_then(|a| parse_address(a).ok());
        self.spawn(move || {
            let status = node_status(&nodes);
            let (account, pq) = match &status {
                Ok(st) => (Some(account_of(&st.node, pk)), pq_pk.map(|p| account_of(&st.node, p))),
                Err(_) => (None, None),
            };
            Ev::Refresh { sid, status, account, pq }
        });
    }

    fn refresh_activity(&mut self) {
        if self.demo {
            return;
        }
        let explorer = self.explorer();
        let Some(s) = self.session_mut() else { return };
        let (sid, addr) = (s.sid, s.entry.address.clone());
        self.spawn(move || Ev::Activity { sid, res: fetch_activity(&explorer, &addr) });
    }

    fn submit_password(&mut self) {
        let nodes = self.nodes();
        let demo = self.demo;
        let now = self.now();
        let Some(s) = self.session_mut() else { return };
        let (sid, path) = (s.sid, s.entry.path.clone());
        let Some(Modal::Password(p)) = &mut s.modal else { return };
        if p.busy {
            return;
        }
        if now < p.locked_until {
            return;
        }
        if p.password.is_empty() {
            p.error = Some("Type the password of this wallet.".into());
            return;
        }
        p.busy = true;
        p.error = None;
        let password = std::mem::take(&mut p.password);
        let action = p.action.clone();
        self.spawn(move || {
            let mut password = password;
            let mut action = action;
            let res = if demo {
                std::thread::sleep(Duration::from_millis(1300));
                Ok(demodata::done_for(&action))
            } else {
                run_action(&path, &password, action.clone(), &nodes)
            };
            password.zeroize();
            action.wipe();
            Ev::Action { sid, res }
        });
    }

    fn send_mail(&mut self) {
        let demo = self.demo;
        let Some(s) = self.session_mut() else { return };
        let Some(l) = s.rewards.lines.clone() else { return };
        if s.rewards.sending {
            return;
        }
        s.rewards.sending = true;
        s.rewards.sent = None;
        let sid = s.sid;
        self.spawn(move || {
            if demo {
                std::thread::sleep(Duration::from_millis(1500));
                return Ev::Mail { sid, res: Ok(()) };
            }
            let reg = vearl_core::regmail::Registration { address: l.address, message: l.message, signature_hex: l.signature };
            Ev::Mail { sid, res: vearl_core::regmail::send_registration(&reg) }
        });
    }

    // -------------------------------------------------------------------------------------------- creating and restoring

    fn create_next(&mut self) {
        let now = self.now();
        let demo = self.demo;
        let Screen::Create(c) = &mut self.screen else { return };
        c.error = None;
        match c.step {
            0 => {
                if let Err(e) = wallet_path_for(&c.name).and_then(|p| if p.exists() { Err(format!("{} already exists: choose another name", p.display())) } else { Ok(()) }) {
                    c.error = Some(e);
                    return;
                }
                if let Err(e) = check_new_password(&c.password, &c.name) {
                    c.error = Some(e);
                    return;
                }
                if c.password != c.confirm {
                    c.error = Some("The two passwords are not the same.".into());
                    return;
                }
                c.step = 1;
                c.t_step = now;
            }
            1 => {
                // the words are drawn now, from everything gathered
                let m = if demo { demodata::creation_mnemonic() } else { c.entropy.finish() };
                let e = m.to_entropy();
                c.check_idx = [(e[0] % 8) as usize, 8 + (e[1] % 8) as usize, 16 + (e[2] % 8) as usize];
                c.phrase = Some(m);
                c.entropy = Entropy::new();
                c.reveal_t0 = None;
                c.written = false;
                c.step = 2;
                c.t_step = now;
            }
            2 => {
                if c.reveal_t0.is_none() {
                    c.error = Some("Show the 24 words first, and write them down.".into());
                    return;
                }
                if !c.written {
                    c.error = Some("Tick the box once the 24 words are written on paper.".into());
                    return;
                }
                c.check_in = Default::default();
                c.reveal_t0 = None;
                c.step = 3;
                c.t_step = now;
            }
            _ => {}
        }
    }

    fn create_finish(&mut self) {
        let now = self.now();
        let Screen::Create(c) = &mut self.screen else { return };
        let words = c.words();
        for (k, &i) in c.check_idx.iter().enumerate() {
            if c.check_in[k].trim().to_lowercase() != words.get(i).cloned().unwrap_or_default() {
                c.error = Some(format!("Word {} is not right. Look at your paper (or go back to see the words again).", i + 1));
                return;
            }
        }
        let (Some(m), Ok(path)) = (c.phrase.clone(), wallet_path_for(&c.name)) else { return };
        c.error = None;
        c.step = 4;
        c.t_step = now;
        c.secure_t0 = now;
        c.result = None;
        let pw = c.password.clone();
        let demo = self.demo;
        self.spawn(move || {
            let mut pw = pw;
            let r = if demo { std::thread::sleep(Duration::from_millis(600)); Ok(demodata::entry("my-wallet", 11)) } else { create_wallet(&path, &m, &pw) };
            pw.zeroize();
            Ev::Created(r)
        });
    }

    fn restore_finish(&mut self) {
        let now = self.now();
        let Screen::Restore(r) = &mut self.screen else { return };
        r.error = None;
        let path = match wallet_path_for(&r.name) {
            Ok(p) if p.exists() => {
                r.error = Some(format!("{} already exists: choose another name", p.display()));
                return;
            }
            Ok(p) => p,
            Err(e) => {
                r.error = Some(e);
                return;
            }
        };
        let phrase = clean_phrase(&r.phrase);
        let m = match parse_mnemonic(&phrase) {
            Ok(m) => m,
            Err(_) if phrase.split_whitespace().count() != 24 => {
                r.error = Some(format!("A recovery phrase has 24 words; you typed {}.", phrase.split_whitespace().count()));
                return;
            }
            Err(_) => {
                r.error = Some("These 24 words are not a valid phrase: one word is wrong or two words are swapped.".into());
                return;
            }
        };
        if let Err(e) = check_new_password(&r.password, &r.name) {
            r.error = Some(e);
            return;
        }
        if r.password != r.confirm {
            r.error = Some("The two passwords are not the same.".into());
            return;
        }
        r.step = 1;
        r.t_step = now;
        r.secure_t0 = now;
        r.result = None;
        let pw = r.password.clone();
        let demo = self.demo;
        self.spawn(move || {
            let mut pw = pw;
            let res = if demo { std::thread::sleep(Duration::from_millis(600)); Ok(demodata::entry("restored-wallet", 5)) } else { create_wallet(&path, &m, &pw) };
            pw.zeroize();
            Ev::Created(res)
        });
    }
}

/// Runs in a thread: unlocks the wallet with the password, does what was asked, and drops the keys.
fn run_action(path: &std::path::Path, password: &str, action: Action, nodes: &[String]) -> Result<Done, String> {
    let m = unlock(path, password)?;
    match action {
        Action::RevealPq => Ok(Done::Pq(pq_address_of(&derive_pq_account(&m, 0).pubkey()))),
        Action::ShowPhrase => Ok(Done::Phrase(m.words().map(str::to_string).collect())),
        Action::Send { to, amount, fee, pq } => {
            let st = node_status(nodes)?;
            let (txid, from) = send_transfer(&st.node, &m, &to, amount, fee, pq)?;
            Ok(Done::Sent { txid, from, to, amount, fee })
        }
        Action::Register { contact } => {
            let r = sign_registration(&m, &contact)?;
            Ok(Done::Registration(RegLines { address: r.address, message: r.message, signature: r.signature_hex }))
        }
        Action::Sign { message } => {
            let (address, signature) = sign_text(&m, &message);
            Ok(Done::Signed(RegLines { address, message, signature }))
        }
        Action::ChangePassword { new } => {
            let mut new = new;
            let r = change_password(path, password, &new);
            new.zeroize();
            r.map(|_| Done::PasswordChanged)
        }
    }
}

/// The rules for a new password (the wallet file format itself accepts 8 characters; the window asks for more).
pub fn check_new_password(p: &str, wallet_name: &str) -> Result<(), String> {
    if p.chars().count() < MIN_PASSWORD {
        return Err(format!("The password needs at least {MIN_PASSWORD} characters."));
    }
    let lower = p.to_lowercase();
    if lower == wallet_name.trim().to_lowercase() {
        return Err("The password must not be the name of the wallet.".into());
    }
    if is_common_password(&lower) {
        return Err("This password is one of the most common ones: choose another.".into());
    }
    Ok(())
}

/// The passwords that every guessing attack tries first (and trivial variations).
pub fn is_common_password(lower: &str) -> bool {
    const COMMON: &[&str] = &[
        "password", "password1", "password12", "password123", "1234567890", "12345678910", "qwertyuiop", "qwertyuiopasdf", "azertyuiop", "iloveyou123", "letmein123",
        "administrator", "welcome123", "abcdefghij", "0123456789", "passw0rd123", "vearlvearl", "vearlwallet", "bitcoinbitcoin", "changeme123", "1q2w3e4r5t", "1qaz2wsx3edc",
    ];
    if COMMON.contains(&lower) {
        return true;
    }
    // one character repeated, or a short block repeated ("abababababab")
    let c: Vec<char> = lower.chars().collect();
    if c.iter().all(|x| *x == c[0]) {
        return true;
    }
    (1..=8).any(|n| c.len() % n == 0 && c.len() / n >= 2 && c.chunks(n).all(|ch| ch == &c[..n]))
}

/// The checks of the send form (shared by the window and the tests): the address, the amount and the fee as typed -> values.
pub fn validate_send(to: &str, amount: &str, fee: &str, balance: Option<u64>) -> Result<(String, u64, u64), String> {
    let to = to.trim();
    if to.is_empty() {
        return Err("Paste the address of the person you are paying (vrl1... or vrlq1...).".into());
    }
    parse_address(to)?;
    let a = parse_amount(amount.trim())?;
    if a == 0 {
        return Err("The amount must be more than zero.".into());
    }
    let f = if fee.trim().is_empty() { 0 } else { parse_amount(fee.trim())? };
    if let Some(b) = balance {
        if b < a.saturating_add(f) {
            return Err(format!("Not enough funds: {} {SYMBOL} available, {} {SYMBOL} needed with the fee.", vearl_core::wallet::fmt_amount(b), vearl_core::wallet::fmt_amount(a.saturating_add(f))));
        }
    }
    Ok((to.to_string(), a, f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vearl_core::crypto::pubkey;
    use vearl_core::wallet::{address_of, derive_account, new_mnemonic};

    #[test]
    fn send_form_checks() {
        let m = new_mnemonic();
        let addr = address_of(&pubkey(&derive_account(&m, 0)));
        assert!(validate_send("", "1", "0", None).is_err());
        assert!(validate_send("vrl1abc", "1", "0", None).is_err());
        assert!(validate_send(&addr, "0", "0", None).is_err());
        assert!(validate_send(&addr, "1.123456789", "0", None).is_err());
        assert!(validate_send(&addr, "1,5", "0", None).is_err());
        let ok = validate_send(&format!("  {addr} "), "12.5", "0.01", Some(20 * vearl_core::state::COIN)).unwrap();
        assert_eq!(ok, (addr.clone(), 1_250_000_000, 1_000_000));
        assert!(validate_send(&addr, "12.5", "", Some(12 * vearl_core::state::COIN)).is_err());
        assert_eq!(validate_send(&addr, "1", "", None).unwrap().2, 0);
    }

    #[test]
    fn password_rules() {
        assert!(check_new_password("short", "w").is_err());
        assert!(check_new_password("123456789", "w").is_err(), "nine characters");
        assert!(check_new_password("passwordpassword", "w").is_err(), "a short block repeated");
        assert!(check_new_password("aaaaaaaaaaaa", "w").is_err());
        assert!(check_new_password("my-wallet-name", "my-wallet-name").is_err());
        assert!(check_new_password("1234567890", "w").is_err());
        assert!(check_new_password("correct horse battery", "w").is_ok());
        assert!(check_new_password("Tr0ub4dor&3-xyz", "w").is_ok());
    }

    #[test]
    fn entropy_changes_with_the_pointer_and_never_repeats() {
        let a = Entropy::new();
        let b = Entropy::new();
        assert_ne!(a.finish().to_entropy(), b.finish().to_entropy(), "the system generator alone already differs");
        assert_ne!(a.finish().to_entropy(), a.finish().to_entropy(), "fresh system randomness is added every time");
        let mut e = Entropy::new();
        assert!(e.add(egui::pos2(10.0, 10.0), 1.0));
        assert!(!e.add(egui::pos2(12.0, 11.0), 1.1), "a tiny move is not counted");
        assert!(e.add(egui::pos2(60.0, 40.0), 1.2));
        assert_eq!(e.samples, 2);
        assert_eq!(e.finish().word_count(), 24);
        for i in 0..400 {
            e.add(egui::pos2(i as f32 * 20.0, 0.0), 2.0 + i as f64);
        }
        assert_eq!(e.progress(), 1.0);
        assert!(e.dots.len() <= 90);
    }
}
