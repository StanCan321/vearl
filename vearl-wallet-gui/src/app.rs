//! State of the window and everything that happens in the background. Drawing is in `ui_start.rs` and `ui_wallet.rs`: they only change
//! the state and push [`Cmd`]s, which are carried out here (threads, files, settings), so no drawing code ever blocks or touches the disk.
use crate::core::*;
use crate::theme::Tone;
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};
use vearl_core::crypto::PubKey;
use vearl_core::wallet::{derive_pq_account, new_mnemonic, parse_address, parse_amount, parse_mnemonic, pq_address_of};
use zeroize::Zeroize;

pub const REFRESH_EVERY: Duration = Duration::from_secs(15);

// ------------------------------------------------------------------------------------------------ screens

pub enum Screen {
    Start(StartState),
    Create(Box<Create>),
    Restore(Box<Restore>),
    Wallet(Box<Session>),
}

#[derive(Default)]
pub struct StartState {
    pub show_open: bool,
    pub open_path: String,
    pub error: Option<String>,
}

pub struct Create {
    /// 0 = name and password, 1 = write the words down, 2 = check them, 3 = done
    pub step: u8,
    pub name: String,
    pub password: String,
    pub confirm: String,
    pub phrase: Option<bip39::Mnemonic>,
    pub revealed: bool,
    pub written: bool,
    pub check_idx: [usize; 3],
    pub check_in: [String; 3],
    pub error: Option<String>,
    pub busy: bool,
    pub done: Option<WalletEntry>,
}

impl Create {
    pub fn new() -> Create {
        Create {
            step: 0,
            name: "my-wallet".into(),
            password: String::new(),
            confirm: String::new(),
            phrase: None,
            revealed: false,
            written: false,
            check_idx: [0, 8, 16],
            check_in: Default::default(),
            error: None,
            busy: false,
            done: None,
        }
    }
    pub fn words(&self) -> Vec<String> {
        self.phrase.as_ref().map(|m| m.words().map(str::to_string).collect()).unwrap_or_default()
    }
}

pub struct Restore {
    pub name: String,
    pub phrase: String,
    pub password: String,
    pub confirm: String,
    pub error: Option<String>,
    pub busy: bool,
    pub done: Option<WalletEntry>,
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

pub struct PwPrompt {
    pub action: Action,
    pub password: String,
    pub error: Option<String>,
    pub busy: bool,
}

pub enum Modal {
    Password(PwPrompt),
    Phrase(Vec<String>),
    Sent { txid: String, from: String, to: String, amount: u64, fee: u64 },
}

pub struct Session {
    pub sid: u64,
    pub entry: WalletEntry,
    pub pk: PubKey,
    pub page: Page,
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
    pub toast: Option<(Tone, String, Instant)>,
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
    TestNode,
    Link(String),
    Reveal(PathBuf),
    Forget(String),
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
    tx: Sender<Ev>,
    rx: Receiver<Ev>,
    next_sid: u64,
    #[cfg(feature = "devshots")]
    pub shots: Option<crate::demo::Shots>,
}

impl App {
    pub fn new(ctx: egui::Context, logo: egui::TextureHandle) -> App {
        let settings = Settings::load();
        let network = load_network();
        let wallets = list_wallets(&settings);
        let (tx, rx) = channel();
        App {
            ctx,
            logo,
            settings,
            network,
            wallets,
            screen: Screen::Start(StartState::default()),
            demo: false,
            cmds: vec![],
            tx,
            rx,
            next_sid: 1,
            #[cfg(feature = "devshots")]
            shots: None,
        }
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

    pub fn open_session(&mut self, entry: WalletEntry) {
        let Ok(pk) = parse_address(&entry.address) else { return };
        let sid = self.next_sid;
        self.next_sid += 1;
        self.settings.last_wallet = entry.path.display().to_string();
        self.settings.save();
        let s = Session {
            sid,
            entry,
            pk,
            page: Page::Overview,
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
            settings: SettingsForm { node: self.settings.node.clone(), explorer: self.settings.explorer.clone(), ..Default::default() },
            modal: None,
            toast: None,
        };
        self.screen = Screen::Wallet(Box::new(s));
        self.cmds.push(Cmd::Refresh);
        self.cmds.push(Cmd::RefreshActivity);
    }

    // -------------------------------------------------------------------------------------------- events from the threads

    pub fn poll(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            self.handle(ev);
        }
    }

    fn handle(&mut self, ev: Ev) {
        match ev {
            Ev::Created(res) => {
                match (&mut self.screen, res) {
                    (Screen::Create(c), Ok(entry)) => {
                        c.busy = false;
                        c.step = 3;
                        c.done = Some(entry);
                    }
                    (Screen::Create(c), Err(e)) => {
                        c.busy = false;
                        c.error = Some(e);
                    }
                    (Screen::Restore(r), Ok(entry)) => {
                        r.busy = false;
                        r.done = Some(entry);
                    }
                    (Screen::Restore(r), Err(e)) => {
                        r.busy = false;
                        r.error = Some(e);
                    }
                    _ => {}
                }
                self.wallets = list_wallets(&self.settings);
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
                let action = match &s.modal {
                    Some(Modal::Password(p)) => Some(p.action.clone()),
                    _ => None,
                };
                match res {
                    Err(e) => {
                        if let Some(Modal::Password(p)) = &mut s.modal {
                            p.busy = false;
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
                        let _ = action;
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
            Cmd::GoStart => {
                self.wallets = list_wallets(&self.settings);
                self.screen = Screen::Start(StartState::default());
            }
            Cmd::StartCreate => self.screen = Screen::Create(Box::new(Create::new())),
            Cmd::StartRestore => {
                self.screen = Screen::Restore(Box::new(Restore { name: "restored-wallet".into(), phrase: String::new(), password: String::new(), confirm: String::new(), error: None, busy: false, done: None }))
            }
            Cmd::CreateNext => self.create_next(),
            Cmd::CreateBack => {
                if let Screen::Create(c) = &mut self.screen {
                    c.error = None;
                    c.step = c.step.saturating_sub(1);
                }
            }
            Cmd::CreateFinish => self.create_finish(),
            Cmd::RestoreFinish => self.restore_finish(),
            Cmd::Refresh => self.refresh(),
            Cmd::RefreshActivity => self.refresh_activity(),
            Cmd::CheckRegistered => {
                if let Some((sid, addr)) = self.session_mut().map(|s| (s.sid, s.entry.address.clone())) {
                    let addr2 = addr.clone();
                    self.spawn(move || Ev::Registered { sid, res: is_registered(&addr) });
                    self.spawn(move || Ev::Standing { sid, res: fetch_standing(&addr2) });
                }
            }
            Cmd::Ask(action) => {
                if let Some(s) = self.session_mut() {
                    s.modal = Some(Modal::Password(PwPrompt { action, password: String::new(), error: None, busy: false }));
                }
            }
            Cmd::CancelModal => {
                if let Some(s) = self.session_mut() {
                    if let Some(Modal::Password(p)) = &mut s.modal {
                        p.password.zeroize();
                    }
                    s.modal = None;
                }
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
            Cmd::TestNode => {
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
                self.spawn(move || Ev::NodeTest { sid, res: node_status(&nodes) });
            }
            Cmd::Link(u) => open_link(&u),
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
        let Some(s) = self.session_mut() else { return };
        let (sid, path) = (s.sid, s.entry.path.clone());
        let Some(Modal::Password(p)) = &mut s.modal else { return };
        if p.busy {
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
            let res = run_action(&path, &password, action, &nodes);
            password.zeroize();
            Ev::Action { sid, res }
        });
    }

    fn send_mail(&mut self) {
        let Some(s) = self.session_mut() else { return };
        let Some(l) = s.rewards.lines.clone() else { return };
        if s.rewards.sending {
            return;
        }
        s.rewards.sending = true;
        s.rewards.sent = None;
        let sid = s.sid;
        self.spawn(move || {
            let reg = vearl_core::regmail::Registration { address: l.address, message: l.message, signature_hex: l.signature };
            Ev::Mail { sid, res: vearl_core::regmail::send_registration(&reg) }
        });
    }

    // -------------------------------------------------------------------------------------------- creating and restoring

    fn create_next(&mut self) {
        let Screen::Create(c) = &mut self.screen else { return };
        c.error = None;
        match c.step {
            0 => {
                if let Err(e) = wallet_path_for(&c.name).and_then(|p| if p.exists() { Err(format!("{} already exists: choose another name", p.display())) } else { Ok(()) }) {
                    c.error = Some(e);
                    return;
                }
                if c.password.chars().count() < 8 {
                    c.error = Some("The password needs at least 8 characters.".into());
                    return;
                }
                if c.password != c.confirm {
                    c.error = Some("The two passwords are not the same.".into());
                    return;
                }
                if c.phrase.is_none() {
                    let m = new_mnemonic();
                    let e = m.to_entropy();
                    c.check_idx = [(e[0] % 8) as usize, 8 + (e[1] % 8) as usize, 16 + (e[2] % 8) as usize];
                    c.phrase = Some(m);
                }
                c.revealed = false;
                c.written = false;
                c.step = 1;
            }
            1 => {
                if !c.written {
                    c.error = Some("Tick the box once the 24 words are written on paper.".into());
                    return;
                }
                c.check_in = Default::default();
                c.step = 2;
            }
            _ => {}
        }
    }

    fn create_finish(&mut self) {
        let Screen::Create(c) = &mut self.screen else { return };
        let words = c.words();
        for (k, &i) in c.check_idx.iter().enumerate() {
            if c.check_in[k].trim().to_lowercase() != words.get(i).cloned().unwrap_or_default() {
                c.error = Some(format!("Word {} is not right. Look at your paper (or go back to see the words again).", i + 1));
                return;
            }
        }
        let (Some(m), Ok(path)) = (c.phrase.clone(), wallet_path_for(&c.name)) else { return };
        c.busy = true;
        c.error = None;
        let pw = c.password.clone();
        self.spawn(move || {
            let mut pw = pw;
            let r = create_wallet(&path, &m, &pw);
            pw.zeroize();
            Ev::Created(r)
        });
    }

    fn restore_finish(&mut self) {
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
        if r.password.chars().count() < 8 {
            r.error = Some("The new password needs at least 8 characters.".into());
            return;
        }
        if r.password != r.confirm {
            r.error = Some("The two passwords are not the same.".into());
            return;
        }
        r.busy = true;
        let pw = r.password.clone();
        self.spawn(move || {
            let mut pw = pw;
            let res = create_wallet(&path, &m, &pw);
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
    use vearl_core::wallet::{address_of, derive_account};

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
}
