//! End-to-end test of the window's logic against a real local node: set VEARL_NODE_BIN to the path of the `node` program of a release
//! (it is skipped otherwise). Creates a wallet through the same commands the buttons use, then reads the balance, pays (standard and
//! post-quantum signature), reveals the post-quantum address, signs the registration, shows the words and changes the password.
use crate::app::*;
use crate::core::*;
use eframe::egui;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use vearl_core::crypto::pubkey;
use vearl_core::state::COIN;
use vearl_core::wallet::{address_of, derive_account, derive_pq_account, new_mnemonic, parse_mnemonic, pq_address_of};

fn wait_until(app: &mut App, what: &str, mut cond: impl FnMut(&mut App) -> bool) {
    let t0 = Instant::now();
    loop {
        app.poll();
        app.run_cmds();
        if cond(app) {
            return;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "timeout waiting for: {what}");
        std::thread::sleep(Duration::from_millis(40));
    }
}

fn set_pw(app: &mut App, pw: &str) {
    if let Some(Modal::Password(p)) = &mut app.session_mut().unwrap().modal {
        p.password = pw.to_string();
    } else {
        panic!("no password prompt");
    }
}

struct Kill(std::process::Child);
impl Drop for Kill {
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}

#[test]
fn window_logic_against_a_real_node() {
    let Some(node_bin) = std::env::var_os("VEARL_NODE_BIN") else {
        eprintln!("skipped: set VEARL_NODE_BIN to run the end-to-end test");
        return;
    };
    let tmp = std::env::temp_dir().join(format!("vearl-gui-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::env::set_var("APPDATA", &tmp);
    std::env::set_var("XDG_DATA_HOME", &tmp);

    let ctx = egui::Context::default();
    let logo = ctx.load_texture("t", egui::ColorImage::example(), Default::default());
    let mut app = App::new(ctx, logo);

    // ---- create a wallet with the buttons' commands
    app.cmds.push(Cmd::StartCreate);
    app.run_cmds();
    {
        let Screen::Create(c) = &mut app.screen else { panic!() };
        c.name = "e2e wallet".into();
        c.password = "first-password-123".into();
        c.confirm = "first-password-123".into();
    }
    app.cmds.push(Cmd::CreateNext);
    app.run_cmds();
    let words = {
        let Screen::Create(c) = &mut app.screen else { panic!() };
        assert_eq!(c.step, 1, "{:?}", c.error);
        c.words()
    };
    assert_eq!(words.len(), 24);
    app.cmds.push(Cmd::CreateNext);
    app.run_cmds();
    {
        let Screen::Create(c) = &mut app.screen else { panic!() };
        assert_eq!(c.step, 1, "must refuse until the paper box is ticked");
        c.written = true;
    }
    app.cmds.push(Cmd::CreateNext);
    app.run_cmds();
    {
        let Screen::Create(c) = &mut app.screen else { panic!() };
        assert_eq!(c.step, 2);
        c.check_in = [words[c.check_idx[0]].clone(), "wrong".into(), words[c.check_idx[2]].clone()];
    }
    app.cmds.push(Cmd::CreateFinish);
    app.run_cmds();
    {
        let Screen::Create(c) = &mut app.screen else { panic!() };
        assert!(c.error.is_some() && !c.busy, "a wrong word must be refused");
        c.check_in[1] = words[c.check_idx[1]].clone();
    }
    app.cmds.push(Cmd::CreateFinish);
    wait_until(&mut app, "wallet created", |a| matches!(&a.screen, Screen::Create(c) if c.step == 3));
    let entry = match &app.screen {
        Screen::Create(c) => c.done.clone().unwrap(),
        _ => unreachable!(),
    };
    assert!(entry.path.exists() && entry.path.ends_with("e2e-wallet.json"));
    let m = parse_mnemonic(&words.join(" ")).unwrap();
    let pq_addr = pq_address_of(&derive_pq_account(&m, 0).pubkey());

    // ---- a local node whose genesis pays the two accounts of that wallet
    let node_dir = std::path::Path::new(&node_bin).parent().unwrap().to_path_buf();
    let genesis = std::fs::read_to_string(node_dir.join("genesis.json")).expect("genesis.json next to the node program");
    let mut g: serde_json::Value = serde_json::from_str(&genesis).unwrap();
    g["alloc"] = serde_json::json!([{ "address": entry.address, "amount": "1000.00000000" }, { "address": pq_addr, "amount": "500.00000000" }]);
    let gpath = tmp.join("genesis.json");
    std::fs::write(&gpath, serde_json::to_string(&g).unwrap()).unwrap();
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let child = Command::new(&node_bin)
        .args(["--genesis", gpath.to_str().unwrap(), "--listen", &format!("127.0.0.1:{port}"), "--data-dir", tmp.join("node").to_str().unwrap()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start the node");
    let guard = Kill(child);
    let node = format!("127.0.0.1:{port}");
    let t0 = Instant::now();
    while node_status(&[node.clone()]).is_err() {
        assert!(t0.elapsed() < Duration::from_secs(40), "the node did not start");
        std::thread::sleep(Duration::from_millis(200));
    }
    app.settings.node = node.clone();
    app.network.nodes = vec![node.clone()];

    // ---- open the wallet: balance
    app.cmds.push(Cmd::OpenWallet(entry.clone()));
    wait_until(&mut app, "balance", |a| a.session_mut().map_or(false, |s| s.account.is_some() && matches!(s.status, Some(Ok(_)))));
    assert_eq!(app.session_mut().unwrap().account.as_ref().unwrap().balance, 1000 * COIN);
    let other = address_of(&pubkey(&derive_account(&new_mnemonic(), 0)));

    // ---- pay: a wrong password first, then the right one
    app.cmds.push(Cmd::Ask(Action::Send { to: other.clone(), amount: 10 * COIN, fee: 1_000_000, pq: false }));
    app.run_cmds();
    set_pw(&mut app, "not-the-password");
    app.cmds.push(Cmd::SubmitPassword);
    wait_until(&mut app, "wrong password refused", |a| matches!(&a.session_mut().unwrap().modal, Some(Modal::Password(p)) if p.error.is_some() && !p.busy));
    set_pw(&mut app, "first-password-123");
    app.cmds.push(Cmd::SubmitPassword);
    wait_until(&mut app, "payment sent", |a| matches!(a.session_mut().unwrap().modal, Some(Modal::Sent { .. })));
    assert_eq!(node_status(&[node.clone()]).unwrap().mempool, 1, "the node holds the payment");
    app.session_mut().unwrap().modal = None;

    // ---- post-quantum address, balance and payment
    app.cmds.push(Cmd::Ask(Action::RevealPq));
    app.run_cmds();
    set_pw(&mut app, "first-password-123");
    app.cmds.push(Cmd::SubmitPassword);
    wait_until(&mut app, "pq balance", |a| a.session_mut().map_or(false, |s| s.pq_account.is_some()));
    {
        let s = app.session_mut().unwrap();
        assert_eq!(s.pq_address.as_deref(), Some(pq_addr.as_str()));
        assert_eq!(s.pq_account.as_ref().unwrap().balance, 500 * COIN);
    }
    app.cmds.push(Cmd::Ask(Action::Send { to: other.clone(), amount: 5 * COIN, fee: 0, pq: true }));
    app.run_cmds();
    set_pw(&mut app, "first-password-123");
    app.cmds.push(Cmd::SubmitPassword);
    wait_until(&mut app, "pq payment sent", |a| matches!(&a.session_mut().unwrap().modal, Some(Modal::Sent { from, .. }) if from.starts_with("vrlq1")));
    app.session_mut().unwrap().modal = None;

    // ---- registration lines, words, password change
    app.cmds.push(Cmd::Ask(Action::Register { contact: "someone@example.org".into() }));
    app.run_cmds();
    set_pw(&mut app, "first-password-123");
    app.cmds.push(Cmd::SubmitPassword);
    wait_until(&mut app, "registration signed", |a| a.session_mut().unwrap().rewards.lines.is_some());
    let l = app.session_mut().unwrap().rewards.lines.clone().unwrap();
    assert_eq!(l.address, entry.address);
    assert_eq!(verify_text(&l.address, &l.message, &l.signature), Ok(true));

    app.cmds.push(Cmd::Ask(Action::ShowPhrase));
    app.run_cmds();
    set_pw(&mut app, "first-password-123");
    app.cmds.push(Cmd::SubmitPassword);
    wait_until(&mut app, "words shown", |a| matches!(a.session_mut().unwrap().modal, Some(Modal::Phrase(_))));
    match &app.session_mut().unwrap().modal {
        Some(Modal::Phrase(w)) => assert_eq!(*w, words),
        _ => unreachable!(),
    }
    app.session_mut().unwrap().modal = None;

    app.cmds.push(Cmd::Ask(Action::ChangePassword { new: "second-password-456".into() }));
    app.run_cmds();
    set_pw(&mut app, "first-password-123");
    app.cmds.push(Cmd::SubmitPassword);
    wait_until(&mut app, "password changed", |a| a.session_mut().unwrap().modal.is_none());
    assert!(unlock(&entry.path, "first-password-123").is_err());
    assert_eq!(unlock(&entry.path, "second-password-456").unwrap().to_entropy(), m.to_entropy());

    drop(guard);
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Wallet files are interchangeable with the command-line wallet: set VEARL_WALLET_BIN to the path of `vearl-wallet` to run this test.
#[test]
fn wallet_files_are_shared_with_the_command_line_wallet() {
    let Some(cli) = std::env::var_os("VEARL_WALLET_BIN") else {
        eprintln!("skipped: set VEARL_WALLET_BIN to run the interchange test");
        return;
    };
    let tmp = std::env::temp_dir().join(format!("vearl-gui-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let run = |args: &[&str]| {
        let out = Command::new(&cli).args(args).env("VEARL_WALLET_PASSWORD", "shared-password-123").output().expect("run vearl-wallet");
        assert!(out.status.success(), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).to_string()
    };

    // made by the command line, read by the window code
    let a = tmp.join("cli.json");
    let created = run(&["new", "--file", a.to_str().unwrap()]);
    let entry = entry_of(&a).unwrap();
    assert!(created.contains(&entry.address));
    let words_cli: Vec<String> = created.lines().find(|l| l.split_whitespace().count() == 24).unwrap().split_whitespace().map(str::to_string).collect();
    let m = unlock(&a, "shared-password-123").unwrap();
    assert_eq!(m.words().map(str::to_string).collect::<Vec<_>>(), words_cli);

    // made by the window code, read by the command line
    let b = tmp.join("gui.json");
    let m2 = new_mnemonic();
    let e2 = create_wallet(&b, &m2, "shared-password-123").unwrap();
    assert_eq!(run(&["address", "--file", b.to_str().unwrap()]).trim(), e2.address);
    assert_eq!(run(&["phrase", "--file", b.to_str().unwrap()]).trim(), m2.to_string());
    let pq_cli = run(&["pq-address", "--file", b.to_str().unwrap()]);
    assert_eq!(pq_cli.trim(), pq_address_of(&derive_pq_account(&m2, 0).pubkey()));
    // the registration signed by the window code is the one the command line signs
    let reg_cli = run(&["register", "--file", b.to_str().unwrap(), "--contact", "me@example.org", "--print-only"]);
    let reg = sign_registration(&m2, "me@example.org").unwrap();
    assert!(reg_cli.contains(&reg.signature_hex) && reg_cli.contains(&reg.message));
    let _ = std::fs::remove_dir_all(&tmp);
}
