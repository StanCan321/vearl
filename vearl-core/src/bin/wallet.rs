//! vearl-wallet: command-line wallet (TESTNET, unaudited).
//!
//!   vearl-wallet new      --file W.json                       creates a wallet (shows the phrase ONCE)
//!   vearl-wallet restore  --file W.json                       reads the phrase from standard input
//!   vearl-wallet address  --file W.json
//!   vearl-wallet phrase   --file W.json                       shows the phrase (after the password)
//!   vearl-wallet balance  (--file W.json | --address vrl1...)  --node HOST:PORT
//!   vearl-wallet send     --file W.json --node HOST:PORT --to vrl1... --amount 12.5 [--fee 0.01]
//!
//! Password: variable VEARL_WALLET_PASSWORD, or --password-file F, or masked prompt on a terminal, or one
//! line on standard input. NEVER pass it as a command-line argument (visible in `ps`).
use ed25519_dalek::SigningKey;
use vearl_core::crypto::*;
use vearl_core::net::*;
use vearl_core::types::*;
use vearl_core::wallet::*;
use std::io::{BufRead, IsTerminal};
use std::path::PathBuf;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn read_password(args: &[String], prompt: &str) -> String {
    if let Ok(p) = std::env::var("VEARL_WALLET_PASSWORD") {
        return p;
    }
    if let Some(f) = arg(args, "--password-file") {
        return std::fs::read_to_string(&f).unwrap_or_else(|e| die(&format!("{f}: {e}"))).trim_end_matches(['\n', '\r']).to_string();
    }
    if std::io::stdin().is_terminal() {
        return rpassword::prompt_password(prompt).unwrap_or_else(|e| die(&e.to_string()));
    }
    let mut l = String::new();
    std::io::stdin().lock().read_line(&mut l).unwrap();
    l.trim_end_matches(['\n', '\r']).to_string()
}

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(1)
}

fn unlock_mnemonic(args: &[String]) -> (Keystore, bip39::Mnemonic) {
    let file = PathBuf::from(arg(args, "--file").unwrap_or_else(|| die("--file required")));
    let ks = Keystore::load(&file).unwrap_or_else(|e| die(&e));
    let m = ks.unlock(&read_password(args, "Password: ")).unwrap_or_else(|e| die(&e));
    (ks, m)
}

fn unlock(args: &[String]) -> (Keystore, SigningKey) {
    let (ks, m) = unlock_mnemonic(args);
    (ks, derive_account(&m, 0))
}

const HELP: &str = r#"vearl-wallet: command-line wallet for the Vearl testnet (tokens have no value, not audited).

  vearl-wallet new        --file wallet.json                    create a wallet; shows 24 recovery words ONCE (write them on paper)
  vearl-wallet restore    --file wallet.json                    restore a wallet from its 24 words (read from standard input)
  vearl-wallet address    --file wallet.json                    show the address (vrl1...) of a wallet
  vearl-wallet pq-address --file wallet.json                    show the post-quantum address (vrlq1...)
  vearl-wallet phrase     --file wallet.json                    show the recovery words again (asks the password)
  vearl-wallet balance    --file wallet.json --node HOST:PORT   balance of a wallet (or --address vrl1... instead of --file)
  vearl-wallet send       --file wallet.json --node HOST:PORT --to vrl1... --amount 12.5 [--fee 0.01] [--pq]
  vearl-wallet sign       --file wallet.json --message "TEXT"                 sign a text to prove that you own the address (testnet rewards)
  vearl-wallet verify     --address vrl1... --message "TEXT" --signature HEX   check such a signature

The password is asked in the terminal (hidden); it can also be given in the VEARL_WALLET_PASSWORD variable or with --password-file FILE.
Never type it as a command-line argument.

Example:  vearl-wallet new --file wallet.json"#;

fn main() {
    vearl_core::launcher::help_or_continue(HELP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("new") => {
            let file = PathBuf::from(arg(&args, "--file").unwrap_or_else(|| die("--file required")));
            if file.exists() {
                die("this file already exists: refusing to overwrite a wallet");
            }
            let pw = read_password(&args, "Choose a password (8+ characters): ");
            let m = new_mnemonic();
            let ks = Keystore::create(&m, &pw, KdfParams::STRONG).unwrap_or_else(|e| die(&e));
            ks.save(&file).unwrap_or_else(|e| die(&e));
            println!("Wallet created: {}", file.display());
            println!("Address: {}\n", ks.address);
            println!("RECOVERY PHRASE (24 words) - write it down on paper, it will not be shown again without the password:\n");
            println!("{m}\n");
            println!("Whoever has this phrase has the funds. Do not photograph it, do not send it to anyone.");
        }
        Some("restore") => {
            let file = PathBuf::from(arg(&args, "--file").unwrap_or_else(|| die("--file required")));
            eprintln!("Enter the 24-word phrase then press Enter:");
            let mut phrase = String::new();
            std::io::stdin().lock().read_line(&mut phrase).unwrap();
            let m = parse_mnemonic(&phrase).unwrap_or_else(|e| die(&e));
            let pw = read_password(&args, "New password (8+ characters): ");
            let ks = Keystore::create(&m, &pw, KdfParams::STRONG).unwrap_or_else(|e| die(&e));
            ks.save(&file).unwrap_or_else(|e| die(&e));
            println!("Wallet restored. Address: {}", ks.address);
        }
        Some("address") => {
            let file = PathBuf::from(arg(&args, "--file").unwrap_or_else(|| die("--file required")));
            println!("{}", Keystore::load(&file).unwrap_or_else(|e| die(&e)).address);
        }
        Some("pq-address") => {
            // POST-QUANTUM address (SLH-DSA-SHAKE-128s) of account 0: derived from the same phrase, prefix `vrlq`
            let (_, m) = unlock_mnemonic(&args);
            println!("{}", pq_address_of(&derive_pq_account(&m, 0).pubkey()));
        }
        Some("phrase") => {
            let (_, m) = unlock_mnemonic(&args);
            println!("{m}");
        }
        Some("balance") => {
            let node = arg(&args, "--node").unwrap_or_else(|| die("--node required"));
            let pk = match arg(&args, "--address") {
                Some(a) => parse_address(&a).unwrap_or_else(|e| die(&e)),
                None => parse_address(&Keystore::load(&PathBuf::from(arg(&args, "--file").unwrap_or_else(|| die("--file or --address required")))).unwrap_or_else(|e| die(&e)).address).unwrap(),
            };
            match request(&node, &Msg::GetAccount(pk)).unwrap_or_else(|e| die(&e.to_string())) {
                Msg::Account { balance, nonce, .. } => println!("{} VRL (nonce {nonce})", fmt_amount(balance)),
                m => die(&format!("unexpected reply: {m:?}")),
            }
        }
        Some("send") => {
            let node = arg(&args, "--node").unwrap_or_else(|| die("--node required"));
            let to = parse_address(&arg(&args, "--to").unwrap_or_else(|| die("--to required"))).unwrap_or_else(|e| die(&e));
            let amount = parse_amount(&arg(&args, "--amount").unwrap_or_else(|| die("--amount required"))).unwrap_or_else(|e| die(&e));
            let fee = arg(&args, "--fee").map_or(0, |f| parse_amount(&f).unwrap_or_else(|e| die(&e)));
            let pq = args.iter().any(|a| a == "--pq");
            let (ks, m) = unlock_mnemonic(&args);
            let sk = derive_account(&m, 0);
            let pqk = pq.then(|| derive_pq_account(&m, 0));
            let from = pqk.as_ref().map_or_else(|| pubkey(&sk), |k| k.pubkey());
            let ks_addr = if pq { pq_address_of(&from) } else { ks.address.clone() };
            let (balance, next_nonce) = match request(&node, &Msg::GetAccount(from)).unwrap_or_else(|e| die(&e.to_string())) {
                Msg::Account { balance, next_nonce, .. } => (balance, next_nonce),
                m => die(&format!("unexpected reply: {m:?}")),
            };
            if balance < amount.saturating_add(fee) {
                die(&format!("insufficient balance: {} VRL available", fmt_amount(balance)));
            }
            println!("Sending {} VRL (fee {}) from {} to {}", fmt_amount(amount), fmt_amount(fee), ks_addr, address_of(&to));
            let tx0 = Tx { from, nonce: next_nonce, fee, kind: TxKind::Transfer { to, amount } };
            let tx = match &pqk {
                Some(k) => {
                    println!("Post-quantum SLH-DSA signature (7.8 KiB, about 2 s)...");
                    SignedTx::new_pq(tx0, k)
                }
                None => SignedTx::new(tx0, &sk),
            };
            match request(&node, &Msg::NewTx(tx.clone())).unwrap_or_else(|e| die(&e.to_string())) {
                Msg::Ack { ok: true, .. } => println!("Transaction accepted by the node: {}", tx.id().iter().map(|b| format!("{b:02x}")).collect::<String>()),
                Msg::Ack { ok: false, info } => die(&format!("transaction refused: {info}")),
                m => die(&format!("unexpected reply: {m:?}")),
            }
        }
        Some("sign") => {
            let msg = arg(&args, "--message").unwrap_or_else(|| die("--message required"));
            let (_, m) = unlock_mnemonic(&args);
            let sk = derive_account(&m, 0);
            let sig = sign_message(&sk, msg.as_bytes());
            println!("address:   {}", address_of(&pubkey(&sk)));
            println!("message:   {msg}");
            println!("signature: {}", sig.iter().map(|b| format!("{b:02x}")).collect::<String>());
        }
        Some("verify") => {
            let addr = parse_address(&arg(&args, "--address").unwrap_or_else(|| die("--address required"))).unwrap_or_else(|e| die(&e));
            let msg = arg(&args, "--message").unwrap_or_else(|| die("--message required"));
            let hexsig = arg(&args, "--signature").unwrap_or_else(|| die("--signature required"));
            let sig: Vec<u8> = (0..hexsig.len() / 2).filter_map(|i| u8::from_str_radix(hexsig.get(2 * i..2 * i + 2)?, 16).ok()).collect();
            if hexsig.len() % 2 == 0 && sig.len() * 2 == hexsig.len() && verify_message(&addr, msg.as_bytes(), &sig) {
                println!("valid: this signature was made by the owner of {}", address_of(&addr));
            } else {
                println!("INVALID signature");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("usage: vearl-wallet (new | restore | address | pq-address | phrase | balance | send | sign | verify) ... (run it without arguments for the help)");
            std::process::exit(2);
        }
    }
}
