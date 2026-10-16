//! vearl-miner: THE single Vearl miner (testnet). One program to open:
//!   * first time: a wizard creates (or chooses) the Vearl wallet, and, only if the optional Quantus add-on is installed, offers to mine
//!     Quantus (QTC) too (answer: no by default);
//!   * afterwards: starts the Vearl node, which mines Vearl with the whole GPU (and computes the paid jobs). If the Quantus add-on is
//!     installed AND enabled, the GPU is shared (80 % Quantus / 20 % Vearl by default) and goes back to Vearl as soon as a paid job is
//!     assigned to it.
//!
//! vearl-miner [setup | check] [--home DIR] [--devnet]
//!   setup   redoes the wizard (wallet, optional Quantus)    check   checks the GPU, the network files and the add-on (if present), and exits
//! Files (in the program folder, or --home): network.json + genesis.json (provided by the release), miner.json (settings),
//! wallet.json, wallet.pw (only if you ask to remember the password), data/, miner.log, external-miner.log, fee-miner.log.
//! The node (`node`) and, when enabled, the Quantus miner (`quantus-miner`) are started as child processes; closing this window stops everything.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use vearl_core::dual::{dev_qtc_address, qtc_fee_pct, valid_qtc_address, QTC_POOL_ADDR, QTC_POOL_CERT};
use vearl_core::wallet::{load_signing_key, new_mnemonic, Keystore, KdfParams};

const WINDOWS: bool = cfg!(windows);

fn die(m: &str) -> ! {
    eprintln!("\nERROR: {m}");
    pause_on_windows();
    std::process::exit(1)
}

/// Double-click on Windows: the window would close on the error message.
fn pause_on_windows() {
    use std::io::IsTerminal;
    if WINDOWS && io::stdin().is_terminal() {
        eprintln!("\nPress Enter to close this window.");
        let mut s = String::new();
        let _ = io::stdin().read_line(&mut s);
    }
}

fn exe(name: &str) -> String {
    if WINDOWS { format!("{name}.exe") } else { name.to_string() }
}

fn ask(prompt: &str, default: &str) -> String {
    print!("{prompt}{}", if default.is_empty() { " ".to_string() } else { format!(" [{default}] ") });
    let _ = io::stdout().flush();
    let mut s = String::new();
    if io::stdin().lock().read_line(&mut s).unwrap_or(0) == 0 {
        die("no input available: run vearl-miner in a terminal (or double-click it) the first time");
    }
    let s = s.trim().to_string();
    if s.is_empty() { default.to_string() } else { s }
}

fn yes(prompt: &str, default_yes: bool) -> bool {
    let a = ask(&format!("{prompt} ({})", if default_yes { "Y/n" } else { "y/N" }), "").to_lowercase();
    if a.is_empty() { default_yes } else { a.starts_with('y') || a.starts_with('o') }
}

fn password_prompt(p: &str) -> String {
    vearl_core::wallet::read_password(p, None).unwrap_or_else(|e| die(&format!("cannot read the password: {e}")))
}

struct Env {
    home: PathBuf,
    network: Value,
}

impl Env {
    fn path(&self, f: &str) -> PathBuf {
        self.home.join(f)
    }
}

fn load_json(p: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()
}

fn gpu_status() -> Result<String, String> {
    #[cfg(feature = "gpu")]
    {
        vearl_core::gpu::check_supported()
    }
    #[cfg(not(feature = "gpu"))]
    {
        Err("this build has no GPU support".to_string())
    }
}

/// Is the optional Quantus add-on installed (the program sits next to vearl-miner)? Nothing is run here: by default the third-party
/// program is never started.
fn quantus_present(env: &Env) -> bool {
    env.path(&exe("quantus-miner")).exists()
}

/// Starts the Quantus miner twice, briefly, to know whether it works here: `--version`, then a 3-second benchmark with the native CUDA
/// engine, otherwise with the Vulkan engine. Returns the program and whether the CUDA engine is the one to use. Only called when the
/// user enabled Quantus mining (or for `vearl-miner check`).
fn prepare_quantus(env: &Env) -> Result<(PathBuf, bool), String> {
    let m = quantus_miner_ok(env)?;
    print!("Testing the Quantus engine on your GPU... ");
    let _ = io::stdout().flush();
    match probe_engine(env, &m, true) {
        Some(r) => {
            println!("CUDA engine OK ({r:.0} MH/s)");
            Ok((m, true))
        }
        None => match probe_engine(env, &m, false) {
            Some(r) => {
                println!("CUDA engine unavailable, using the Vulkan engine instead ({r:.0} MH/s, slower)");
                Ok((m, false))
            }
            None => {
                println!("no working Quantus engine");
                Err("the Quantus miner cannot use this GPU".to_string())
            }
        },
    }
}

/// Is the Quantus miner there, and does it start? (`--version` exits immediately)
fn quantus_miner_ok(env: &Env) -> Result<PathBuf, String> {
    let p = env.path(&exe("quantus-miner"));
    if !p.exists() {
        return Err(format!("{} is missing (it must sit next to vearl-miner)", p.display()));
    }
    let mut c = Command::new(&p);
    c.arg("--version").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    apply_lib_path(&mut c, &env.home);
    match c.status() {
        Ok(s) if s.success() => Ok(p),
        Ok(s) => Err(format!("{} does not start (exit code {:?})", p.display(), s.code())),
        Err(e) => Err(format!("{} cannot be run: {e}", p.display())),
    }
}

/// Measures the engine of the Quantus miner for a few seconds (`benchmark`): returns the rate in MH/s, or None if it fails (CUDA library
/// missing, driver too old, crash...). `cuda` chooses the native CUDA engine, otherwise the Vulkan/wgpu engine (3 to 4 times slower).
fn probe_engine(env: &Env, miner: &Path, cuda: bool) -> Option<f64> {
    let mut c = Command::new(miner);
    c.arg("benchmark");
    if cuda {
        c.arg("--cuda-gpu");
    }
    c.args(["--gpu-devices", "1", "--cpu-workers", "0", "-d", "3"]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    apply_lib_path(&mut c, &env.home);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            c.pre_exec(|| {
                prctl(1, 15, 0, 0, 0); // PR_SET_PDEATHSIG, SIGTERM
                Ok(())
            });
        }
    }
    let mut child = c.spawn().ok()?;
    let start = std::time::Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) if start.elapsed().as_secs() < 60 => std::thread::sleep(std::time::Duration::from_millis(100)),
            _ => {
                let _ = child.kill();
                return None;
            }
        }
    };
    let mut out = String::new();
    if let Some(mut o) = child.stdout.take() {
        let _ = std::io::Read::read_to_string(&mut o, &mut out);
    }
    if !status.success() {
        return None;
    }
    // "Average rate: 123.45M H/s"
    let line = out.lines().find(|l| l.contains("Average rate"))?;
    let num: String = line.split(':').nth(1)?.trim().chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let v: f64 = num.parse().ok()?;
    let unit = line.split(':').nth(1)?.trim().trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
    Some(if unit.starts_with('G') { v * 1000.0 } else if unit.starts_with('k') || unit.starts_with('K') { v / 1000.0 } else { v })
}

/// Linux: the libraries shipped alongside (NVRTC for the Quantus miner) are found through LD_LIBRARY_PATH.
fn apply_lib_path(c: &mut Command, home: &Path) {
    if !WINDOWS {
        let old = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        c.env("LD_LIBRARY_PATH", if old.is_empty() { home.display().to_string() } else { format!("{}:{old}", home.display()) });
    }
}

// ------------------------------------------------------------------ assistant

fn wizard(env: &Env, gpu_ok: bool, quantus_present: bool) -> Value {
    println!("\n=== Vearl miner: first-time setup ===\n");
    println!("Vearl is a TESTNET: its tokens have no value and the software is not audited.");
    println!("By default this miner mines Vearl only, with the whole GPU: nothing else runs on your machine.\n");
    // 1. Quantus (optional add-on, off by default)
    let mut quantus = String::new();
    if gpu_ok && quantus_present {
        println!("OPTIONAL, QUANTUS (QTC): the Quantus add-on is installed in this folder, so this miner can also mine Quantus with your GPU");
        println!("while Vearl has no paid job for you. It is OFF unless you say yes now.");
        println!("  - You mine to YOUR OWN address; the Quantus pool pays you directly, Vearl never holds your funds.");
        println!("  - Create an address with the official Quantus wallet (it starts with \"qz\").");
        println!("  - Development fee: {} % of your Quantus mining time goes to the Vearl development address", qtc_fee_pct());
        println!("    {}", dev_qtc_address());
        println!("    (it is printed at every start, see docs/15 section 4 bis and docs/17).");
        println!("  - Fee audit: every 5 minutes the miner sends the Vearl seed nodes a signed message with your Vearl account, your Quantus");
        println!("    address and the versions of the programs, so the project can check that the fee worker runs. Nothing else, no secret.\n");
        if yes("Also mine Quantus with this miner?", false) {
            loop {
                let a = ask("Your Quantus address (qz...):", "");
                if valid_qtc_address(&a) {
                    quantus = a;
                    break;
                }
                println!("  This does not look like a Quantus address (\"qz\" followed by letters and digits). Try again, or press Ctrl+C.");
            }
        }
        println!();
    }
    // 2. Vearl wallet
    println!("VEARL wallet: it receives the Vearl block rewards and pays/receives for compute jobs.");
    let wallet_path = env.path("wallet.json");
    let mut wallet_file = "wallet.json".to_string();
    let existing = wallet_path.exists();
    // 0: use wallet.json as is, 1: create a new one, 2: point to another file
    let mode = if existing {
        println!("A wallet file already exists here ({}).", wallet_path.display());
        if yes("Use it?", true) {
            0
        } else if yes("Create ANOTHER wallet (the old file is kept as wallet.old.json)?", false) {
            1
        } else {
            2
        }
    } else if ask("Create a new wallet (1) or use an existing wallet file (2)?", "1") == "2" {
        2
    } else {
        1
    };
    if mode == 1 {
        if existing {
            let _ = std::fs::rename(&wallet_path, env.path("wallet.old.json"));
        }
        loop {
            let pw = password_prompt("Choose a wallet password (8 characters or more): ");
            if pw.len() < 8 {
                println!("  Too short.");
                continue;
            }
            if pw != password_prompt("Repeat the password: ") {
                println!("  The two passwords differ.");
                continue;
            }
            let m = new_mnemonic();
            let ks = Keystore::create(&m, &pw, KdfParams::STRONG).unwrap_or_else(|e| die(&e));
            let words: Vec<String> = m.to_string().split_whitespace().map(|w| w.to_string()).collect();
            println!("\nYour RECOVERY PHRASE (24 words). Write it on PAPER now: it is the only backup of your wallet and it is shown ONLY ONCE.");
            println!("Do not photograph it, do not store it in a file, do not send it to anybody.\n");
            for (i, w) in words.chunks(6).enumerate() {
                println!("  {:>2}-{:<2}  {}", i * 6 + 1, i * 6 + w.len(), w.join(" "));
            }
            println!();
            // check: three words asked at random, to be sure the phrase was written down
            let mut idx: Vec<usize> = Vec::new();
            while idx.len() < 3 {
                let k = rand::random::<usize>() % 24;
                if !idx.contains(&k) {
                    idx.push(k);
                }
            }
            idx.sort();
            let _ = ask("Press Enter once you have written the 24 words on paper.", "");
            loop {
                let mut ok = true;
                for &k in &idx {
                    if ask(&format!("Type word number {}:", k + 1), "").to_lowercase() != words[k] {
                        ok = false;
                    }
                }
                if ok {
                    break;
                }
                println!("  Wrong word(s): check your paper copy and try again.");
            }
            ks.save(&wallet_path).unwrap_or_else(|e| die(&e));
            clear_screen(); // the screen does not keep the words
            println!("Wallet created: {}", wallet_path.display());
            break;
        }
    } else if mode == 2 {
        loop {
            let f = ask("Path of your wallet file:", "");
            let p = PathBuf::from(f.trim_matches('"'));
            if Keystore::load(&p).is_ok() {
                wallet_file = p.display().to_string();
                break;
            }
            println!("  Not a Vearl wallet file. Try again.");
        }
    }
    let wpath = if Path::new(&wallet_file).is_absolute() { PathBuf::from(&wallet_file) } else { env.path(&wallet_file) };
    let address = Keystore::load(&wpath).map(|k| k.address).unwrap_or_default();
    println!("Your Vearl address: {address}\n");
    // 3. remembered password?
    let save_pw = yes("Remember the wallet password on this computer so the miner can restart without asking? (anybody who can read this folder could then spend your tokens)", false);
    if save_pw {
        let pw = password_prompt("Wallet password: ");
        if load_signing_key(&wpath, &pw, 0).is_err() {
            die("wrong wallet password");
        }
        write_private(&env.path("wallet.pw"), &pw);
    }
    let cfg = json!({
        "version": 1,
        "quantus_address": quantus,
        "vearl_wallet": wallet_file,
        "vearl_address": address,
        "account": 0,
        "partner_share": 80,
        "partner_cycle": 60,
        "worker": true,
        "max_jobs": 1,
        "remember_password": save_pw,
    });
    std::fs::write(env.path("miner.json"), serde_json::to_string_pretty(&cfg).unwrap()).unwrap_or_else(|e| die(&format!("cannot save miner.json: {e}")));
    println!("Settings saved in {}. Run \"vearl-miner setup\" to change them.\n", env.path("miner.json").display());
    cfg
}

fn clear_screen() {
    if WINDOWS {
        let _ = Command::new("cmd").args(["/c", "cls"]).status();
    } else {
        print!("\x1b[2J\x1b[H");
        let _ = io::stdout().flush();
    }
}

fn write_private(p: &Path, content: &str) {
    let _ = std::fs::write(p, content);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600));
    }
}

/// Linux: if the launcher disappears (kill -9, terminal closed), the node receives SIGTERM; it then stops the Quantus miner in turn
/// (safeguard of dual.rs). Windows: see `job` below.
#[cfg(target_os = "linux")]
extern "C" {
    fn prctl(option: i32, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> i32;
}

// ------------------------------------------------------------------ Windows: everything stops with the window

#[cfg(windows)]
mod job {
    use std::ffi::c_void;
    #[repr(C)]
    struct Basic {
        per_process_user_time: i64,
        per_job_user_time: i64,
        flags: u32,
        min_ws: usize,
        max_ws: usize,
        active_limit: u32,
        affinity: usize,
        priority: u32,
        scheduling: u32,
    }
    #[repr(C)]
    struct Io([u64; 6]);
    #[repr(C)]
    struct Extended {
        basic: Basic,
        io: Io,
        process_mem: usize,
        job_mem: usize,
        peak_process: usize,
        peak_job: usize,
    }
    extern "system" {
        fn CreateJobObjectW(attrs: *mut c_void, name: *const u16) -> *mut c_void;
        fn SetInformationJobObject(job: *mut c_void, class: i32, info: *mut c_void, len: u32) -> i32;
        fn AssignProcessToJobObject(job: *mut c_void, process: *mut c_void) -> i32;
    }
    /// Creates a "job" object whose closing kills all the assigned processes (hence also the Quantus miner, grandchild of the launcher).
    pub fn kill_on_close(child: &std::process::Child) -> *mut c_void {
        use std::os::windows::io::AsRawHandle;
        unsafe {
            let j = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
            if j.is_null() {
                return j;
            }
            let mut info: Extended = std::mem::zeroed();
            info.basic.flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            SetInformationJobObject(j, 9, &mut info as *mut _ as *mut c_void, std::mem::size_of::<Extended>() as u32);
            AssignProcessToJobObject(j, child.as_raw_handle() as *mut c_void);
            j
        }
    }
}

// ------------------------------------------------------------------ main

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |n: &str| args.iter().any(|a| a == n);
    if flag("--help") || flag("-h") {
        println!("vearl-miner [setup | check] [--home DIR] [--devnet]\n  (no argument)  start mining Vearl; the first time, an assistant creates your Vearl wallet (and offers the optional Quantus add-on, if installed)\n  setup          run the assistant again\n  check          test the GPU, the network files and the Quantus add-on (if installed), then stop");
        return;
    }
    // maintenance commands used by the release scripts (not shown in --help)
    if args.first().map(String::as_str) == Some("hash") {
        let f = args.get(1).unwrap_or_else(|| die("vearl-miner hash FILE"));
        println!("{}", vearl_core::feeguard::file_hash(Path::new(f)).unwrap_or_else(|e| die(&e)));
        return;
    }
    for a in &args {
        if a.starts_with("--") && !["--home", "--devnet", "--help"].contains(&a.as_str()) {
            die(&format!("unknown option {a}"));
        }
    }
    let home = args.iter().position(|a| a == "--home").and_then(|i| args.get(i + 1)).map(PathBuf::from).unwrap_or_else(|| {
        std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())).unwrap_or_else(|| PathBuf::from("."))
    });
    let devnet = flag("--devnet");
    let network = load_json(&home.join("network.json")).unwrap_or_else(|| {
        if devnet {
            json!({"name": "devnet", "genesis": null, "peers": [], "listen": "127.0.0.1:9000"})
        } else {
            die(&format!("{} is missing or unreadable: this folder is not a complete Vearl release (download it again from vearl.org)", home.join("network.json").display()))
        }
    });
    let env = Env { home, network };
    let cmd = {
        let mut it = args.iter();
        let mut c = String::new();
        while let Some(a) = it.next() {
            if a == "--home" {
                it.next();
            } else if !a.starts_with("--") {
                c = a.clone();
                break;
            }
        }
        c
    };

    println!("Vearl miner (testnet \"{}\"): tokens have no value, not audited.\n", env.network["name"].as_str().unwrap_or("?"));

    // --- diagnostic
    let gpu = gpu_status();
    match &gpu {
        Ok(d) => println!("GPU: {d}"),
        Err(e) => println!("GPU: not usable ({e}). Vearl will mine on the CPU."),
    }
    let has_quantus = quantus_present(&env);
    if has_quantus {
        println!("Quantus add-on: installed (optional, off unless you enabled it with \"vearl-miner setup\")");
    } else {
        println!("Quantus add-on: not installed (optional): this miner mines Vearl only");
    }
    let node_path = env.path(&exe("node"));
    if !node_path.exists() {
        die(&format!("{} is missing: this folder is not a complete Vearl release", node_path.display()));
    }
    let genesis = match env.network["genesis"].as_str() {
        Some(g) => {
            let p = env.path(g);
            if !p.exists() {
                die(&format!("{} is missing (network file of the testnet)", p.display()));
            }
            Some(p)
        }
        None if devnet => None,
        None => die("this release has no genesis file (pre-release build): it cannot join the testnet. Download the official release from vearl.org"),
    };
    if !devnet && env.network["peers"].as_array().map_or(true, |p| p.is_empty()) {
        die("this release lists no seed node (network.json has no peers): it would mine alone on a private chain. Download the official release from vearl.org");
    }
    println!("Network: {} ({} seed node(s))\n", env.network["name"].as_str().unwrap_or("?"), env.network["peers"].as_array().map_or(0, |p| p.len()));
    if cmd == "check" {
        if has_quantus && gpu.is_ok() {
            if let Err(e) = prepare_quantus(&env) {
                println!("Quantus add-on: {e}");
            }
        }
        println!("Check done.");
        return;
    }

    // --- settings
    let cfg_path = env.path("miner.json");
    let mut cfg = if cmd == "setup" { None } else { load_json(&cfg_path) };
    if cfg.is_none() {
        cfg = Some(wizard(&env, gpu.is_ok(), has_quantus));
    }
    let cfg = cfg.unwrap();
    let wallet_file = cfg["vearl_wallet"].as_str().unwrap_or("wallet.json").to_string();
    let wallet = if Path::new(&wallet_file).is_absolute() { PathBuf::from(&wallet_file) } else { env.path(&wallet_file) };
    let account = cfg["account"].as_u64().unwrap_or(0) as u32;
    // Quantus mining only if the user enabled it (an address in miner.json) AND the add-on works on this GPU; the add-on is not even
    // started otherwise.
    let quantus_addr = cfg["quantus_address"].as_str().unwrap_or("").to_string();
    let mut qm: Result<(PathBuf, bool), String> = Err(String::new());
    if !quantus_addr.is_empty() {
        qm = if !gpu.is_ok() {
            Err("no usable GPU".to_string())
        } else if !has_quantus {
            Err("the Quantus add-on is not installed in this folder".to_string())
        } else {
            prepare_quantus(&env)
        };
    }
    let use_quantus = qm.is_ok();
    if let (false, Err(e)) = (quantus_addr.is_empty(), &qm) {
        println!("Note: Quantus mining is OFF in this session ({e}); your Quantus address stays saved. Vearl keeps mining.\n");
    }

    // --- password
    let mut pw = String::new();
    let pw_file = env.path("wallet.pw");
    if cfg["remember_password"].as_bool().unwrap_or(false) && pw_file.exists() {
        pw = std::fs::read_to_string(&pw_file).unwrap_or_default().trim_end_matches(['\r', '\n']).to_string();
    }
    for attempt in 0..4 {
        if pw.is_empty() {
            pw = password_prompt("Wallet password: ");
        }
        match load_signing_key(&wallet, &pw, account) {
            Ok(_) => break,
            Err(_) if attempt < 3 => {
                println!("Wrong password.");
                pw.clear();
            }
            Err(e) => die(&format!("cannot unlock the wallet {}: {e}", wallet.display())),
        }
    }

    // --- node command
    let mut a: Vec<String> = Vec::new();
    if let Some(g) = &genesis {
        a.extend(["--genesis".into(), g.display().to_string()]);
    }
    for p in env.network["peers"].as_array().cloned().unwrap_or_default() {
        if let Some(p) = p.as_str() {
            a.extend(["--peer".into(), p.into()]);
        }
    }
    a.extend(["--listen".into(), env.network["listen"].as_str().unwrap_or("0.0.0.0:9000").into()]);
    a.extend(["--miner-wallet".into(), wallet.display().to_string(), "--miner-account".into(), account.to_string()]);
    if gpu.is_ok() {
        a.push("--gpu".into());
    }
    if cfg["worker"].as_bool().unwrap_or(true) {
        a.extend(["--worker".into(), "--max-jobs".into(), cfg["max_jobs"].as_u64().unwrap_or(1).to_string()]);
    }
    a.extend(["--data-dir".into(), env.path("data").display().to_string(), "--log-file".into(), env.path("miner.log").display().to_string()]);
    if let Ok((qpath, quantus_cuda)) = &qm {
        a.extend([
            "--qtc-miner".into(),
            qpath.display().to_string(),
            "--qtc-address".into(),
            quantus_addr.clone(),
            "--qtc-pool".into(),
            env.network["quantus_pool"].as_str().unwrap_or(QTC_POOL_ADDR).into(),
            "--qtc-cert".into(),
            env.network["quantus_cert"].as_str().unwrap_or(QTC_POOL_CERT).into(),
            "--qtc-engine".into(),
            if *quantus_cuda { "cuda".to_string() } else { "vulkan".to_string() },
            "--partner-share".into(),
            cfg["partner_share"].as_u64().unwrap_or(80).to_string(),
            "--partner-cycle".into(),
            cfg["partner_cycle"].as_u64().unwrap_or(60).to_string(),
        ]);
    }

    println!("Starting the miner.");
    println!("  Vearl address:   {}", cfg["vearl_address"].as_str().unwrap_or("?"));
    if use_quantus {
        println!("  Quantus address: {quantus_addr}");
        println!("  GPU time:        {} % Quantus / {} % Vearl, and the whole GPU goes to Vearl as soon as a paid job is yours", cfg["partner_share"].as_u64().unwrap_or(80), 100 - cfg["partner_share"].as_u64().unwrap_or(80).min(100));
        println!("  Dev fee:         {} % of the Quantus mining time -> {}", qtc_fee_pct(), dev_qtc_address());
    } else {
        println!("  Mining:          Vearl only");
    }
    println!("  Log:             {}  (status line every 10 s; Ctrl+C or closing this window stops everything)\n", env.path("miner.log").display());

    let mut c = Command::new(&node_path);
    c.args(&a).env("VEARL_WALLET_PASSWORD", &pw).current_dir(&env.home);
    apply_lib_path(&mut c, &env.home);
    let mut child = c.spawn().unwrap_or_else(|e| die(&format!("cannot start {}: {e}", node_path.display())));
    #[cfg(windows)]
    let _job = job::kill_on_close(&child);
    let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(1);
    if code != 0 {
        eprintln!("\nThe miner stopped (exit code {code}). See {} for details.", env.path("miner.log").display());
        pause_on_windows();
    }
    std::process::exit(code)
}
