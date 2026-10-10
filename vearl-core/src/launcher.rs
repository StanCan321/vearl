//! Start-up helpers shared by the command-line programs: a real help text when the program is run without arguments, and, on Windows,
//! a pause before the window closes when the program was started by a double-click (otherwise the window flashes and disappears
//! before anybody can read it).
use std::io::{self, BufRead};

#[cfg(windows)]
mod sys {
    extern "system" {
        pub fn GetConsoleProcessList(list: *mut u32, count: u32) -> u32;
    }
}

/// True when the program was started by a double-click in the file explorer: it is then the only process attached to its console
/// window. From a terminal (cmd, PowerShell, Windows Terminal, a script) the shell is attached too. Always false outside Windows,
/// where a terminal is always there.
pub fn started_by_double_click() -> bool {
    #[cfg(windows)]
    {
        let mut list = [0u32; 4];
        // SAFETY: the buffer is valid for `count` entries; the call only writes process ids into it.
        let n = unsafe { sys::GetConsoleProcessList(list.as_mut_ptr(), list.len() as u32) };
        n == 1
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// Waits for Enter before returning, only when the program was started by a double-click.
pub fn pause_if_double_clicked() {
    if started_by_double_click() {
        eprintln!("\nPress Enter to close this window.");
        let mut s = String::new();
        let _ = io::stdin().lock().read_line(&mut s);
    }
}

const DOUBLE_CLICK_NOTE: &str = "\nThis is a command-line program: it has no window of its own, so it closes as soon as it has printed this text.\n\
Open a terminal in this folder (Windows: click the address bar of the folder, type cmd and press Enter), then type one of the commands above.\n\
To start mining, double-click vearl-miner.exe instead.";

/// Call first in `main`. When the program is run without arguments, or with `--help`, `-h` or `help`, prints `text` and exits
/// (code 0 when help was asked for or the program was double-clicked, 2 for a plain run without arguments, as a usage error).
/// A double-clicked program also explains how to use it and waits for Enter. Otherwise returns and the program goes on.
pub fn help_or_continue(text: &str) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let asked = matches!(args.first().map(String::as_str), Some("--help" | "-h" | "help"));
    if !args.is_empty() && !asked {
        return;
    }
    println!("{text}");
    let clicked = started_by_double_click();
    if clicked {
        println!("{DOUBLE_CLICK_NOTE}");
        pause_if_double_clicked();
    }
    std::process::exit(if asked || clicked { 0 } else { 2 });
}

/// For a program that does something useful without arguments (the node starts a development chain): explains itself and waits only
/// when it was started by a double-click, and otherwise lets the program go on.
pub fn explain_if_double_clicked(text: &str) {
    if started_by_double_click() && std::env::args().count() == 1 {
        println!("{text}");
        println!("{DOUBLE_CLICK_NOTE}");
        pause_if_double_clicked();
        std::process::exit(0);
    }
}
