//! What only the operating system can do for the window: a title bar in the colours of the wallet (Windows), hiding the window from
//! screenshots and recordings while the recovery words are shown (Windows), and no memory dumps of the process (Linux).
use raw_window_handle::HasWindowHandle;

#[cfg(windows)]
mod win {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};

    #[link(name = "dwmapi")]
    extern "system" {
        fn DwmSetWindowAttribute(hwnd: isize, attr: u32, value: *const core::ffi::c_void, size: u32) -> i32;
    }
    #[link(name = "user32")]
    extern "system" {
        fn SetWindowDisplayAffinity(hwnd: isize, affinity: u32) -> i32;
    }

    fn hwnd(h: &impl HasWindowHandle) -> Option<isize> {
        match h.window_handle().ok()?.as_raw() {
            RawWindowHandle::Win32(w) => Some(w.hwnd.get()),
            _ => None,
        }
    }

    /// COLORREF is 0x00BBGGRR.
    fn colorref(r: u8, g: u8, b: u8) -> u32 {
        (b as u32) << 16 | (g as u32) << 8 | r as u32
    }

    fn set(hwnd: isize, attr: u32, v: u32) {
        // SAFETY: `hwnd` is the handle of our own window and the value is a 4-byte integer, as the attribute requires; the call only
        // reads it. Windows that do not know the attribute simply return an error, which is ignored.
        unsafe {
            DwmSetWindowAttribute(hwnd, attr, &v as *const u32 as *const core::ffi::c_void, 4);
        }
    }

    pub fn style(h: &impl HasWindowHandle) {
        let Some(w) = hwnd(h) else { return };
        set(w, 20, 1); // DWMWA_USE_IMMERSIVE_DARK_MODE
        set(w, 35, colorref(0x07, 0x08, 0x0b)); // DWMWA_CAPTION_COLOR (Windows 11)
        set(w, 36, colorref(0xec, 0xeb, 0xe6)); // DWMWA_TEXT_COLOR
        set(w, 34, colorref(0x1b, 0x20, 0x29)); // DWMWA_BORDER_COLOR
    }

    pub fn capture(h: &impl HasWindowHandle, hide: bool) {
        let Some(w) = hwnd(h) else { return };
        // WDA_EXCLUDEFROMCAPTURE (0x11): the window is not in screenshots or recordings (Windows 10 2004 and later). SAFETY: own window.
        unsafe {
            SetWindowDisplayAffinity(w, if hide { 0x11 } else { 0 });
        }
    }
}

/// Colours the title bar like the wallet.
pub fn style_window(h: &impl HasWindowHandle) {
    #[cfg(windows)]
    win::style(h);
    #[cfg(not(windows))]
    let _ = h;
}

/// Hides the window from screen capture (or shows it again). Windows only.
pub fn capture_protection(h: &impl HasWindowHandle, hide: bool) {
    #[cfg(windows)]
    win::capture(h, hide);
    #[cfg(not(windows))]
    let _ = (h, hide);
}

/// Linux: the process cannot be dumped or inspected by another process of the same user (no core file, no ptrace attach).
pub fn harden_process() {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: plain system calls with constant arguments, no pointers other than the valid `rlimit` below.
        unsafe {
            libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0);
            let zero = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
            libc::setrlimit(libc::RLIMIT_CORE, &zero);
        }
    }
}
