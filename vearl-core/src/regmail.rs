//! Registration for the testnet rewards by e-mail (docs/19, section 5).
//!
//! `vearl-wallet register` signs the registration message with the wallet and sends it, after the user has confirmed, from the dedicated address
//! `registrar@vearl.org` to `register@vearl.org`, over TLS (port 465 of mail.vearl.org). A script on the server reads that mailbox, checks the
//! signature and publishes the list of registered addresses (deploy/rewards/).
//!
//! **The SMTP account below is PUBLIC on purpose**: it is in the source, so anybody can read it. It is harmless by design:
//!   * the server lets it send to `register@vearl.org` ONLY, on a dedicated port, with a small size limit and a rate limit;
//!   * it has no mailbox (it cannot read anything), and nobody can register somebody else's address: without the wallet key there is no valid
//!     signature, and the script ignores every message whose signature is not valid;
//!   * what is sent is not secret: an address, a message and a signature (a signature proves ownership without revealing any secret).
//! The worst that anybody can do with it is to send noise to `register@`, which the limits and the script absorb.
//!
//! The SMTP dialogue is written against `Read + Write`, so it is tested without a network (see the tests); only the TLS layer needs the `mail`
//! feature (rustls with the `ring` provider and the Mozilla trust anchors: the certificate of mail.vearl.org is checked).
use base64::Engine;
use std::io::{Read, Write};

pub const SMTP_HOST: &str = "mail.vearl.org";
pub const SMTP_PORT: u16 = 465;
pub const SENDER: &str = "registrar@vearl.org";
pub const RECIPIENT: &str = "register@vearl.org";
pub const SMTP_USER: &str = "registrar@vearl.org";
/// Public by design (see the module documentation).
pub const SMTP_PASSWORD: &str = "Bqxij9OWb-5GtGpmDlO-mP5LDIGPf3gB";
/// Where a registered address shows up once the server has processed the mail (public list of addresses, no contact details).
pub const PUBLIC_LIST_URL: &str = "https://vearl.org/rewards/registered.json";
const TIMEOUT_SECS: u64 = 20;
const MAX_CONTACT: usize = 80;

/// The exact text that is signed (docs/19, and `tools/verify_registration.py` checks the same form).
pub fn registration_message(address: &str, contact: &str) -> String {
    format!("Vearl testnet rewards registration. Address: {address}. Contact: {contact}. I accept the terms in docs/19.")
}

/// The contact is a Discord name or an e-mail address: one printable line, no control characters.
pub fn check_contact(contact: &str) -> Result<(), String> {
    let c = contact.trim();
    if c.is_empty() {
        return Err("the contact is empty: give your Discord name or an e-mail address".into());
    }
    if c != contact {
        return Err("the contact must not start or end with a space".into());
    }
    if c.chars().count() > MAX_CONTACT {
        return Err(format!("the contact is too long (at most {MAX_CONTACT} characters)"));
    }
    if c.chars().any(|ch| ch.is_control()) {
        return Err("the contact must be a single line of plain text".into());
    }
    Ok(())
}

pub struct Registration {
    pub address: String,
    pub message: String,
    pub signature_hex: String,
}

pub struct Mail {
    pub from: String,
    pub to: String,
    pub subject: String,
    pub body: String,
}

/// The e-mail that carries a registration: three labelled lines that the server script parses.
pub fn registration_mail(r: &Registration) -> Mail {
    Mail {
        from: SENDER.to_string(),
        to: RECIPIENT.to_string(),
        subject: "Vearl testnet registration".to_string(),
        body: format!("Vearl testnet rewards registration (docs/19)\r\naddress: {}\r\nmessage: {}\r\nsignature: {}\r\n", r.address, r.message, r.signature_hex),
    }
}

fn civil_date(t: u64) -> (u64, u64, u64, u64, u64, u64, usize) {
    // days since 1970-01-01 to (year, month, day), weekday: Howard Hinnant's algorithm
    let days = (t / 86400) as i64;
    let secs = t % 86400;
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    let wd = ((days + 4).rem_euclid(7)) as usize; // 1970-01-01 was a Thursday
    (y as u64, m as u64, d as u64, secs / 3600, (secs % 3600) / 60, secs % 60, wd)
}

impl Mail {
    /// RFC 5322 text with CRLF line ends. `now` (Unix seconds) and `id` (random hex) make the Date and Message-ID headers.
    pub fn to_wire(&self, now: u64, id: &str) -> String {
        const WD: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
        const MO: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        let (y, mo, d, h, mi, s, wd) = civil_date(now);
        let wd = WD[wd % 7];
        let head = format!(
            "From: {}\r\nTo: {}\r\nSubject: {}\r\nDate: {wd}, {d:02} {} {y} {h:02}:{mi:02}:{s:02} +0000\r\nMessage-ID: <{id}@vearl.org>\r\nMIME-Version: 1.0\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: 7bit\r\n\r\n",
            self.from,
            self.to,
            self.subject,
            MO[(mo as usize - 1) % 12]
        );
        let mut out = head;
        // normalise the body to CRLF
        let body = self.body.replace("\r\n", "\n").replace('\n', "\r\n");
        out.push_str(&body);
        if !out.ends_with("\r\n") {
            out.push_str("\r\n");
        }
        out
    }
}

/// One reply of the server: the code and the text of the last line (multi-line replies are read to their end).
fn read_reply<S: Read>(s: &mut S) -> Result<(u16, String), String> {
    loop {
        let mut line = Vec::new();
        let mut b = [0u8; 1];
        loop {
            match s.read(&mut b) {
                Ok(0) => return Err("the server closed the connection".into()),
                Ok(_) => {
                    if b[0] == b'\n' {
                        break;
                    }
                    if b[0] != b'\r' {
                        line.push(b[0]);
                    }
                    if line.len() > 2000 {
                        return Err("the server sent a line that is too long".into());
                    }
                }
                Err(e) => return Err(format!("network error while reading: {e}")),
            }
        }
        let text = String::from_utf8_lossy(&line).to_string();
        if text.len() < 3 || !text.is_char_boundary(3) {
            return Err(format!("unexpected reply from the server: {text:?}"));
        }
        let code: u16 = text[..3].parse().map_err(|_| format!("unexpected reply from the server: {text:?}"))?;
        let more = text.as_bytes().get(3) == Some(&b'-');
        if !more {
            return Ok((code, text.get(4..).unwrap_or("").to_string()));
        }
    }
}

fn expect<S: Read>(s: &mut S, want: u16, what: &str) -> Result<(), String> {
    let (code, text) = read_reply(s)?;
    if code == want {
        Ok(())
    } else {
        Err(format!("{what}: the server answered {code} {text}"))
    }
}

fn send_line<S: Write>(s: &mut S, line: &str) -> Result<(), String> {
    s.write_all(line.as_bytes()).and_then(|_| s.write_all(b"\r\n")).and_then(|_| s.flush()).map_err(|e| format!("network error while writing: {e}"))
}

/// The SMTP dialogue (RFC 5321) over an already secured stream: EHLO, AUTH PLAIN, MAIL FROM, RCPT TO, DATA, QUIT.
pub fn smtp_dialog<S: Read + Write>(s: &mut S, user: &str, pass: &str, mail: &Mail, now: u64, id: &str) -> Result<(), String> {
    expect(s, 220, "greeting")?;
    send_line(s, "EHLO vearl-wallet")?;
    expect(s, 250, "EHLO")?;
    let auth = base64::engine::general_purpose::STANDARD.encode(format!("\0{user}\0{pass}"));
    send_line(s, &format!("AUTH PLAIN {auth}"))?;
    expect(s, 235, "authentication")?;
    send_line(s, &format!("MAIL FROM:<{}>", mail.from))?;
    expect(s, 250, "sender")?;
    send_line(s, &format!("RCPT TO:<{}>", mail.to))?;
    expect(s, 250, "recipient")?;
    send_line(s, "DATA")?;
    expect(s, 354, "DATA")?;
    let wire = mail.to_wire(now, id);
    let mut stuffed = String::with_capacity(wire.len() + 8);
    for line in wire.split("\r\n") {
        if line.starts_with('.') {
            stuffed.push('.'); // dot-stuffing
        }
        stuffed.push_str(line);
        stuffed.push_str("\r\n");
    }
    // `split` leaves an empty last element when the text ends with CRLF: do not add one more empty line
    if stuffed.ends_with("\r\n\r\n") {
        stuffed.truncate(stuffed.len() - 2);
    }
    s.write_all(stuffed.as_bytes()).and_then(|_| s.write_all(b".\r\n")).and_then(|_| s.flush()).map_err(|e| format!("network error while writing: {e}"))?;
    expect(s, 250, "message")?;
    let _ = send_line(s, "QUIT");
    Ok(())
}

fn now_and_id() -> (u64, String) {
    use rand::RngCore;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mut r = [0u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut r);
    (now, r.iter().map(|b| format!("{b:02x}")).collect())
}

/// Sends the registration over TLS (certificate of mail.vearl.org checked).
#[cfg(feature = "mail")]
pub fn send_registration(r: &Registration) -> Result<(), String> {
    use std::net::{TcpStream, ToSocketAddrs};
    use std::sync::Arc;
    use std::time::Duration;
    let mail = registration_mail(r);
    let addrs: Vec<_> = (SMTP_HOST, SMTP_PORT).to_socket_addrs().map_err(|e| format!("cannot find {SMTP_HOST}: {e}"))?.collect();
    let mut tcp = None;
    let mut last_err = String::from("no address");
    for a in addrs {
        match TcpStream::connect_timeout(&a, Duration::from_secs(TIMEOUT_SECS)) {
            Ok(t) => {
                tcp = Some(t);
                break;
            }
            Err(e) => last_err = format!("{a}: {e}"),
        }
    }
    let tcp = tcp.ok_or_else(|| format!("cannot connect to {SMTP_HOST}:{SMTP_PORT} ({last_err}). Some networks block outgoing mail: use the manual way instead"))?;
    tcp.set_read_timeout(Some(Duration::from_secs(TIMEOUT_SECS))).ok();
    tcp.set_write_timeout(Some(Duration::from_secs(TIMEOUT_SECS))).ok();
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let cfg = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| format!("TLS setup: {e}"))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    let name = rustls::pki_types::ServerName::try_from(SMTP_HOST.to_string()).map_err(|e| format!("TLS name: {e}"))?;
    let conn = rustls::ClientConnection::new(Arc::new(cfg), name).map_err(|e| format!("TLS setup: {e}"))?;
    let mut tls = rustls::StreamOwned::new(conn, tcp);
    let (now, id) = now_and_id();
    smtp_dialog(&mut tls, SMTP_USER, SMTP_PASSWORD, &mail, now, &id)
}

#[cfg(not(feature = "mail"))]
pub fn send_registration(_r: &Registration) -> Result<(), String> {
    Err("this build of the wallet cannot send e-mail".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::{TcpListener, TcpStream};

    /// A scripted SMTP server on the loopback: records every command line and the DATA, answers with the given codes.
    fn fake_server(auth_ok: bool, rcpt_ok: bool) -> (u16, std::thread::JoinHandle<(Vec<String>, String)>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let h = std::thread::spawn(move || {
            let (s, _) = l.accept().unwrap();
            let mut w = s.try_clone().unwrap();
            let mut r = BufReader::new(s);
            let mut cmds = Vec::new();
            let mut data = String::new();
            w.write_all(b"220 mail.vearl.org ESMTP\r\n").unwrap();
            let mut line = String::new();
            loop {
                line.clear();
                if r.read_line(&mut line).unwrap() == 0 {
                    break;
                }
                let l = line.trim_end().to_string();
                cmds.push(l.clone());
                if l.starts_with("EHLO") {
                    w.write_all(b"250-mail.vearl.org\r\n250-PIPELINING\r\n250 AUTH PLAIN LOGIN\r\n").unwrap();
                } else if l.starts_with("AUTH PLAIN") {
                    w.write_all(if auth_ok { b"235 2.7.0 Authentication successful\r\n" as &[u8] } else { b"535 5.7.8 Authentication failed\r\n" }).unwrap();
                    if !auth_ok {
                        break;
                    }
                } else if l.starts_with("MAIL FROM") {
                    w.write_all(b"250 2.1.0 Ok\r\n").unwrap();
                } else if l.starts_with("RCPT TO") {
                    w.write_all(if rcpt_ok { b"250 2.1.5 Ok\r\n" as &[u8] } else { b"554 5.7.1 Relay access denied\r\n" }).unwrap();
                    if !rcpt_ok {
                        break;
                    }
                } else if l == "DATA" {
                    w.write_all(b"354 End data with <CR><LF>.<CR><LF>\r\n").unwrap();
                    loop {
                        line.clear();
                        r.read_line(&mut line).unwrap();
                        if line == ".\r\n" {
                            break;
                        }
                        data.push_str(&line);
                    }
                    w.write_all(b"250 2.0.0 Ok: queued as ABC\r\n").unwrap();
                } else if l == "QUIT" {
                    w.write_all(b"221 Bye\r\n").unwrap();
                    break;
                }
            }
            (cmds, data)
        });
        (port, h)
    }

    fn sample() -> Registration {
        let address = "vrl1testaddress".to_string();
        Registration { message: registration_message(&address, "clovis#1234"), address, signature_hex: "ab".repeat(64) }
    }

    #[test]
    fn the_message_has_the_form_that_the_server_script_checks() {
        let m = registration_message("vrl1abc", "name@example.org");
        assert_eq!(m, "Vearl testnet rewards registration. Address: vrl1abc. Contact: name@example.org. I accept the terms in docs/19.");
    }

    #[test]
    fn the_contact_is_checked() {
        assert!(check_contact("clovis#1234").is_ok());
        assert!(check_contact("name@example.org").is_ok());
        assert!(check_contact("").is_err());
        assert!(check_contact(" x").is_err());
        assert!(check_contact("two\nlines").is_err());
        assert!(check_contact(&"x".repeat(81)).is_err());
    }

    #[test]
    fn the_dialogue_sends_one_mail_from_the_registrar_to_the_register_address() {
        let (port, h) = fake_server(true, true);
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mail = registration_mail(&sample());
        smtp_dialog(&mut s, "registrar@vearl.org", "pw", &mail, 1_792_771_200, "deadbeef").unwrap();
        let (cmds, data) = h.join().unwrap();
        assert_eq!(cmds[0], "EHLO vearl-wallet");
        let auth = base64::engine::general_purpose::STANDARD.encode("\0registrar@vearl.org\0pw");
        assert_eq!(cmds[1], format!("AUTH PLAIN {auth}"));
        assert_eq!(cmds[2], "MAIL FROM:<registrar@vearl.org>");
        assert_eq!(cmds[3], "RCPT TO:<register@vearl.org>");
        assert_eq!(cmds[4], "DATA");
        assert!(data.contains("From: registrar@vearl.org\r\n"));
        assert!(data.contains("To: register@vearl.org\r\n"));
        assert!(data.contains("Date: Fri, 23 Oct 2026 16:00:00 +0000\r\n"), "{data}");
        assert!(data.contains("address: vrl1testaddress\r\n"));
        assert!(data.contains(&format!("signature: {}\r\n", "ab".repeat(64))));
        assert_eq!(cmds.last().unwrap(), "QUIT");
    }

    #[test]
    fn a_line_that_starts_with_a_dot_is_stuffed() {
        let (port, h) = fake_server(true, true);
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mail = Mail { from: SENDER.into(), to: RECIPIENT.into(), subject: "t".into(), body: "a\n.b\n..c\n".into() };
        smtp_dialog(&mut s, "u", "p", &mail, 0, "x").unwrap();
        let (_, data) = h.join().unwrap();
        assert!(data.contains("\r\n..b\r\n"));
        assert!(data.contains("\r\n...c\r\n"));
    }

    #[test]
    fn a_refused_authentication_is_reported() {
        let (port, h) = fake_server(false, true);
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let e = smtp_dialog(&mut s, "u", "bad", &registration_mail(&sample()), 0, "x").unwrap_err();
        assert!(e.contains("authentication") && e.contains("535"), "{e}");
        h.join().unwrap();
    }

    #[test]
    fn a_refused_recipient_is_reported() {
        let (port, h) = fake_server(true, false);
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let e = smtp_dialog(&mut s, "u", "p", &registration_mail(&sample()), 0, "x").unwrap_err();
        assert!(e.contains("recipient") && e.contains("554"), "{e}");
        h.join().unwrap();
    }

    #[test]
    fn the_date_header_is_right() {
        // 2026-10-23 16:00:00 UTC is a Friday; 2000-02-29 12:34:56 is a Tuesday
        let m = Mail { from: "a@b".into(), to: "c@d".into(), subject: "s".into(), body: "x".into() };
        assert!(m.to_wire(1_792_771_200, "i").contains("Date: Fri, 23 Oct 2026 16:00:00 +0000"));
        assert!(m.to_wire(951_827_696, "i").contains("Date: Tue, 29 Feb 2000 12:34:56 +0000"));
    }
}
