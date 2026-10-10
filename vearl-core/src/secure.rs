//! Encrypted and authenticated channel between peers: Noise_XX_25519_ChaChaPoly_BLAKE2s through the `snow` library.
//! No home-made primitive: a standard protocol is assembled (XX pattern = mutual authentication of the static
//! keys, forward secrecy after compromise of long-term keys) on top of an existing library.
//!
//! What this provides: confidentiality + integrity of the stream, identity of a peer = its static X25519 key
//! (the key of a known peer can be PINNED). What it does NOT provide: it does not prevent an attacker from
//! multiplying its own identities (Sybil); network anonymity (the IP stays visible); no external review.
use snow::{Builder, TransportState};
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const PATTERN: &str = "Noise_XX_25519_ChaChaPoly_BLAKE2s";
const MAX_FRAME: usize = 65535;
const TAG: usize = 16;
/// Maximum size of an application message (before encryption).
pub const MAX_APP_MSG: usize = 16 << 20;

fn ioerr<E: ToString>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

#[derive(Clone)]
pub struct Keys {
    pub private: [u8; 32],
    pub public: [u8; 32],
}

impl Keys {
    pub fn generate() -> Keys {
        let kp = Builder::new(PATTERN.parse().unwrap()).generate_keypair().expect("keygen");
        Keys { private: kp.private.try_into().unwrap(), public: kp.public.try_into().unwrap() }
    }

    /// Loads the node's keys from `path` (64 bytes: private || public), or creates them (0600 on Unix).
    pub fn load_or_create(path: &Path) -> Keys {
        if let Ok(b) = std::fs::read(path) {
            if b.len() == 64 {
                return Keys { private: b[..32].try_into().unwrap(), public: b[32..].try_into().unwrap() };
            }
        }
        let k = Keys::generate();
        if let Some(d) = path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let mut raw = k.private.to_vec();
        raw.extend_from_slice(&k.public);
        let _ = std::fs::write(path, raw);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        k
    }

    pub fn public_hex(&self) -> String {
        self.public.iter().map(|b| format!("{b:02x}")).collect()
    }
}

pub struct SecureReader {
    stream: TcpStream,
    st: Arc<Mutex<TransportState>>,
    buf: Vec<u8>,
}

pub struct SecureWriter {
    stream: TcpStream,
    st: Arc<Mutex<TransportState>>,
}

fn read_frame(s: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut l = [0u8; 2];
    s.read_exact(&mut l)?;
    let n = u16::from_be_bytes(l) as usize;
    let mut b = vec![0u8; n];
    s.read_exact(&mut b)?;
    Ok(b)
}

fn write_frame(s: &mut TcpStream, b: &[u8]) -> io::Result<()> {
    s.write_all(&(b.len() as u16).to_be_bytes())?;
    s.write_all(b)
}

/// Noise XX handshake. Returns (reader, writer, authenticated static key of the peer).
pub fn handshake(mut stream: TcpStream, keys: &Keys, initiator: bool, timeout: Duration) -> io::Result<(SecureReader, SecureWriter, [u8; 32])> {
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let b = Builder::new(PATTERN.parse().map_err(ioerr)?).local_private_key(&keys.private);
    let mut hs = if initiator { b.build_initiator() } else { b.build_responder() }.map_err(ioerr)?;
    let mut out = vec![0u8; MAX_FRAME];
    let mut scratch = vec![0u8; MAX_FRAME];
    if initiator {
        let n = hs.write_message(&[], &mut out).map_err(ioerr)?; // -> e
        write_frame(&mut stream, &out[..n])?;
        let m = read_frame(&mut stream)?; // <- e, ee, s, es
        hs.read_message(&m, &mut scratch).map_err(ioerr)?;
        let n = hs.write_message(&[], &mut out).map_err(ioerr)?; // -> s, se
        write_frame(&mut stream, &out[..n])?;
    } else {
        let m = read_frame(&mut stream)?;
        hs.read_message(&m, &mut scratch).map_err(ioerr)?;
        let n = hs.write_message(&[], &mut out).map_err(ioerr)?;
        write_frame(&mut stream, &out[..n])?;
        let m = read_frame(&mut stream)?;
        hs.read_message(&m, &mut scratch).map_err(ioerr)?;
    }
    if !hs.is_handshake_finished() {
        return Err(ioerr("handshake incomplete"));
    }
    let remote: [u8; 32] = hs.get_remote_static().and_then(|s| <[u8; 32]>::try_from(s).ok()).ok_or_else(|| ioerr("no remote static key"))?;
    let st = Arc::new(Mutex::new(hs.into_transport_mode().map_err(ioerr)?));
    let w = stream.try_clone()?;
    Ok((SecureReader { stream, st: st.clone(), buf: Vec::new() }, SecureWriter { stream: w, st }, remote))
}

impl SecureReader {
    pub fn set_read_timeout(&self, d: Option<Duration>) -> io::Result<()> {
        self.stream.set_read_timeout(d)
    }

    /// Reads a complete application message (the encrypted bytes are all authenticated; any alteration => error).
    pub fn read_msg_bytes(&mut self) -> io::Result<Vec<u8>> {
        loop {
            if self.buf.len() >= 4 {
                let n = u32::from_le_bytes(self.buf[..4].try_into().unwrap()) as usize;
                if n > MAX_APP_MSG {
                    return Err(ioerr("message too large"));
                }
                if self.buf.len() >= 4 + n {
                    let msg = self.buf[4..4 + n].to_vec();
                    self.buf.drain(..4 + n);
                    return Ok(msg);
                }
            }
            let ct = read_frame(&mut self.stream)?;
            let mut pt = vec![0u8; ct.len()];
            let n = self.st.lock().unwrap().read_message(&ct, &mut pt).map_err(ioerr)?;
            self.buf.extend_from_slice(&pt[..n]);
        }
    }

    pub fn shutdown(&self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}

impl SecureWriter {
    pub fn write_msg_bytes(&mut self, payload: &[u8]) -> io::Result<()> {
        if payload.len() > MAX_APP_MSG {
            return Err(ioerr("message too large"));
        }
        let mut plain = Vec::with_capacity(4 + payload.len());
        plain.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        plain.extend_from_slice(payload);
        let mut wire = Vec::with_capacity(plain.len() + (plain.len() / (MAX_FRAME - TAG) + 1) * (TAG + 2));
        let mut ct = vec![0u8; MAX_FRAME];
        for chunk in plain.chunks(MAX_FRAME - TAG) {
            let n = self.st.lock().unwrap().write_message(chunk, &mut ct).map_err(ioerr)?;
            wire.extend_from_slice(&(n as u16).to_be_bytes());
            wire.extend_from_slice(&ct[..n]);
        }
        self.stream.write_all(&wire)?;
        self.stream.flush()
    }

    pub fn shutdown(&self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}
