//! A very small HTTP(S) GET client: the window only needs to read two public JSON files (the activity list of the block explorer and
//! the list of registered addresses). HTTP/1.0 with `Connection: close` (no chunked encoding to decode), a size limit and timeouts.
//! TLS is rustls with the Mozilla trust anchors, like the registration e-mail of the command-line wallet.
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

const MAX_BODY: usize = 6 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(8);

/// (https?, host, port, path)
pub fn split_url(url: &str) -> Result<(bool, String, u16, String), String> {
    let (tls, rest) = if let Some(r) = url.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = url.strip_prefix("http://") {
        (false, r)
    } else {
        return Err("the address must start with https:// (or http://)".into());
    };
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].to_string()),
        None => (rest, "/".to_string()),
    };
    if hostport.is_empty() || hostport.contains('@') || hostport.contains(' ') {
        return Err("invalid address".into());
    }
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse::<u16>().map_err(|_| "invalid port".to_string())?),
        None => (hostport.to_string(), if tls { 443 } else { 80 }),
    };
    if host.is_empty() || !host.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-')) {
        return Err("invalid host name".into());
    }
    Ok((tls, host, port, path))
}

/// Splits a raw HTTP reply into (status code, body). Only 200 is accepted by `get`.
pub fn split_reply(raw: &[u8]) -> Result<(u16, &[u8]), String> {
    let end = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or("incomplete reply")?;
    let head = std::str::from_utf8(&raw[..end]).map_err(|_| "unreadable reply")?;
    let status = head.lines().next().and_then(|l| l.split_whitespace().nth(1)).and_then(|c| c.parse::<u16>().ok()).ok_or("unreadable status")?;
    Ok((status, &raw[end + 4..]))
}

pub fn get(url: &str) -> Result<String, String> {
    let (tls, host, port, path) = split_url(url)?;
    let addr = (host.as_str(), port).to_socket_addrs().map_err(|e| format!("{host}: {e}"))?.next().ok_or_else(|| format!("{host}: no address"))?;
    let tcp = TcpStream::connect_timeout(&addr, TIMEOUT).map_err(|e| format!("{host}: {e}"))?;
    tcp.set_read_timeout(Some(TIMEOUT)).ok();
    tcp.set_write_timeout(Some(TIMEOUT)).ok();
    let req = format!("GET {path} HTTP/1.0\r\nHost: {host}\r\nUser-Agent: vearl-wallet-gui\r\nAccept: application/json\r\nConnection: close\r\n\r\n");
    let mut raw = Vec::new();
    if tls {
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let cfg = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(|e| e.to_string())?
            .with_root_certificates(roots)
            .with_no_client_auth();
        let name = rustls::pki_types::ServerName::try_from(host.clone()).map_err(|e| e.to_string())?;
        let conn = rustls::ClientConnection::new(Arc::new(cfg), name).map_err(|e| e.to_string())?;
        let mut tls_stream = rustls::StreamOwned::new(conn, tcp);
        tls_stream.write_all(req.as_bytes()).map_err(|e| format!("{host}: {e}"))?;
        read_limited(&mut tls_stream, &mut raw)?;
    } else {
        let mut tcp = tcp;
        tcp.write_all(req.as_bytes()).map_err(|e| format!("{host}: {e}"))?;
        read_limited(&mut tcp, &mut raw)?;
    }
    let (status, body) = split_reply(&raw)?;
    if status != 200 {
        return Err(format!("{host} answered {status}"));
    }
    String::from_utf8(body.to_vec()).map_err(|_| "the reply is not text".to_string())
}

fn read_limited<R: Read>(r: &mut R, out: &mut Vec<u8>) -> Result<(), String> {
    let mut buf = [0u8; 8192];
    loop {
        match r.read(&mut buf) {
            Ok(0) => return Ok(()),
            Ok(n) => {
                out.extend_from_slice(&buf[..n]);
                if out.len() > MAX_BODY {
                    return Err("the reply is too large".into());
                }
            }
            // a server that closes without a TLS close_notify ends the stream with this error: the data read so far is complete
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && !out.is_empty() => return Ok(()),
            Err(e) => return Err(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_are_split() {
        assert_eq!(split_url("https://vearl.org/api/x?y=1").unwrap(), (true, "vearl.org".into(), 443, "/api/x?y=1".into()));
        assert_eq!(split_url("http://127.0.0.1:8080").unwrap(), (false, "127.0.0.1".into(), 8080, "/".into()));
        assert!(split_url("ftp://x").is_err());
        assert!(split_url("https://user@evil.example/").is_err());
        assert!(split_url("https:///path").is_err());
        assert!(split_url("https://a b/").is_err());
    }

    #[test]
    fn replies_are_split() {
        let (s, b) = split_reply(b"HTTP/1.1 200 OK\r\nContent-Type: x\r\n\r\n{\"a\":1}").unwrap();
        assert_eq!((s, b), (200, &b"{\"a\":1}"[..]));
        assert_eq!(split_reply(b"HTTP/1.0 404 Not Found\r\n\r\n").unwrap().0, 404);
        assert!(split_reply(b"garbage").is_err());
    }

    #[test]
    fn plain_http_get_against_a_local_server() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf);
            s.write_all(b"HTTP/1.0 200 OK\r\n\r\nhello").unwrap();
        });
        assert_eq!(get(&format!("http://127.0.0.1:{port}/x")).unwrap(), "hello");
    }
}
