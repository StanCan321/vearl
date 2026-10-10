//! Minimal HTTP/1.1 server (no dependency) for the node API and the faucet.
//! Deliberately restricted: no keep-alive, no chunked encoding, no TLS (put it behind a TLS reverse proxy
//! for public exposure). Bounded inputs and short timeouts against slow or hostile clients.
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub const MAX_HEADER: usize = 16 * 1024;
pub const MAX_BODY: usize = 2 * 1024 * 1024;
pub const MAX_CONNS: usize = 64;

pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    pub body: Vec<u8>,
    pub peer: String,
}

pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(status: u16, v: &serde_json::Value) -> Response {
        Response { status, content_type: "application/json", body: v.to_string().into_bytes() }
    }
    pub fn html(body: &str) -> Response {
        Response { status: 200, content_type: "text/html; charset=utf-8", body: body.as_bytes().to_vec() }
    }
    pub fn error(status: u16, msg: &str) -> Response {
        Response::json(status, &serde_json::json!({ "error": msg }))
    }
}

fn reason(s: u16) -> &'static str {
    match s {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

pub fn read_request(stream: &mut TcpStream) -> io::Result<Request> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let tcp_peer = stream.peer_addr().ok().map(|a| a.ip());
    let mut peer = tcp_peer.map(|a| a.to_string()).unwrap_or_default();
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 2048];
    let header_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if buf.len() > MAX_HEADER {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "headers too long"));
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "connection closed"));
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let first = lines.next().unwrap_or("");
    let mut parts = first.split(' ');
    let (method, target) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
    if !matches!(parts.next(), Some("HTTP/1.1") | Some("HTTP/1.0")) || method.is_empty() || !target.starts_with('/') {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid request"));
    }
    let mut content_length = 0usize;
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            // behind the local reverse proxy (nginx sets X-Real-IP to the address it saw) the client is the one it names: without this every
            // visitor of the faucet would share the proxy's address and its per-IP limit. Only a connection from the loopback is trusted.
            if k.eq_ignore_ascii_case("x-real-ip") && tcp_peer.map_or(false, |a| a.is_loopback()) {
                if let Ok(ip) = v.trim().parse::<std::net::IpAddr>() {
                    peer = ip.to_string();
                }
            }
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid content-length"))?;
            }
            if k.eq_ignore_ascii_case("transfer-encoding") {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "transfer-encoding not supported"));
            }
        }
    }
    if content_length > MAX_BODY {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "body too large"));
    }
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "incomplete body"));
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    let (path, query) = target.split_once('?').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((target.clone(), String::new()));
    Ok(Request { method, path, query, body, peer })
}

pub fn write_response(stream: &mut TcpStream, r: &Response) -> io::Result<()> {
    write_response_ex(stream, r, false)
}

pub fn write_response_ex(stream: &mut TcpStream, r: &Response, head_only: bool) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nReferrer-Policy: no-referrer\r\n\
         Content-Security-Policy: default-src 'none'; connect-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\r\n\r\n",
        r.status,
        reason(r.status),
        r.content_type,
        r.body.len()
    );
    stream.write_all(head.as_bytes())?;
    if !head_only {
        stream.write_all(&r.body)?;
    }
    stream.flush()
}

/// Starts a server; `handler` is called in one thread per connection (cap MAX_CONNS). Returns the bound address.
pub fn serve<F>(addr: &str, handler: F) -> io::Result<String>
where
    F: Fn(Request) -> Response + Send + Sync + 'static,
{
    let listener = TcpListener::bind(addr)?;
    let local = listener.local_addr()?.to_string();
    let handler = Arc::new(handler);
    let active = Arc::new(AtomicUsize::new(0));
    thread::spawn(move || {
        for s in listener.incoming() {
            let Ok(mut s) = s else { continue };
            if active.load(Ordering::SeqCst) >= MAX_CONNS {
                let _ = write_response(&mut s, &Response::error(503, "server busy"));
                continue;
            }
            active.fetch_add(1, Ordering::SeqCst);
            let (h, a) = (handler.clone(), active.clone());
            thread::spawn(move || {
                let mut head_only = false;
                let resp = match read_request(&mut s) {
                    Ok(mut req) => {
                        if req.method == "HEAD" {
                            head_only = true; // same response as a GET, without a body
                            req.method = "GET".into();
                        }
                        h(req)
                    }
                    Err(e) => Response::error(if e.to_string().contains("too") { 413 } else { 400 }, &e.to_string()),
                };
                let _ = write_response_ex(&mut s, &resp, head_only);
                a.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    Ok(local)
}

/// Splits "a=1&b=2" (no percent decoding: the expected values are numeric or hexadecimal).
pub fn query_get<'a>(q: &'a str, key: &str) -> Option<&'a str> {
    q.split('&').find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v))
}
