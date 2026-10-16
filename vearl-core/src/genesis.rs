//! JSON genesis file of the testnet: work parameters, initial allocations (`vrl1...` addresses), durations, retargeting.
//! The genesis hash (used by the network to refuse peers of another network) covers ALL these fields.
use crate::chain::{GenesisConfig, Retarget};
use crate::state::Timing;
use crate::wallet::{address_of, fmt_amount, parse_address, parse_amount};
use crate::work::WorkParams;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct ParamsJson {
    n: usize,
    tile: usize,
    rank: usize,
    fold_k: usize,
}

#[derive(Serialize, Deserialize)]
struct AllocJson {
    address: String,
    amount: String,
}

#[derive(Serialize, Deserialize)]
struct TimingJson {
    challenge_secs: u64,
    serve_secs: u64,
}

#[derive(Serialize, Deserialize)]
struct GenesisFile {
    network: String,
    params: ParamsJson,
    alloc: Vec<AllocJson>,
    timestamp: u64,
    /// Initial target (64-bit decimal integer; smaller = harder).
    target: String,
    timing: TimingJson,
    retarget: String,
}

pub const MAX_ALLOC: usize = 10_000;

pub fn parse_genesis(json: &str) -> Result<(String, GenesisConfig), String> {
    let f: GenesisFile = serde_json::from_str(json).map_err(|e| format!("unreadable genesis: {e}"))?;
    let params = WorkParams { n: f.params.n, tile: f.params.tile, rank: f.params.rank, fold_k: f.params.fold_k };
    params.security_check().map_err(|e| format!("work parameters refused: {e}"))?;
    if f.alloc.len() > MAX_ALLOC {
        return Err("too many allocations".into());
    }
    let mut alloc = Vec::new();
    let mut total: u64 = 0;
    let mut seen = std::collections::HashSet::new();
    for a in &f.alloc {
        let pk = parse_address(&a.address)?;
        if !seen.insert(pk) {
            return Err(format!("duplicate address in the genesis: {}", a.address));
        }
        let amt = parse_amount(&a.amount)?;
        total = total.checked_add(amt).ok_or("initial supply too large")?;
        alloc.push((pk, amt));
    }
    let target: u64 = f.target.parse().map_err(|_| "invalid target".to_string())?;
    if target < 1 << 16 {
        return Err("target too hard (< 2^16)".into());
    }
    if f.timing.challenge_secs < 60 || f.timing.serve_secs < 1 {
        return Err("consensus durations too short".into());
    }
    let retarget = match f.retarget.as_str() {
        "Lwma" => Retarget::Lwma,
        "Ema" => Retarget::Ema,
        "Fixed" => Retarget::Fixed,
        "Asert" => Retarget::Asert,
        o => return Err(format!("unknown retargeting '{o}'")),
    };
    Ok((f.network, GenesisConfig { params, alloc, ts: f.timestamp, target, timing: Timing { challenge_secs: f.timing.challenge_secs, serve_secs: f.timing.serve_secs }, retarget }))
}

pub fn genesis_to_json(network: &str, c: &GenesisConfig) -> String {
    let f = GenesisFile {
        network: network.to_string(),
        params: ParamsJson { n: c.params.n, tile: c.params.tile, rank: c.params.rank, fold_k: c.params.fold_k },
        alloc: c.alloc.iter().map(|(pk, a)| AllocJson { address: address_of(pk), amount: fmt_amount(*a) }).collect(),
        timestamp: c.ts,
        target: c.target.to_string(),
        timing: TimingJson { challenge_secs: c.timing.challenge_secs, serve_secs: c.timing.serve_secs },
        retarget: format!("{:?}", c.retarget),
    };
    serde_json::to_string_pretty(&f).unwrap()
}

/// Days since 1970-01-01 of a civil date (proleptic Gregorian calendar).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + if m <= 2 { 1 } else { 0 }, m, d)
}

/// Parses `YYYY-MM-DDTHH:MM:SSZ` (UTC only) into Unix seconds.
pub fn parse_utc(s: &str) -> Result<u64, String> {
    let bad = || format!("invalid date '{s}': expected format YYYY-MM-DDTHH:MM:SSZ (UTC), for example 2026-10-23T16:00:00Z");
    let b = s.as_bytes();
    if b.len() != 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' || b[19] != b'Z' {
        return Err(bad());
    }
    let n = |r: std::ops::Range<usize>| s.get(r).and_then(|x| x.parse::<i64>().ok()).ok_or_else(bad);
    let (y, mo, d, h, mi, se) = (n(0..4)?, n(5..7)?, n(8..10)?, n(11..13)?, n(14..16)?, n(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || se > 59 || y < 1970 {
        return Err(bad());
    }
    let days = days_from_civil(y, mo, d);
    if civil_from_days(days) != (y, mo, d) {
        return Err(bad()); // 31 February and other days that do not exist
    }
    Ok((days * 86_400 + h * 3600 + mi * 60 + se) as u64)
}

/// Unix seconds as `AAAA-MM-JJ HH:MM:SS UTC`.
pub fn fmt_utc(ts: u64) -> String {
    let (days, rem) = ((ts / 86_400) as i64, ts % 86_400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC", rem / 3600, rem % 3600 / 60, rem % 60)
}
