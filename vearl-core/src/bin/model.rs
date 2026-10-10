//! vearl-model: tools for imported quantised models.
//!   vearl-model info MODEL.json                       summary (operations, weights, samples)
//!   vearl-model eval MODEL.json [--all]               runs the samples and compares with the labels (int8 accuracy)
//!   vearl-model dump PROGRAM.json                     runs a program {n,ops,weights,x0} and prints all the tensors
//!                                                     (base64): used for bit-for-bit conformance with an external engine
use vearl_core::model::*;

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    std::process::exit(1)
}

const HELP: &str = r#"vearl-model: tools for imported quantized models.

  vearl-model info MODEL.json              summary: operations, weights, samples
  vearl-model eval MODEL.json [--all]      run the samples and compare with the labels (int8 accuracy)
  vearl-model dump PROGRAM.json            run a program and print all its tensors (base64), for checks against another engine

Details: docs/12."#;

fn main() {
    vearl_core::launcher::help_or_continue(HELP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    match (args.first().map(String::as_str), args.get(1)) {
        (Some("info"), Some(p)) => {
            let m = load(std::path::Path::new(p)).unwrap_or_else(|e| die(&e));
            let gemms = m.ops.iter().filter(|o| o.is_gemm()).count();
            println!("model '{}': n={}, {} operations of which {} GEMM, {} weights, {} samples, classes={}, batch={}", m.name, m.n, m.ops.len(), gemms, m.weights.len(), m.samples.len(), m.classes, m.batch);
        }
        (Some("eval"), Some(p)) => {
            let m = load(std::path::Path::new(p)).unwrap_or_else(|e| die(&e));
            let (mut ok, mut total) = (0usize, 0usize);
            for (idx, s) in m.samples.iter().enumerate() {
                let logits = m.logits(idx).unwrap_or_else(|e| die(&e));
                for (b, col) in logits.iter().enumerate() {
                    let pred = col.iter().enumerate().max_by_key(|(i, v)| (**v, std::cmp::Reverse(*i))).map(|(i, _)| i).unwrap();
                    if let Some(l) = s.labels.get(b) {
                        total += 1;
                        ok += (pred == *l as usize) as usize;
                    }
                }
            }
            println!("int8 accuracy of the whole program over {total} samples: {ok}/{total} = {:.2} %", 100.0 * ok as f64 / total.max(1) as f64);
        }
        (Some("dump"), Some(p)) => {
            let txt = std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p}: {e}")));
            let input: DumpInput = serde_json::from_str(&txt).unwrap_or_else(|e| die(&format!("unreadable input: {e}")));
            let t = dump(&input).unwrap_or_else(|e| die(&e));
            println!("{}", serde_json::json!({ "tensors": t }));
        }
        _ => {
            eprintln!("usage: vearl-model (info | eval) MODEL.json | dump PROGRAM.json");
            std::process::exit(2);
        }
    }
}
