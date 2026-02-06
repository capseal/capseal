use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use bicep_crypto::{bef_verify_fast, ChunkSketch, FieldElement, TraceCommitParams, TraceCommitment};
use serde::Deserialize;

#[derive(Deserialize)]
struct TraceCommitmentJson {
    len: u64,
    root_hex: String,
    challenges: Vec<u64>,
    sketches: Vec<u64>,
}

#[derive(Deserialize)]
struct ChunkJson {
    chunk_index: u64,
    offset: u64,
    length: u64,
    root_hex: String,
    sketch_vec: Vec<u64>,
}

#[derive(Deserialize)]
struct SketchFileJson {
    trace_commitment: TraceCommitmentJson,
    chunks: Vec<ChunkJson>,
}

fn parse_hex32(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("expected 64 hex chars, got {}", hex.len()));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        let byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .map_err(|e| format!("invalid hex at byte {}: {}", i, e))?;
        out[i] = byte;
    }
    Ok(out)
}

fn load_sketch(path: &PathBuf) -> Result<(TraceCommitment, Vec<ChunkSketch>), String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let parsed: SketchFileJson = serde_json::from_str(&data)
        .map_err(|e| format!("failed to parse JSON: {}", e))?;

    let commitment = TraceCommitment {
        len: parsed.trace_commitment.len,
        root: parse_hex32(&parsed.trace_commitment.root_hex)?,
        challenges: parsed
            .trace_commitment
            .challenges
            .iter()
            .map(|&v| FieldElement::new(v))
            .collect(),
        sketches: parsed
            .trace_commitment
            .sketches
            .iter()
            .map(|&v| FieldElement::new(v))
            .collect(),
    };

    let mut chunk_summaries = Vec::with_capacity(parsed.chunks.len());
    for chunk in parsed.chunks.into_iter() {
        let root = parse_hex32(&chunk.root_hex)?;
        let sketch_vec = chunk
            .sketch_vec
            .into_iter()
            .map(FieldElement::new)
            .collect();
        chunk_summaries.push(ChunkSketch {
            chunk_index: chunk.chunk_index,
            offset: chunk.offset,
            length: chunk.length,
            root,
            sketch_vec,
        });
    }

    Ok((commitment, chunk_summaries))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: bef_verify_fast_cli <sketch.json>");
            std::process::exit(2);
        }
    };

    let (commitment, chunks) = match load_sketch(&path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("error: {}", err);
            std::process::exit(1);
        }
    };

    let params = TraceCommitParams {
        num_challenges: commitment.challenges.len(),
    };

    let start = Instant::now();
    let ok = bef_verify_fast(&params, &commitment, &chunks);
    let elapsed = start.elapsed().as_secs_f64() * 1_000.0;

    if !ok {
        eprintln!("bef_verify_fast rejected sketch");
        std::process::exit(1);
    }

    println!("{:.6}", elapsed);
}
