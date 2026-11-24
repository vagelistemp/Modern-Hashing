//! Shared model-based test driver: run a long random sequence of operations against a `HashMap` oracle and check, after
//! EVERY operation, the result of the operation and all internal invariants of the structure under test.

#![allow(dead_code)]

use modern_hashing::rng::SplitMix64;
use modern_hashing::{Dictionary, Error};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct ModelCfg {
    /// Keys are drawn from `0..universe` (plus the extreme keys) so that keys are inserted, deleted and re-inserted.
    pub universe: u32,
    pub ops: usize,
    pub seed: u64,
    /// Check every stored key against the oracle every this many operations (invariants are checked every operation).
    pub full_compare_every: usize,
}

impl ModelCfg {
    pub fn new(universe: u32, ops: usize, seed: u64) -> Self {
        ModelCfg { universe, ops, seed, full_compare_every: 25 }
    }
}

fn pick_key(rng: &mut SplitMix64, universe: u32) -> u32 {
    match rng.below(24) {
        0 => 0,
        1 => u32::MAX,
        2 => 0x8000_0000,
        3 => 1,
        _ => rng.below(universe as u64) as u32,
    }
}

/// Returns the table at the end so that callers can inspect its statistics.
pub fn run_model<D: Dictionary>(mut d: D, cfg: ModelCfg) -> D {
    let mut model: HashMap<u32, u32> = HashMap::new();
    let mut rng = SplitMix64::new(cfg.seed);
    let cap = d.capacity();
    for step in 0..cfg.ops {
        let key = pick_key(&mut rng, cfg.universe);
        let ctx = format!("{} step {step} key {key}", d.name());
        match rng.below(10) {
            // upsert
            0..=2 => {
                let val = rng.next_u32();
                if !model.contains_key(&key) && model.len() >= cap {
                    assert_eq!(d.insert(key, val), Err(Error::CapacityExceeded), "{ctx}");
                } else {
                    let expect = model.insert(key, val);
                    assert_eq!(d.insert(key, val), Ok(expect), "{ctx}");
                }
            }
            // insert_new (only when the key is absent, as the contract demands)
            3 | 4 => {
                if !model.contains_key(&key) {
                    let val = rng.next_u32();
                    if model.len() >= cap {
                        assert_eq!(d.insert_new(key, val), Err(Error::CapacityExceeded), "{ctx}");
                    } else {
                        assert_eq!(d.insert_new(key, val), Ok(()), "{ctx}");
                        model.insert(key, val);
                    }
                }
            }
            // remove (present or absent)
            5 | 6 | 7 => {
                assert_eq!(d.remove(key), model.remove(&key), "{ctx}");
            }
            // get
            _ => {
                assert_eq!(d.get(key), model.get(&key).copied(), "{ctx}");
            }
        }
        assert_eq!(d.len(), model.len(), "len, {ctx}");
        d.check_invariants();
        if cfg.full_compare_every > 0 && step % cfg.full_compare_every == 0 {
            for (k, v) in &model {
                assert_eq!(d.get(*k), Some(*v), "stored key {k} lost or corrupted, {ctx}");
            }
        }
    }
    for (k, v) in &model {
        assert_eq!(d.get(*k), Some(*v));
    }
    d
}
