//! Throwaway check (not part of the deliverable crate): do the e2 / e1 benchmark configurations ever trigger a rebuild?
//! It repeats exactly what the benchmark did (same geometry, same seed, same keys, same insertion order) and prints the
//! counters. The hash functions are drawn from a deterministic PRNG, so the counts equal those of the benchmark run.

use modern_hashing::mapper::{BTreeMapper, PackedNode, StdBTreeMapper, StdHashMapper};
use modern_hashing::rng::scramble32;
use modern_hashing::{Buffered, Dictionary, IndexedPartition, Params, QueryMapper, SlowPartition};

type Packed = BTreeMapper<PackedNode>;

fn keys(items: usize, random: bool) -> Vec<u32> {
    (0..items as u32).map(|i| if random { scramble32(i) } else { i }).collect()
}

fn fill<D: Dictionary>(mut d: D, ks: &[u32]) -> (u64, u64, u64) {
    for &k in ks {
        d.insert_new(k, k ^ 0xA5A5_A5A5).unwrap();
    }
    let s = d.stats();
    (s.global_rebuilds, s.bucket_rebuilds, s.fp_widenings)
}

fn ktries<M: QueryMapper>(p: Params, n: usize) -> Buffered<IndexedPartition<M>> {
    let mut p = p;
    p.capacity = Buffered::<IndexedPartition<M>>::inner_capacity(n);
    Buffered::new(IndexedPartition::<M>::new(p), n)
}

fn main() {
    let mut total = (0u64, 0u64, 0u64);
    let mut runs = 0;
    println!("{:<12}{:>5}{:>12}  {:<22}{:>8}{:>8}{:>8}", "n", "lf", "keys", "implementation", "global", "bucket", "widen");
    for &n in &[1_000usize, 10_000, 100_000, 1_000_000, 10_000_000] {
        for &lf in &[0.5f64, 0.9] {
            for &random in &[false, true] {
                let items = (n as f64 * lf) as usize;
                let ks = keys(items, random);
                let p = Params::paper(n).with_seed(1);
                let mut rows: Vec<(&str, (u64, u64, u64))> = Vec::new();
                if n < 10_000_000 {
                    rows.push(("slow", fill(SlowPartition::new(p.clone()), &ks)));
                }
                rows.push(("indexed-packed", fill(IndexedPartition::<Packed>::new(p.clone()), &ks)));
                rows.push(("indexed-std-btree", fill(IndexedPartition::<StdBTreeMapper>::new(p.clone()), &ks)));
                rows.push(("indexed-std-hash", fill(IndexedPartition::<StdHashMapper>::new(p.clone()), &ks)));
                rows.push(("k-tries", fill(ktries::<Packed>(p.clone(), n), &ks)));
                for (name, s) in rows {
                    println!("{:<12}{:>5}{:>12}  {:<22}{:>8}{:>8}{:>8}", n, lf, if random { "random" } else { "sequential" }, name, s.0, s.1, s.2);
                    total.0 += s.0;
                    total.1 += s.1;
                    total.2 += s.2;
                    runs += 1;
                }
            }
        }
    }
    println!("\nTOTAL over {runs} configurations: global rebuilds {}, bucket rebuilds {}, fingerprint widenings {}", total.0, total.1, total.2);
}
