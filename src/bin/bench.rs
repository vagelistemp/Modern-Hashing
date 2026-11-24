//! Benchmark harness: measures what each section of the paper promises.
//!
//! Method (the thesis measurements were single cold runs with keys built inside the timed loop):
//!   * release build, keys generated before timing, tables built outside the timed region,
//!   * one discarded warm-up repetition, then the *median* of several repetitions,
//!   * hits and misses reported separately; sequential and pseudo-random keys,
//!   * for the buffered table the final drain of the buffers is part of the timed region,
//!   * `std::hint::black_box` on everything that is read.
//!
//! Experiments (CSV files are written to the output directory):
//!   e1  per-operation cost as the BUCKET grows      -> Section 4 vs Section 3   (query/delete flat vs growing)
//!   e2  per-operation cost as n and the load grow    -> all structures vs HashMap baselines
//!   e3  latency distribution of single insertions    -> Section 5 vs Section 4   (p50 ... max)
//!   e4  space                                         -> bits per key vs n*w + O(n log log n)
//!   e5  the Four Russians tables themselves           -> build time, size, node-operation throughput
//!   e6  Section 6: resizable buckets and growing/shrinking tables -> space follows the number of keys
//!
//! Usage: bench [--quick] [--out DIR] [--only e1,e2,...] [--big]

use modern_hashing::mapper::{
    BTreeMapper, PackedNode, SortedVecMapper, Spec4x2, StdBTreeMapper, StdHashMapper, TabulatedNode, Tables,
};
use modern_hashing::partition::{BucketOps, Partition};
use modern_hashing::rng::{scramble32, SplitMix64};
use modern_hashing::{
    Buffered, Dictionary, Error, IndexedPartition, Key, Params, QueryMapper, Resizable, SlabIndexedPartition,
    SlowPartition, SpaceReport, Stats, Val,
};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::hash::{BuildHasherDefault, Hasher};
use std::hint::black_box;
use std::time::Instant;

type Packed = BTreeMapper<PackedNode>;
type Tab4x2 = BTreeMapper<TabulatedNode<Spec4x2>>;

// ---------------------------------------------------------------------------------------------------------------
// baselines: std HashMap with the default SipHash and with a fast multiplicative hasher
// ---------------------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
}

struct StdMap<S: std::hash::BuildHasher + Default> {
    map: HashMap<u32, u32, S>,
    cap: usize,
    label: &'static str,
}

impl<S: std::hash::BuildHasher + Default> StdMap<S> {
    fn new(cap: usize, label: &'static str) -> Self {
        StdMap { map: HashMap::with_capacity_and_hasher(cap, S::default()), cap, label }
    }
}

impl<S: std::hash::BuildHasher + Default> Dictionary for StdMap<S> {
    fn insert_new(&mut self, key: Key, val: Val) -> Result<(), Error> {
        self.map.insert(key, val);
        Ok(())
    }
    fn insert(&mut self, key: Key, val: Val) -> Result<Option<Val>, Error> {
        Ok(self.map.insert(key, val))
    }
    fn get(&self, key: Key) -> Option<Val> {
        self.map.get(&key).copied()
    }
    fn remove(&mut self, key: Key) -> Option<Val> {
        self.map.remove(&key)
    }
    fn len(&self) -> usize {
        self.map.len()
    }
    fn capacity(&self) -> usize {
        self.cap
    }
    fn stats(&self) -> Stats {
        Stats::default()
    }
    fn space(&self) -> SpaceReport {
        SpaceReport { items: self.map.len() as u64, ..Default::default() }
    }
    fn check_invariants(&self) {}
    fn name(&self) -> String {
        self.label.to_string()
    }
}

type SipMap = StdMap<std::collections::hash_map::RandomState>;
type FxMap = StdMap<BuildHasherDefault<Fx>>;

/// A dictionary the harness can drive, with a hook to finish pending work (the buffers of K-Tries).
trait BenchDict: Dictionary {
    fn drain(&mut self) {}
}
impl<B: BucketOps> BenchDict for Partition<B> {}
impl<D: Dictionary> BenchDict for Buffered<D> {
    fn drain(&mut self) {
        self.flush().expect("flush");
    }
}
impl<S: std::hash::BuildHasher + Default> BenchDict for StdMap<S> {}

// ---------------------------------------------------------------------------------------------------------------
// keys and measurement
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KeyMode {
    Sequential,
    Random,
}

impl KeyMode {
    fn name(self) -> &'static str {
        match self {
            KeyMode::Sequential => "sequential",
            KeyMode::Random => "random",
        }
    }
}

struct KeySet {
    /// Keys in insertion order.
    ins: Vec<u32>,
    /// The same keys, shuffled (query / upsert order).
    shuffled_a: Vec<u32>,
    /// The same keys, shuffled differently (deletion order).
    shuffled_b: Vec<u32>,
    /// Keys that are NOT in `ins`.
    miss: Vec<u32>,
}

fn shuffle(v: &mut [u32], rng: &mut SplitMix64) {
    for i in (1..v.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        v.swap(i, j);
    }
}

fn make_keys(n_items: usize, mode: KeyMode) -> KeySet {
    let gen = |i: u32| match mode {
        KeyMode::Sequential => i,
        KeyMode::Random => scramble32(i),
    };
    let ins: Vec<u32> = (0..n_items as u32).map(gen).collect();
    // scramble32 is a bijection, so indices n..2n give keys that differ from every inserted key
    let miss: Vec<u32> = (n_items as u32..2 * n_items as u32).map(gen).collect();
    let mut rng = SplitMix64::new(0xBEEF ^ n_items as u64);
    let mut shuffled_a = ins.clone();
    shuffle(&mut shuffled_a, &mut rng);
    let mut shuffled_b = ins.clone();
    shuffle(&mut shuffled_b, &mut rng);
    KeySet { ins, shuffled_a, shuffled_b, miss }
}

#[derive(Clone, Copy, Debug, Default)]
struct Phases {
    insert_new: f64,
    upsert: f64,
    get_hit: f64,
    get_miss: f64,
    remove: f64,
}

const PHASE_NAMES: [&str; 5] = ["insert_new", "upsert", "get_hit", "get_miss", "remove"];

impl Phases {
    fn get(&self, i: usize) -> f64 {
        [self.insert_new, self.upsert, self.get_hit, self.get_miss, self.remove][i]
    }
}

/// One repetition: fill, overwrite, query hits, query misses, delete everything. Returns nanoseconds per operation.
fn run_once<D: BenchDict>(mut d: D, ks: &KeySet, do_upsert: bool) -> Phases {
    let n = ks.ins.len() as f64;
    let mut p = Phases::default();

    let t = Instant::now();
    for &k in &ks.ins {
        d.insert_new(k, k ^ 0xA5A5_A5A5).unwrap();
    }
    d.drain();
    p.insert_new = t.elapsed().as_nanos() as f64 / n;
    assert_eq!(d.len(), ks.ins.len());

    if do_upsert {
        let t = Instant::now();
        let mut s = 0u64;
        for &k in &ks.shuffled_a {
            s += d.insert(k, k).unwrap().is_some() as u64;
        }
        d.drain();
        p.upsert = t.elapsed().as_nanos() as f64 / n;
        assert_eq!(s as usize, ks.ins.len());
        // restore the original values so that the following phases see the same table in both modes
    }

    let t = Instant::now();
    let mut found = 0usize;
    let mut sum = 0u64;
    for &k in &ks.shuffled_a {
        if let Some(v) = d.get(k) {
            found += 1;
            sum = sum.wrapping_add(v as u64);
        }
    }
    black_box(sum);
    p.get_hit = t.elapsed().as_nanos() as f64 / n;
    assert_eq!(found, ks.ins.len());

    let t = Instant::now();
    let mut found = 0usize;
    for &k in &ks.miss {
        found += d.get(k).is_some() as usize;
    }
    black_box(found);
    p.get_miss = t.elapsed().as_nanos() as f64 / n;
    assert_eq!(found, 0);

    let t = Instant::now();
    let mut removed = 0usize;
    for &k in &ks.shuffled_b {
        removed += d.remove(k).is_some() as usize;
    }
    d.drain();
    p.remove = t.elapsed().as_nanos() as f64 / n;
    assert_eq!(removed, ks.ins.len());
    assert_eq!(d.len(), 0);
    black_box(&d);
    p
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let l = v.len();
    if l % 2 == 1 {
        v[l / 2]
    } else {
        (v[l / 2 - 1] + v[l / 2]) / 2.0
    }
}

struct Summary {
    median: Phases,
    min: Phases,
    max: Phases,
    reps: usize,
}

fn measure<D: BenchDict>(mk: &dyn Fn() -> D, ks: &KeySet, reps: usize, warmup: bool, do_upsert: bool) -> Summary {
    if warmup {
        black_box(run_once(mk(), ks, do_upsert));
    }
    let mut all: Vec<Phases> = Vec::new();
    for _ in 0..reps {
        all.push(run_once(mk(), ks, do_upsert));
    }
    let mut med = Phases::default();
    let mut mn = Phases::default();
    let mut mx = Phases::default();
    for i in 0..5 {
        let mut col: Vec<f64> = all.iter().map(|p| p.get(i)).collect();
        let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = col.iter().cloned().fold(0.0, f64::max);
        let m = median(&mut col);
        match i {
            0 => (med.insert_new, mn.insert_new, mx.insert_new) = (m, lo, hi),
            1 => (med.upsert, mn.upsert, mx.upsert) = (m, lo, hi),
            2 => (med.get_hit, mn.get_hit, mx.get_hit) = (m, lo, hi),
            3 => (med.get_miss, mn.get_miss, mx.get_miss) = (m, lo, hi),
            _ => (med.remove, mn.remove, mx.remove) = (m, lo, hi),
        }
    }
    Summary { median: med, min: mn, max: mx, reps }
}

fn reps_for(n: usize, quick: bool) -> (usize, bool) {
    if quick {
        return (2, false);
    }
    if n <= 100_000 {
        (7, true)
    } else if n <= 1_000_000 {
        (5, true)
    } else {
        (3, false)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// implementations under test (generic dispatch, no virtual calls inside the timed loops)
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Imp {
    Slow,
    IdxPacked,
    IdxStdBTree,
    IdxStdHash,
    KTries,
    SipMap,
    FxMap,
}

impl Imp {
    fn label(self) -> &'static str {
        match self {
            Imp::Slow => "slow-partition",
            Imp::IdxPacked => "indexed[btree-packed7]",
            Imp::IdxStdBTree => "indexed[std-btree] (thesis-like)",
            Imp::IdxStdHash => "indexed[std-hash]",
            Imp::KTries => "k-tries[btree-packed7]",
            Imp::SipMap => "std HashMap (SipHash)",
            Imp::FxMap => "std HashMap (Fx hasher)",
        }
    }
}

const ALL_IMPS: [Imp; 7] =
    [Imp::Slow, Imp::IdxPacked, Imp::IdxStdBTree, Imp::IdxStdHash, Imp::KTries, Imp::SipMap, Imp::FxMap];

fn ktries_with<M: QueryMapper>(p: Params, n: usize) -> Buffered<IndexedPartition<M>> {
    let mut p = p;
    p.capacity = Buffered::<IndexedPartition<M>>::inner_capacity(n);
    Buffered::new(IndexedPartition::<M>::new(p), n)
}

/// Measure implementation `imp` for geometry `p` (capacity `n`).
fn measure_imp(imp: Imp, p: &Params, ks: &KeySet, reps: usize, warmup: bool, do_upsert: bool) -> Summary {
    let n = p.capacity;
    match imp {
        Imp::Slow => measure(&|| SlowPartition::new(p.clone()), ks, reps, warmup, do_upsert),
        Imp::IdxPacked => measure(&|| IndexedPartition::<Packed>::new(p.clone()), ks, reps, warmup, do_upsert),
        Imp::IdxStdBTree => measure(&|| IndexedPartition::<StdBTreeMapper>::new(p.clone()), ks, reps, warmup, do_upsert),
        Imp::IdxStdHash => measure(&|| IndexedPartition::<StdHashMapper>::new(p.clone()), ks, reps, warmup, do_upsert),
        Imp::KTries => measure(&|| ktries_with::<Packed>(p.clone(), n), ks, reps, warmup, do_upsert),
        Imp::SipMap => measure(&|| SipMap::new(n, "sip"), ks, reps, warmup, do_upsert),
        Imp::FxMap => measure(&|| FxMap::new(n, "fx"), ks, reps, warmup, do_upsert),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// output helpers
// ---------------------------------------------------------------------------------------------------------------

struct Out {
    dir: std::path::PathBuf,
}

impl Out {
    fn write(&self, name: &str, content: &str) {
        let p = self.dir.join(name);
        std::fs::write(&p, content).unwrap_or_else(|e| panic!("cannot write {}: {e}", p.display()));
        eprintln!("  wrote {}", p.display());
    }
}

fn fmt_ns(x: f64) -> String {
    if x >= 1000.0 {
        format!("{:>8.0}", x)
    } else {
        format!("{:>8.1}", x)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// e1: cost per operation as the bucket grows (Section 4 vs Section 3)
// ---------------------------------------------------------------------------------------------------------------

fn e1(out: &Out, quick: bool) {
    eprintln!("== e1: per-operation cost vs bucket size");
    let n_items: usize = if quick { 1 << 15 } else { 1 << 18 };
    let loads: &[usize] = if quick { &[32, 256, 2048] } else { &[32, 128, 512, 2048, 8192] };
    let imps = [Imp::Slow, Imp::IdxPacked, Imp::IdxStdBTree, Imp::IdxStdHash, Imp::FxMap];
    let ks = make_keys(n_items, KeyMode::Random);
    let mut csv = String::from("experiment,impl,n,mean_bucket_load,bucket_cap,keys,phase,ns_per_op_median,ns_per_op_min,ns_per_op_max,reps\n");
    println!("\n[e1] n = {n_items} keys (random), ns per operation as the mean bucket load grows");
    println!("{:<34}{:>8} {:>9} {:>9} {:>9} {:>9}", "implementation", "load", "insert", "get-hit", "get-miss", "remove");
    for &load in loads {
        let m = (n_items / load).max(1);
        let cap = load + (6.0 * (load as f64).sqrt()) as usize + 8;
        let p = Params::custom(n_items, m, cap, 39).with_seed(1);
        for &imp in &imps {
            let (reps, warm) = reps_for(n_items, quick);
            let s = measure_imp(imp, &p, &ks, reps, warm, false);
            println!(
                "{:<34}{:>8} {} {} {} {}",
                imp.label(),
                load,
                fmt_ns(s.median.insert_new),
                fmt_ns(s.median.get_hit),
                fmt_ns(s.median.get_miss),
                fmt_ns(s.median.remove)
            );
            for (i, ph) in PHASE_NAMES.iter().enumerate() {
                if *ph == "upsert" {
                    continue;
                }
                writeln!(
                    csv,
                    "e1,{},{},{},{},random,{},{:.3},{:.3},{:.3},{}",
                    imp.label(),
                    n_items,
                    load,
                    cap,
                    ph,
                    s.median.get(i),
                    s.min.get(i),
                    s.max.get(i),
                    s.reps
                )
                .unwrap();
            }
        }
    }
    out.write("e1_bucket_size.csv", &csv);
}

// ---------------------------------------------------------------------------------------------------------------
// e2: scaling with n and load factor (paper-sized geometry)
// ---------------------------------------------------------------------------------------------------------------

fn e2(out: &Out, quick: bool, big: bool) {
    eprintln!("== e2: scaling with n and load factor");
    let mut sizes: Vec<usize> = if quick { vec![1_000, 10_000] } else { vec![1_000, 10_000, 100_000, 1_000_000] };
    if big && !quick {
        sizes.push(10_000_000);
    }
    let lfs: &[f64] = if quick { &[0.9] } else { &[0.5, 0.9] };
    let mut csv = String::from("experiment,impl,n,lf,keys,phase,ns_per_op_median,ns_per_op_min,ns_per_op_max,reps\n");
    for &n in &sizes {
        for &lf in lfs {
            for mode in [KeyMode::Sequential, KeyMode::Random] {
                let items = (n as f64 * lf) as usize;
                let ks = make_keys(items, mode);
                let p = Params::paper(n).with_seed(1);
                println!(
                    "\n[e2] n = {n}, load factor {lf}, {} keys ({} items), ns per operation   (buckets {} x {} slots, fp {} bits)",
                    mode.name(),
                    items,
                    p.num_buckets,
                    p.bucket_cap,
                    p.fp_bits
                );
                println!(
                    "{:<34}{:>9} {:>9} {:>9} {:>9} {:>9}",
                    "implementation", "insert", "upsert", "get-hit", "get-miss", "remove"
                );
                for &imp in &ALL_IMPS {
                    if imp == Imp::Slow && n >= 10_000_000 {
                        println!("{:<34}  (skipped: O(bucket) scans at this size take minutes)", imp.label());
                        continue;
                    }
                    let (reps, warm) = reps_for(n, quick);
                    let s = measure_imp(imp, &p, &ks, reps, warm, true);
                    println!(
                        "{:<34} {} {} {} {} {}",
                        imp.label(),
                        fmt_ns(s.median.insert_new),
                        fmt_ns(s.median.upsert),
                        fmt_ns(s.median.get_hit),
                        fmt_ns(s.median.get_miss),
                        fmt_ns(s.median.remove)
                    );
                    for (i, ph) in PHASE_NAMES.iter().enumerate() {
                        writeln!(
                            csv,
                            "e2,{},{},{},{},{},{:.3},{:.3},{:.3},{}",
                            imp.label(),
                            n,
                            lf,
                            mode.name(),
                            ph,
                            s.median.get(i),
                            s.min.get(i),
                            s.max.get(i),
                            s.reps
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
    out.write("e2_scaling.csv", &csv);
}

// ---------------------------------------------------------------------------------------------------------------
// e3: latency of single operations (Section 5 vs Section 4)
// ---------------------------------------------------------------------------------------------------------------

fn timer_overhead_ns() -> f64 {
    let n = 200_000;
    let t = Instant::now();
    let mut acc = 0u128;
    for _ in 0..n {
        let a = Instant::now();
        acc += a.elapsed().as_nanos();
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / n as f64
}

fn percentile(sorted: &[u32], q: f64) -> u32 {
    let i = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[i]
}

/// Latency (ns) of each insertion of `ks.ins`, then of each deletion of `ks.shuffled_b`.
fn latencies<D: BenchDict>(mut d: D, ks: &KeySet) -> (Vec<u32>, Vec<u32>, Stats) {
    let mut ins = Vec::with_capacity(ks.ins.len());
    for &k in &ks.ins {
        let t = Instant::now();
        d.insert_new(k, k).unwrap();
        ins.push(t.elapsed().as_nanos().min(u32::MAX as u128) as u32);
    }
    let stats = d.stats();
    let mut del = Vec::with_capacity(ks.ins.len());
    for &k in &ks.shuffled_b {
        let t = Instant::now();
        black_box(d.remove(k));
        del.push(t.elapsed().as_nanos().min(u32::MAX as u128) as u32);
    }
    black_box(&d);
    (ins, del, stats)
}

fn e3(out: &Out, quick: bool) {
    eprintln!("== e3: latency distribution of single operations");
    let n: usize = if quick { 100_000 } else { 1_000_000 };
    let items = n * 9 / 10;
    let ks = make_keys(items, KeyMode::Random);
    let overhead = timer_overhead_ns();
    println!("\n[e3] n = {n}, {items} random keys; each operation timed individually (timer overhead ~{overhead:.0} ns is included in every figure)");
    let mut csv = String::from("experiment,config,impl,operation,p50_ns,p90_ns,p99_ns,p999_ns,p9999_ns,max_ns,ops_over_100us,global_rebuilds,bucket_rebuilds,fp_widenings,timer_overhead_ns\n");
    // configuration A: the paper's geometry. B: a deliberately hard geometry where rebuilds do happen.
    let paper = Params::paper(n).with_seed(1);
    // 14-bit fingerprints in buckets of thousands of keys: constant fingerprint collisions -> bucket rebuilds
    let stress = Params::paper(n).with_fp_bits(14).with_seed(1);
    for (cfg, p) in [("paper-geometry", &paper), ("14-bit-fingerprints", &stress)] {
        println!("\n  configuration: {cfg}");
        println!(
            "  {:<30}{:<8}{:>9}{:>9}{:>9}{:>9}{:>10}{:>11}{:>9}{:>9}",
            "implementation", "op", "p50", "p90", "p99", "p99.9", "p99.99", "max", ">100us", "rebuilds"
        );
        for imp in [Imp::IdxPacked, Imp::KTries, Imp::FxMap] {
            let (mut ins, mut del, stats) = match imp {
                Imp::IdxPacked => latencies(IndexedPartition::<Packed>::new(p.clone()), &ks),
                Imp::KTries => latencies(ktries_with::<Packed>(p.clone(), n), &ks),
                _ => latencies(FxMap::new(n, "fx"), &ks),
            };
            ins.sort_unstable();
            del.sort_unstable();
            for (op, v) in [("insert", &ins), ("delete", &del)] {
                let over = v.iter().filter(|&&x| x > 100_000).count();
                println!(
                    "  {:<30}{:<8}{:>9}{:>9}{:>9}{:>9}{:>10}{:>11}{:>9}{:>9}",
                    imp.label(),
                    op,
                    percentile(v, 0.5),
                    percentile(v, 0.9),
                    percentile(v, 0.99),
                    percentile(v, 0.999),
                    percentile(v, 0.9999),
                    v[v.len() - 1],
                    over,
                    stats.bucket_rebuilds + stats.global_rebuilds
                );
                writeln!(
                    csv,
                    "e3,{},{},{},{},{},{},{},{},{},{},{},{},{},{:.1}",
                    cfg,
                    imp.label(),
                    op,
                    percentile(v, 0.5),
                    percentile(v, 0.9),
                    percentile(v, 0.99),
                    percentile(v, 0.999),
                    percentile(v, 0.9999),
                    v[v.len() - 1],
                    over,
                    stats.global_rebuilds,
                    stats.bucket_rebuilds,
                    stats.fp_widenings,
                    overhead
                )
                .unwrap();
            }
        }
    }
    out.write("e3_latency.csv", &csv);
}

// ---------------------------------------------------------------------------------------------------------------
// e4: space
// ---------------------------------------------------------------------------------------------------------------

fn fill<D: Dictionary>(mut d: D, keys: &[u32]) -> D {
    for &k in keys {
        d.insert_new(k, k).unwrap();
    }
    d
}

fn e4(out: &Out, quick: bool) {
    eprintln!("== e4: space");
    let sizes: Vec<usize> = if quick { vec![10_000, 100_000] } else { vec![1_000, 10_000, 100_000, 1_000_000, 10_000_000] };
    let mut csv = String::from("experiment,impl,n,items,key_slot_bits_per_item,counter_bits_per_item,mapper_bits_per_item,buffer_bits_per_item,overhead_bits_per_item,paper_loglogn,fp_bits,pos_bits\n");
    println!("\n[e4] bits per stored key, table filled to capacity (w = 32 bits per key)");
    println!(
        "{:<34}{:>10}{:>11}{:>9}{:>9}{:>9}{:>12}{:>11}",
        "implementation", "n", "slots/key", "counter", "mapper", "buffer", "overhead", "log2 log2 n"
    );
    for &n in &sizes {
        let keys: Vec<u32> = (0..n as u32).map(scramble32).collect();
        let p = Params::paper(n).with_seed(1);
        let loglog = (n as f64).log2().log2();
        let reports: Vec<(String, SpaceReport)> = vec![
            ("slow-partition".to_string(), fill(SlowPartition::new(p.clone()), &keys).space()),
            ("indexed[btree-packed7]".to_string(), fill(IndexedPartition::<Packed>::new(p.clone()), &keys).space()),
            ("indexed[sorted-vec]".to_string(), fill(IndexedPartition::<SortedVecMapper>::new(p.clone()), &keys).space()),
            ("k-tries[btree-packed7]".to_string(), fill(ktries_with::<Packed>(p.clone(), n), &keys).space()),
        ];
        for (name, r) in reports {
            let per = |x: u64| x as f64 / n as f64;
            println!(
                "{:<34}{:>10}{:>11.2}{:>9.2}{:>9.2}{:>9.2}{:>12.2}{:>11.2}",
                name,
                n,
                per(r.key_slot_bits),
                per(r.counter_bits),
                per(r.mapper_bits),
                per(r.buffer_bits),
                r.overhead_bits_per_item(),
                loglog
            );
            writeln!(
                csv,
                "e4,{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{}",
                name,
                n,
                r.items,
                per(r.key_slot_bits),
                per(r.counter_bits),
                per(r.mapper_bits),
                per(r.buffer_bits),
                r.overhead_bits_per_item(),
                loglog,
                p.fp_bits,
                p.pos_bits
            )
            .unwrap();
        }
    }
    out.write("e4_space.csv", &csv);
}

// ---------------------------------------------------------------------------------------------------------------
// e5: the Four Russians tables
// ---------------------------------------------------------------------------------------------------------------

fn mapper_workload<M: QueryMapper>(fp_bits: u32, pos_bits: u32, universe: u64, ops: usize) -> f64 {
    let mut m = M::new(fp_bits, pos_bits);
    let mut rng = SplitMix64::new(5);
    let pos_mod = 1u64 << pos_bits;
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..ops {
        let fp = rng.below(universe);
        match rng.below(4) {
            0 => {
                m.insert(fp, rng.below(pos_mod) as u32);
            }
            1 => {
                acc += m.remove(fp).unwrap_or(0) as u64;
            }
            _ => acc += m.get(fp).unwrap_or(0) as u64,
        }
    }
    black_box(acc);
    black_box(&m);
    t.elapsed().as_nanos() as f64 / ops as f64
}

fn e5(out: &Out, quick: bool) {
    eprintln!("== e5: Four Russians tables");
    let t = Instant::now();
    let tables = Tables::build(4, 2);
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("\n[e5] tabulated node, 4-bit fingerprints, 2-bit positions, 3 entries per node");
    println!("  states: {}   table entries: {}   build time: {:.1} ms", tables.num_states, tables.table_entries(), build_ms);
    let ops = if quick { 1_000_000 } else { 5_000_000 };
    let mut csv = String::from("experiment,mapper,universe,ns_per_op\n");
    println!("  mapper workload (25% insert, 25% remove, 50% get on a 16-fingerprint universe), ns per operation:");
    let rows: Vec<(&str, f64)> = vec![
        ("btree-tabulated-4x2 (every node op is a table read)", mapper_workload::<Tab4x2>(4, 2, 16, ops)),
        ("btree-packed7", mapper_workload::<Packed>(4, 2, 16, ops)),
        ("sorted-vec", mapper_workload::<SortedVecMapper>(4, 2, 16, ops)),
        ("std-btree", mapper_workload::<StdBTreeMapper>(4, 2, 16, ops)),
        ("std-hash", mapper_workload::<StdHashMapper>(4, 2, 16, ops)),
    ];
    for (name, ns) in rows {
        println!("    {:<56}{:>8.1}", name, ns);
        writeln!(csv, "e5,{},16,{:.3}", name, ns).unwrap();
    }
    writeln!(csv, "e5-tables,states,{},{}", tables.num_states, build_ms).unwrap();
    writeln!(csv, "e5-tables,entries,{},{}", tables.table_entries(), build_ms).unwrap();
    out.write("e5_four_russians.csv", &csv);
}


// ---------------------------------------------------------------------------------------------------------------
// e6: Section 6 -- resizable buckets and a table that grows and shrinks
// ---------------------------------------------------------------------------------------------------------------

fn fill_and_space<D: Dictionary>(mut d: D, keys: &[u32]) -> SpaceReport {
    for &k in keys {
        d.insert_new(k, k).unwrap();
    }
    d.space()
}

fn e6(out: &Out, quick: bool) {
    eprintln!("== e6: resizing");
    // (a) memory of fixed-size buckets (Vec::with_capacity) versus slab buckets (Lemma 2.3) as the table fills
    let n: usize = if quick { 100_000 } else { 1_000_000 };
    let p = Params::paper(n).with_seed(1);
    let mut csv = String::from(
        "experiment,impl,n,fill,key_slot_bits_per_item,mapper_bits_per_item,insert_ns,get_hit_ns\n",
    );
    println!("\n[e6a] key-slot memory per stored key, capacity n = {n}: fixed buckets vs slab buckets");
    println!(
        "{:<26}{:>7}{:>19}{:>17}{:>12}{:>10}",
        "implementation", "fill", "key-slot bits/key", "mapper bits/key", "insert ns", "get ns"
    );
    for &fill in &[0.1f64, 0.3, 0.5, 0.7, 0.9] {
        let items = (n as f64 * fill) as usize;
        let ks = make_keys(items, KeyMode::Random);
        for (label, slab) in [("indexed (fixed buckets)", false), ("indexed (slab buckets)", true)] {
            let (reps, warm) = reps_for(n, quick);
            let (s, space) = if slab {
                let s = measure(&|| SlabIndexedPartition::<Packed>::new(p.clone()), &ks, reps, warm, false);
                (s, fill_and_space(SlabIndexedPartition::<Packed>::new(p.clone()), &ks.ins))
            } else {
                let s = measure(&|| IndexedPartition::<Packed>::new(p.clone()), &ks, reps, warm, false);
                (s, fill_and_space(IndexedPartition::<Packed>::new(p.clone()), &ks.ins))
            };
            let per = |x: u64| x as f64 / items as f64;
            println!(
                "{:<26}{:>7.1}{:>19.2}{:>17.2}{:>12.1}{:>10.1}",
                label,
                fill,
                per(space.key_slot_bits),
                per(space.mapper_bits),
                s.median.insert_new,
                s.median.get_hit
            );
            writeln!(
                csv,
                "e6a,{},{},{},{:.4},{:.4},{:.3},{:.3}",
                label,
                n,
                fill,
                per(space.key_slot_bits),
                per(space.mapper_bits),
                s.median.insert_new,
                s.median.get_hit
            )
            .unwrap();
        }
    }
    out.write("e6a_slab_space.csv", &csv);

    // (b) a table that grows from 1 000 keys to `target` keys and shrinks back
    let target = if quick { 50_000usize } else { 1_000_000 };
    let mut d =
        Resizable::new(1_000, |cap| SlabIndexedPartition::<Packed>::new(Params::paper(cap).with_seed(cap as u64)));
    let keys: Vec<u32> = (0..target as u32).map(scramble32).collect();
    let mut csv =
        String::from("experiment,phase,ops_done,len,capacity,resizing,bits_per_key,window_p99_ns,window_max_ns,resizes\n");
    println!(
        "\n[e6b] a Resizable table growing to {target} keys and shrinking back (a checkpoint every {} operations)",
        target / 10
    );
    println!(
        "{:<8}{:>10}{:>10}{:>10}{:>10}{:>12}{:>15}{:>15}{:>9}",
        "phase", "ops", "keys", "capacity", "resizing", "bits/key", "window p99 ns", "window max ns", "resizes"
    );
    let window = (target / 10).max(1);
    for phase in ["grow", "shrink"] {
        let mut lat: Vec<u32> = Vec::with_capacity(window);
        for (i, &k) in keys.iter().enumerate() {
            let t = Instant::now();
            if phase == "grow" {
                d.insert_new(k, k).unwrap();
            } else {
                black_box(d.remove(k));
            }
            lat.push(t.elapsed().as_nanos().min(u32::MAX as u128) as u32);
            if (i + 1) % window == 0 {
                lat.sort_unstable();
                let sp = d.space();
                let bits = (sp.key_slot_bits + sp.mapper_bits) as f64 / d.len().max(1) as f64;
                println!(
                    "{:<8}{:>10}{:>10}{:>10}{:>10}{:>12.1}{:>15}{:>15}{:>9}",
                    phase,
                    i + 1,
                    d.len(),
                    d.current_capacity(),
                    d.is_resizing(),
                    bits,
                    percentile(&lat, 0.99),
                    lat[lat.len() - 1],
                    d.stats().resizes
                );
                writeln!(
                    csv,
                    "e6b,{},{},{},{},{},{:.3},{},{},{}",
                    phase,
                    i + 1,
                    d.len(),
                    d.current_capacity(),
                    d.is_resizing() as u8,
                    bits,
                    percentile(&lat, 0.99),
                    lat[lat.len() - 1],
                    d.stats().resizes
                )
                .unwrap();
                lat.clear();
            }
        }
    }
    out.write("e6b_resizable.csv", &csv);
}

// ---------------------------------------------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let quick = args.iter().any(|a| a == "--quick");
    let big = args.iter().any(|a| a == "--big");
    let out_dir = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "results".to_string());
    let only: Option<Vec<String>> = args
        .iter()
        .position(|a| a == "--only")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.split(',').map(|x| x.to_string()).collect());
    std::fs::create_dir_all(&out_dir).expect("cannot create the output directory");
    let out = Out { dir: out_dir.into() };
    let want = |name: &str| only.as_ref().map_or(true, |v| v.iter().any(|x| x == name));
    println!(
        "modern_hashing benchmark  (quick = {quick}, big = {big}, logical cpus = {})",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );
    let t = Instant::now();
    if want("e1") {
        e1(&out, quick);
    }
    if want("e2") {
        e2(&out, quick, big);
    }
    if want("e3") {
        e3(&out, quick);
    }
    if want("e4") {
        e4(&out, quick);
    }
    if want("e5") {
        e5(&out, quick);
    }
    if want("e6") {
        e6(&out, quick);
    }
    println!("\ntotal time: {:.1} s", t.elapsed().as_secs_f64());
}
