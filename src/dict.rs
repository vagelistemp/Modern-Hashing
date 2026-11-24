//! The common interface of every dictionary in this crate, plus the statistics and space reports
//! used by the tests and the benchmark.

/// Keys are `w = 32`-bit words, values are 32-bit payloads. Every `u32` (including `0` and
/// `u32::MAX`) is a valid key: nothing is reserved as an "empty" marker.
pub type Key = u32;
pub type Val = u32;

/// Word size `w` of the keys, in bits (the paper's `w`).
pub const W_BITS: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The table already holds `capacity` keys (all three structures of the paper have fixed capacity `n`).
    CapacityExceeded,
    /// A bucket overflowed and no new hash function managed to redistribute the keys (probability 1/poly(n)
    /// for paper-sized parameters, but certain if the parameters are deliberately too tight).
    RebuildFailed,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::CapacityExceeded => write!(f, "capacity exceeded"),
            Error::RebuildFailed => write!(f, "rebuild failed: no hash function distributes the keys"),
        }
    }
}

impl std::error::Error for Error {}

/// Counters for the rare events the paper's analysis is about.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Whole-table rebuilds (a bucket overflowed, new bucket hash function): probability 1/poly(n) in the paper.
    pub global_rebuilds: u64,
    /// Rebuilds of one bucket's fingerprints (two keys of a bucket got the same fingerprint), Lemma 4.1.
    pub bucket_rebuilds: u64,
    /// Times the fingerprint width had to be increased after repeated collisions. *Not* in the paper; a safety net
    /// that is only reachable with deliberately tiny fingerprints.
    pub fp_widenings: u64,
    /// K-Tries only: the growing trie filled up while the shrinking trie was not yet empty (should stay 0).
    pub forced_flushes: u64,
    /// K-Tries only: how many times the two tries exchanged roles.
    pub buffer_swaps: u64,
    /// Resizable table only: how many times the table changed its capacity (Section 6).
    pub resizes: u64,
}

/// Where the bits of a table go, to be compared with the paper's `n*w + O(n log log n)`.
#[derive(Clone, Debug, Default)]
pub struct SpaceReport {
    /// Number of keys currently stored.
    pub items: u64,
    /// Allocated key slots times `w` (the paper's `n + cn/log n` slots of `w` bits).
    pub key_slot_bits: u64,
    /// Allocated value slots (the paper analyses the key-only dictionary; values are reported separately).
    pub value_slot_bits: u64,
    /// Per-bucket counters `k` (O(log log n) bits each).
    pub counter_bits: u64,
    /// Query mappers: stored entries `(fingerprint, position)` plus node/pointer overhead.
    pub mapper_bits: u64,
    /// K-Tries only: the two buffers (tries).
    pub buffer_bits: u64,
}

impl SpaceReport {
    /// Everything except values: the quantity the paper bounds by `n*w + O(n log log n)`.
    pub fn key_dictionary_bits(&self) -> u64 {
        self.key_slot_bits + self.counter_bits + self.mapper_bits + self.buffer_bits
    }

    /// The overhead above the `items * w` bits needed just to store the keys, per stored item.
    pub fn overhead_bits_per_item(&self) -> f64 {
        if self.items == 0 {
            return 0.0;
        }
        (self.key_dictionary_bits() as f64 - (self.items * W_BITS as u64) as f64) / self.items as f64
    }
}

/// A fixed-capacity dictionary from `u32` keys to `u32` values.
pub trait Dictionary {
    /// Insert a key that is **not** in the dictionary. This is the operation the paper analyses
    /// (O(1) w.h.p. for Slow Partition). Passing a key that is already present is a logic error:
    /// it is caught by a `debug_assert!` in debug builds and corrupts nothing in release builds
    /// only for the structures whose bucket lookup is O(1) -- use [`Dictionary::insert`] when unsure.
    fn insert_new(&mut self, key: Key, val: Val) -> Result<(), Error>;

    /// Insert or overwrite; returns the previous value. For Slow Partition this has to scan the bucket
    /// (O(log^3 n)); for the indexed structures it is O(1).
    fn insert(&mut self, key: Key, val: Val) -> Result<Option<Val>, Error>;

    fn get(&self, key: Key) -> Option<Val>;

    fn contains(&self, key: Key) -> bool {
        self.get(key).is_some()
    }

    /// Remove a key; returns its value if it was present. Removing an absent key changes nothing.
    fn remove(&mut self, key: Key) -> Option<Val>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum number of keys `n`.
    fn capacity(&self) -> usize;

    fn stats(&self) -> Stats;

    fn space(&self) -> SpaceReport;

    /// Panics if any internal invariant is violated (left-justified buckets, consistent counters, every query-mapper
    /// entry pointing at a slot holding the matching key, ...). Used by the tests after every operation.
    fn check_invariants(&self);

    fn name(&self) -> String;
}
