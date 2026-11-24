//! The bucket array shared by Slow Partition (Section 3) and Indexed Partition (Section 4).
//!
//! `Partition<B>` owns `m` buckets and the hash function that sends a key to a bucket; what a bucket *is* (a plain
//! array that is scanned, or an array plus a query mapper) is the type parameter `B`. Keeping this layer common makes
//! the comparison of Sections 3 and 4 fair: the only difference between the two tables is the bucket.

use crate::dict::{Dictionary, Error, Key, SpaceReport, Stats, Val};
use crate::hash::UniversalHash;
use crate::params::{ceil_log2, Params};
use crate::rng::SplitMix64;

/// How many fresh bucket hash functions a whole-table rebuild may try before giving up.
pub const MAX_REBUILD_ATTEMPTS: usize = 64;

/// The bucket has no free slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketFull;

/// Randomness and statistics handed to bucket operations that may need to rebuild something.
pub struct Ctx<'a> {
    pub rng: &'a mut SplitMix64,
    pub stats: &'a mut Stats,
}

/// A bucket: up to `cap` `(key, value)` slots kept *left justified* (the `k` used slots come first), so that an
/// insertion appends at slot `k` and a deletion moves the last item into the hole.
pub trait BucketOps {
    fn new(params: &Params, rng: &mut SplitMix64) -> Self;
    /// Number of used slots `k`.
    fn len(&self) -> usize;
    /// Slot of `key`, if present.
    fn find(&self, key: Key) -> Option<usize>;
    /// Existence check used only inside `debug_assert!`s (so that a debug build's self-checks do not distort probes
    /// that count the work done by the operation under test).
    fn debug_contains(&self, key: Key) -> bool {
        self.find(key).is_some()
    }
    fn key_at(&self, pos: usize) -> Key;
    fn val_at(&self, pos: usize) -> Val;
    fn set_val_at(&mut self, pos: usize, val: Val);
    /// Append a key that is **not** in the bucket. `Err(BucketFull)` if all `cap` slots are used.
    fn insert_new(&mut self, key: Key, val: Val, ctx: &mut Ctx) -> Result<(), BucketFull>;
    /// Remove the item in slot `pos` by moving the last item into it; returns the removed item.
    fn remove_at(&mut self, pos: usize) -> (Key, Val);
    /// Remove and return the last item (O(1) for every bucket type: nothing has to move).
    fn pop_last(&mut self) -> Option<(Key, Val)> {
        let n = self.len();
        if n == 0 {
            None
        } else {
            Some(self.remove_at(n - 1))
        }
    }
    fn add_space(&self, r: &mut SpaceReport);
    /// Panics if an invariant of the bucket is violated.
    fn check(&self);
    fn kind() -> String;
}

pub struct Partition<B: BucketOps> {
    params: Params,
    hash: UniversalHash,
    buckets: Vec<B>,
    len: usize,
    rng: SplitMix64,
    stats: Stats,
    /// First bucket that may still hold items, for [`Partition::take_any`] (incremental migration, Section 6).
    drain_cursor: usize,
}

impl<B: BucketOps> Partition<B> {
    pub fn new(params: Params) -> Self {
        let mut rng = SplitMix64::new(params.seed);
        let hash = UniversalHash::random(&mut rng);
        let buckets = (0..params.num_buckets).map(|_| B::new(&params, &mut rng)).collect();
        Partition { params, hash, buckets, len: 0, rng, stats: Stats::default(), drain_cursor: 0 }
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The buckets (read-only), for probes and experiments.
    pub fn buckets(&self) -> &[B] {
        &self.buckets
    }

    /// Number of keys in each bucket (for experiments on load balance).
    pub fn bucket_lens(&self) -> Vec<usize> {
        self.buckets.iter().map(|b| b.len()).collect()
    }

    /// Remove and return some item in O(1) amortised time (the last item of the first non-empty bucket). Used to move
    /// the contents of a table into a bigger or smaller one a few items at a time (Section 6).
    pub fn take_any(&mut self) -> Option<(Key, Val)> {
        while self.drain_cursor < self.buckets.len() {
            if let Some(kv) = self.buckets[self.drain_cursor].pop_last() {
                self.len -= 1;
                return Some(kv);
            }
            self.drain_cursor += 1;
        }
        None
    }

    #[inline]
    fn bucket_of(&self, key: Key) -> usize {
        self.hash.bucket(key, self.params.num_buckets)
    }

    /// Redistribute every key (plus `extra`) with a freshly drawn bucket hash function. The old table is kept
    /// untouched if no function works, so a failed rebuild loses nothing.
    fn rebuild_all(&mut self, extra: Option<(Key, Val)>) -> Result<(), Error> {
        self.stats.global_rebuilds += 1;
        let mut items: Vec<(Key, Val)> = Vec::with_capacity(self.len + 1);
        for b in &self.buckets {
            for p in 0..b.len() {
                items.push((b.key_at(p), b.val_at(p)));
            }
        }
        if let Some(e) = extra {
            items.push(e);
        }
        let m = self.params.num_buckets;
        'attempt: for _ in 0..MAX_REBUILD_ATTEMPTS {
            let hash = UniversalHash::random(&mut self.rng);
            let mut buckets: Vec<B> = (0..m).map(|_| B::new(&self.params, &mut self.rng)).collect();
            for &(k, v) in &items {
                let b = hash.bucket(k, m);
                let mut ctx = Ctx { rng: &mut self.rng, stats: &mut self.stats };
                if buckets[b].insert_new(k, v, &mut ctx).is_err() {
                    continue 'attempt;
                }
            }
            self.hash = hash;
            self.buckets = buckets;
            self.drain_cursor = 0;
            return Ok(());
        }
        Err(Error::RebuildFailed)
    }
}

impl<B: BucketOps> Dictionary for Partition<B> {
    fn insert_new(&mut self, key: Key, val: Val) -> Result<(), Error> {
        if self.len >= self.params.capacity {
            return Err(Error::CapacityExceeded);
        }
        let b = self.bucket_of(key);
        debug_assert!(!self.buckets[b].debug_contains(key), "insert_new called with a key that is already present");
        let mut ctx = Ctx { rng: &mut self.rng, stats: &mut self.stats };
        match self.buckets[b].insert_new(key, val, &mut ctx) {
            Ok(()) => {
                self.len += 1;
                self.drain_cursor = self.drain_cursor.min(b);
                Ok(())
            }
            Err(BucketFull) => {
                // the paper: "If the bucket is full on insertion, the entire hash table is rebuilt."
                self.rebuild_all(Some((key, val)))?;
                self.len += 1;
                Ok(())
            }
        }
    }

    fn insert(&mut self, key: Key, val: Val) -> Result<Option<Val>, Error> {
        let b = self.bucket_of(key);
        if let Some(pos) = self.buckets[b].find(key) {
            let old = self.buckets[b].val_at(pos);
            self.buckets[b].set_val_at(pos, val);
            return Ok(Some(old));
        }
        self.insert_new(key, val).map(|_| None)
    }

    fn get(&self, key: Key) -> Option<Val> {
        let bucket = &self.buckets[self.bucket_of(key)];
        bucket.find(key).map(|p| bucket.val_at(p))
    }

    fn remove(&mut self, key: Key) -> Option<Val> {
        let b = self.bucket_of(key);
        let pos = self.buckets[b].find(key)?;
        let (_, v) = self.buckets[b].remove_at(pos);
        self.len -= 1;
        Some(v)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        self.params.capacity
    }

    fn stats(&self) -> Stats {
        self.stats.clone()
    }

    fn space(&self) -> SpaceReport {
        let mut r = SpaceReport { items: self.len as u64, ..Default::default() };
        for b in &self.buckets {
            b.add_space(&mut r);
        }
        // one counter k per bucket, 0..=cap
        r.counter_bits += self.params.num_buckets as u64 * ceil_log2(self.params.bucket_cap + 1) as u64;
        r
    }

    fn check_invariants(&self) {
        let mut total = 0usize;
        let mut seen = std::collections::HashSet::with_capacity(self.len);
        for (bi, b) in self.buckets.iter().enumerate() {
            b.check();
            assert!(b.len() <= self.params.bucket_cap, "bucket {bi} over capacity");
            total += b.len();
            for p in 0..b.len() {
                let k = b.key_at(p);
                assert_eq!(self.bucket_of(k), bi, "key {k} is in bucket {bi} but hashes elsewhere");
                assert!(seen.insert(k), "key {k} stored twice");
                assert_eq!(b.find(k), Some(p), "find({k}) does not return the slot that holds it");
            }
        }
        assert_eq!(total, self.len, "len counter out of sync");
        assert!(self.len <= self.params.capacity);
    }

    fn name(&self) -> String {
        B::kind()
    }
}
