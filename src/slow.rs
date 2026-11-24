//! Slow Partition Hash Tables (Section 3).
//!
//! > "When an item arrives, it is hashed to a random bucket in the bucket array, and it is placed in an arbitrary
//! > empty slot in that bucket. If the bucket is full on insertion, the entire hash table is rebuilt. When a query is
//! > made, it is hashed to a bucket and every slot in the bucket is examined."
//!
//! Insertion of a new key is O(1): the bucket is left justified and keeps a counter `k` (here the length of the
//! two slot arrays), so the key goes into slot `k`. Queries and deletions scan the bucket: O(log^3 n).
//!
//! **Why `insert_new` and `insert` are different operations.** The paper's O(1) insertion assumes the key is not
//! already in the table. The thesis code checked for an existing key by scanning the bucket from slot 0 on every
//! insertion, which made its "O(1)" insertion cost O(bucket fill): between 126 and 924 slots per insertion in the
//! thesis workload. `insert_new` here is the O(1) operation of the paper; `insert` (upsert) is the one that has to scan.

use crate::dict::{Key, SpaceReport, Val, W_BITS};
use crate::params::Params;
use crate::partition::{BucketFull, BucketOps, Ctx, Partition};
use crate::rng::SplitMix64;
use crate::slab::Slots;

pub struct SlowBucket<S: Slots = Vec<u32>> {
    keys: S,
    vals: S,
    cap: usize,
    /// Slots examined by `find` so far (a probe used by the regression test of the thesis' O(bucket) insertion).
    scanned: std::cell::Cell<u64>,
}

impl<S: Slots> BucketOps for SlowBucket<S> {
    fn new(params: &Params, _rng: &mut SplitMix64) -> Self {
        SlowBucket {
            keys: S::with_max(params.bucket_cap),
            vals: S::with_max(params.bucket_cap),
            cap: params.bucket_cap,
            scanned: std::cell::Cell::new(0),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    fn find(&self, key: Key) -> Option<usize> {
        let r = self.keys.position(key);
        self.scanned.set(self.scanned.get() + r.map_or(self.keys.len(), |p| p + 1) as u64);
        r
    }

    fn debug_contains(&self, key: Key) -> bool {
        self.keys.position(key).is_some()
    }

    #[inline]
    fn key_at(&self, pos: usize) -> Key {
        self.keys.get(pos)
    }

    #[inline]
    fn val_at(&self, pos: usize) -> Val {
        self.vals.get(pos)
    }

    #[inline]
    fn set_val_at(&mut self, pos: usize, val: Val) {
        self.vals.set(pos, val);
    }

    #[inline]
    fn insert_new(&mut self, key: Key, val: Val, _ctx: &mut Ctx) -> Result<(), BucketFull> {
        if self.keys.len() >= self.cap {
            return Err(BucketFull);
        }
        debug_assert!(self.keys.position(key).is_none(), "insert_new: key already in the bucket");
        self.keys.push(key);
        self.vals.push(val);
        Ok(())
    }

    #[inline]
    fn remove_at(&mut self, pos: usize) -> (Key, Val) {
        // "the rightmost item is moved to that location, and then the counter k is decremented"
        (self.keys.swap_remove(pos), self.vals.swap_remove(pos))
    }

    fn add_space(&self, r: &mut SpaceReport) {
        r.key_slot_bits += self.keys.allocated_slots() as u64 * W_BITS as u64 + self.keys.overhead_bits();
        r.value_slot_bits += self.vals.allocated_slots() as u64 * 32 + self.vals.overhead_bits();
    }

    fn check(&self) {
        assert_eq!(self.keys.len(), self.vals.len());
        assert!(self.keys.len() <= self.cap);
    }

    fn kind() -> String {
        "slow-partition".to_string()
    }
}

/// Section 3: the Slow Partition hash table (fixed capacity, slot arrays allocated up front).
pub type SlowPartition = Partition<SlowBucket>;

impl<S: Slots> Partition<SlowBucket<S>> {
    /// Total number of slots examined by lookups, deletions and upserts so far. `insert_new` contributes nothing.
    pub fn slots_scanned(&self) -> u64 {
        self.buckets().iter().map(|b| b.scanned.get()).sum()
    }
}
