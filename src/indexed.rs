//! Indexed Partition Hash Tables (Section 4): constant-time queries and deletions.
//!
//! Every bucket gets a *fingerprint function* `f` (a hash into `[2^fp_bits]`, `fp_bits = 9 log log n` bits in the
//! paper, i.e. range `log^9 n`) and a *query mapper* from fingerprint to slot. With distinct fingerprints inside a
//! bucket (Lemma 4.1: they are, with probability `1 - O(1/log^3 n)`), a query hashes to a bucket, asks the mapper for
//! the slot and compares **one** key:
//!
//! > "To query an item x, we simply use the query mapper to obtain a position i. If the query mapper fails to find
//! > f(x), or if a different item x' is in position i, then x is not in the hash table."
//!
//! Deletion "removes the item and possibly moves one other item x' in order to keep the bucket left aligned; finally,
//! we update the query mapper to remove x, and to update the position stored for x'": exactly one mapper removal and
//! one mapper update.
//!
//! Differences from the thesis implementation, each of which the tests pin down:
//! * the slots are arrays (indexed by position), never linked lists, so reading slot `i` is O(1);
//! * `find` compares the stored key with the requested one (the thesis `get` did not, so absent keys with a colliding
//!   fingerprint returned another key's value and `delete` removed the wrong key);
//! * deletion moves one item (the last) instead of shifting the whole bucket and renumbering every mapper entry;
//! * a bucket rebuild *and* a whole-table rebuild both re-check that all fingerprints are distinct.

use crate::dict::{Key, SpaceReport, Val, W_BITS};
use crate::hash::UniversalHash;
use crate::mapper::QueryMapper;
use crate::params::Params;
use crate::partition::{BucketFull, BucketOps, Ctx, Partition};
use crate::rng::SplitMix64;
use crate::slab::Slots;

/// Consecutive failed fingerprint functions after which the fingerprint width is increased (a safety net that is not
/// in the paper; unreachable with paper-sized fingerprints, reachable in tests that use tiny fingerprints).
const WIDEN_AFTER: u32 = 32;

pub struct IndexedBucket<M: QueryMapper, S: Slots = Vec<u32>> {
    keys: S,
    vals: S,
    cap: usize,
    mapper: M,
    f: UniversalHash,
    fp_bits: u32,
    pos_bits: u32,
}

impl<M: QueryMapper, S: Slots> IndexedBucket<M, S> {
    #[inline]
    fn fp(&self, key: Key) -> u64 {
        self.f.fingerprint(key, self.fp_bits)
    }

    /// Current fingerprint width of this bucket.
    pub fn fp_bits(&self) -> u32 {
        self.fp_bits
    }

    /// Rebuild the mapper from scratch with a fresh fingerprint function under which all current keys, and `pending`
    /// if given, have pairwise distinct fingerprints. "In the event that the fingerprints in a bucket are not all
    /// different, we reconstruct the fingerprints from scratch using a different hash function (and so on, until we
    /// get distinct fingerprints)."
    fn rebuild_fingerprints(&mut self, pending: Option<Key>, ctx: &mut Ctx) {
        ctx.stats.bucket_rebuilds += 1;
        let need = self.keys.len() + pending.is_some() as usize;
        // a fingerprint space smaller than twice the number of keys makes distinctness hopeless: widen first
        while (1u64 << self.fp_bits) < 2 * need as u64 && self.fp_bits < M::max_fp_bits() {
            self.fp_bits += 1;
            ctx.stats.fp_widenings += 1;
        }
        assert!(
            (1u64 << self.fp_bits) >= need as u64,
            "bucket of {need} keys cannot get distinct fingerprints from a {}-bit space",
            self.fp_bits
        );
        let keys = self.keys.to_vec();
        let mut failures = 0u32;
        loop {
            let f = UniversalHash::random(ctx.rng);
            self.mapper.clear();
            let mut ok = true;
            for (pos, &k) in keys.iter().enumerate() {
                if !self.mapper.insert(f.fingerprint(k, self.fp_bits), pos as u32) {
                    ok = false;
                    break;
                }
            }
            if ok {
                if let Some(k) = pending {
                    ok = !self.mapper.contains(f.fingerprint(k, self.fp_bits));
                }
            }
            if ok {
                self.f = f;
                return;
            }
            failures += 1;
            if failures % WIDEN_AFTER == 0 && self.fp_bits < M::max_fp_bits() {
                self.fp_bits += 1;
                ctx.stats.fp_widenings += 1;
            }
        }
    }
}

impl<M: QueryMapper, S: Slots> BucketOps for IndexedBucket<M, S> {
    fn new(params: &Params, rng: &mut SplitMix64) -> Self {
        let fp_bits = params.fp_bits.min(M::max_fp_bits());
        assert!(params.pos_bits <= M::max_pos_bits(), "bucket too large for this query mapper");
        assert!(
            M::max_fp_bits() >= 24 || (1u64 << M::max_fp_bits()) >= 2 * params.bucket_cap as u64,
            "fingerprint space of the query mapper is too small for buckets of {} slots",
            params.bucket_cap
        );
        IndexedBucket {
            keys: S::with_max(params.bucket_cap),
            vals: S::with_max(params.bucket_cap),
            cap: params.bucket_cap,
            mapper: M::new(fp_bits, params.pos_bits),
            f: UniversalHash::random(rng),
            fp_bits,
            pos_bits: params.pos_bits,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    fn find(&self, key: Key) -> Option<usize> {
        let pos = self.mapper.get(self.fp(key))? as usize;
        // The mapper only knows fingerprints. Two different keys may share one (unless the key is present, in which
        // case it is *the* key with this fingerprint), so the stored key must be compared.
        if self.keys.get(pos) == key {
            Some(pos)
        } else {
            None
        }
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

    fn insert_new(&mut self, key: Key, val: Val, ctx: &mut Ctx) -> Result<(), BucketFull> {
        if self.keys.len() >= self.cap {
            return Err(BucketFull);
        }
        debug_assert!(self.find(key).is_none(), "insert_new: key already in the bucket");
        let pos = self.keys.len() as u32;
        let fp = self.fp(key);
        if !self.mapper.insert(fp, pos) {
            // the fingerprint belongs to another key: rebuild this bucket's mapper, the new key included
            self.rebuild_fingerprints(Some(key), ctx);
            let fp = self.fp(key);
            let inserted = self.mapper.insert(fp, pos);
            debug_assert!(inserted);
        }
        self.keys.push(key);
        self.vals.push(val);
        Ok(())
    }

    fn remove_at(&mut self, pos: usize) -> (Key, Val) {
        let key = self.keys.get(pos);
        let removed = self.mapper.remove(self.fp(key));
        debug_assert_eq!(removed, Some(pos as u32));
        let last = self.keys.len() - 1;
        if pos != last {
            // keep the bucket left justified: the last item moves into the hole, and its stored position is updated
            let moved = self.keys.get(last);
            self.mapper.set(self.fp(moved), pos as u32);
        }
        (self.keys.swap_remove(pos), self.vals.swap_remove(pos))
    }

    fn add_space(&self, r: &mut SpaceReport) {
        r.key_slot_bits += self.keys.allocated_slots() as u64 * W_BITS as u64 + self.keys.overhead_bits();
        r.value_slot_bits += self.vals.allocated_slots() as u64 * 32 + self.vals.overhead_bits();
        r.mapper_bits += self.mapper.space_bits();
    }

    fn check(&self) {
        assert_eq!(self.keys.len(), self.vals.len());
        assert!(self.keys.len() <= self.cap);
        assert_eq!(self.mapper.len(), self.keys.len(), "mapper and bucket disagree on the number of keys");
        self.mapper.check();
        for (pos, k) in self.keys.to_vec().into_iter().enumerate() {
            assert_eq!(
                self.mapper.get(self.fp(k)),
                Some(pos as u32),
                "mapper entry of the key in slot {pos} does not point back at that slot"
            );
        }
        assert!(self.pos_bits <= 16);
    }

    fn kind() -> String {
        format!("indexed-partition[{}]", M::name())
    }
}

/// Section 4: the Indexed Partition hash table over a chosen query mapper (fixed capacity, slots allocated up front).
pub type IndexedPartition<M> = Partition<IndexedBucket<M>>;

/// Section 4 with the slot arrays of Section 6 (resizable arrays of slabs): memory follows the number of stored keys.
pub type SlabIndexedPartition<M> = Partition<IndexedBucket<M, crate::slab::SlabVec>>;
