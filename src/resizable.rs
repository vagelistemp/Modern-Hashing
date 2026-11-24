//! Section 6: dynamic resizing without hurting space or time.
//!
//! The paper has two ingredients. *Local* resizing: every bucket is a resizable array ([`crate::slab::SlabVec`]), so
//! memory follows the number of items instead of the capacity (use [`crate::indexed::SlabIndexedPartition`] as the
//! table). *Global* resizing, implemented here:
//!
//! > "Our approach for allowing n to change dramatically over time will be the standard one: whenever n changes by a
//! > constant factor, we rebuild the hash table according to Lemma 6.1 from scratch using a new n-bar = Θ(n). We
//! > spread the rebuild over O(n) operations so that each operation spends only O(1) time on it."
//!
//! `Resizable` holds the table in use (`current`) and, while a resize is in progress, the table that is being
//! emptied (`old`). Every operation also moves a constant number of items from `old` to `current`. The invariant is
//! that **a key lives in at most one of the two tables**: a lookup asks `current` first and then `old`; an insertion
//! or deletion of a key that is still in `old` removes it from there first, so a migration step can never overwrite a
//! newer value with an older one.

use crate::dict::{Dictionary, Error, Key, SpaceReport, Stats, Val};
use crate::partition::{BucketOps, Partition};

/// A table whose contents can be taken out a few items at a time.
pub trait Migratable: Dictionary {
    fn take_any(&mut self) -> Option<(Key, Val)>;
}

impl<B: BucketOps> Migratable for Partition<B> {
    fn take_any(&mut self) -> Option<(Key, Val)> {
        Partition::take_any(self)
    }
}

pub struct Resizable<D: Migratable, F: Fn(usize) -> D> {
    make: F,
    current: D,
    old: Option<D>,
    cap: usize,
    min_capacity: usize,
    len: usize,
    migrate_per_op: usize,
    resizes: u64,
}

/// Largest number of keys: the key space is `u32`.
const MAX_KEYS: usize = 1 << 32;

impl<D: Migratable, F: Fn(usize) -> D> Resizable<D, F> {
    /// A table that starts with room for `min_capacity` keys, never shrinks below that, and calls `make(capacity)` to
    /// build the table of each new capacity.
    pub fn new(min_capacity: usize, make: F) -> Self {
        let min_capacity = min_capacity.max(16);
        let current = make(min_capacity);
        Resizable { make, current, old: None, cap: min_capacity, min_capacity, len: 0, migrate_per_op: 4, resizes: 0 }
    }

    /// Items moved from the old table per operation while a resize is in progress (the paper's constant).
    pub fn with_migration_rate(mut self, per_op: usize) -> Self {
        assert!(per_op >= 2, "a rate below 2 may not finish a resize before the next one is due");
        self.migrate_per_op = per_op;
        self
    }

    /// Capacity of the table currently in use.
    pub fn current_capacity(&self) -> usize {
        self.cap
    }

    pub fn is_resizing(&self) -> bool {
        self.old.is_some()
    }

    pub fn current(&self) -> &D {
        &self.current
    }

    fn start_resize(&mut self, new_cap: usize) {
        debug_assert!(self.old.is_none());
        let new_table = (self.make)(new_cap);
        self.old = Some(std::mem::replace(&mut self.current, new_table));
        self.cap = new_cap;
        self.resizes += 1;
    }

    /// Called after every operation: move a few items, and decide whether a new resize is due.
    fn tick(&mut self) -> Result<(), Error> {
        if let Some(old) = self.old.as_mut() {
            for _ in 0..self.migrate_per_op {
                match old.take_any() {
                    Some((k, v)) => self.current.insert_new(k, v)?,
                    None => break,
                }
            }
            if old.len() == 0 {
                self.old = None;
            }
        }
        if self.old.is_none() {
            // grow when 3/4 full, shrink when 1/4 full: both leave the new table half full
            if self.len > self.cap / 4 * 3 && self.cap < MAX_KEYS {
                self.start_resize((self.cap * 2).min(MAX_KEYS));
            } else if self.len < self.cap / 4 && self.cap > self.min_capacity {
                self.start_resize((self.cap / 2).max(self.min_capacity));
            }
        }
        Ok(())
    }
}

impl<D: Migratable, F: Fn(usize) -> D> Dictionary for Resizable<D, F> {
    fn insert_new(&mut self, key: Key, val: Val) -> Result<(), Error> {
        debug_assert!(self.get(key).is_none(), "insert_new called with a key that is already present");
        self.current.insert_new(key, val)?;
        self.len += 1;
        self.tick()
    }

    fn insert(&mut self, key: Key, val: Val) -> Result<Option<Val>, Error> {
        let mut previous = None;
        if let Some(old) = self.old.as_mut() {
            previous = old.remove(key);
        }
        let result = if previous.is_some() {
            self.current.insert_new(key, val)?;
            previous
        } else {
            let r = self.current.insert(key, val)?;
            if r.is_none() {
                self.len += 1;
            }
            r
        };
        self.tick()?;
        Ok(result)
    }

    fn get(&self, key: Key) -> Option<Val> {
        self.current.get(key).or_else(|| self.old.as_ref().and_then(|o| o.get(key)))
    }

    fn remove(&mut self, key: Key) -> Option<Val> {
        let found = match self.current.remove(key) {
            Some(v) => Some(v),
            None => self.old.as_mut().and_then(|o| o.remove(key)),
        };
        if found.is_some() {
            self.len -= 1;
            self.tick().expect("migrating an item to the new table failed");
        }
        found
    }

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        MAX_KEYS
    }

    fn stats(&self) -> Stats {
        let mut s = self.current.stats();
        if let Some(o) = &self.old {
            let os = o.stats();
            s.global_rebuilds += os.global_rebuilds;
            s.bucket_rebuilds += os.bucket_rebuilds;
            s.fp_widenings += os.fp_widenings;
        }
        s.resizes = self.resizes;
        s
    }

    fn space(&self) -> SpaceReport {
        let mut r = self.current.space();
        if let Some(o) = &self.old {
            let os = o.space();
            r.key_slot_bits += os.key_slot_bits;
            r.value_slot_bits += os.value_slot_bits;
            r.counter_bits += os.counter_bits;
            r.mapper_bits += os.mapper_bits;
            r.buffer_bits += os.buffer_bits;
        }
        r.items = self.len as u64;
        r
    }

    fn check_invariants(&self) {
        self.current.check_invariants();
        let old_len = self.old.as_ref().map_or(0, |o| {
            o.check_invariants();
            o.len()
        });
        assert_eq!(self.len, self.current.len() + old_len, "length out of sync with the two tables");
        assert!(self.cap >= self.min_capacity);
        assert!(self.current.len() <= self.cap, "the table in use holds more than its capacity");
    }

    fn name(&self) -> String {
        format!("resizable({})", self.current.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexed::SlabIndexedPartition;
    use crate::mapper::{BTreeMapper, PackedNode};
    use crate::params::Params;
    use crate::rng::{scramble32, SplitMix64};
    use std::collections::HashMap;

    type Packed = BTreeMapper<PackedNode>;
    type Table = SlabIndexedPartition<Packed>;

    fn new_resizable(min: usize) -> Resizable<Table, impl Fn(usize) -> Table> {
        Resizable::new(min, |cap| Table::new(Params::paper(cap).with_seed(cap as u64)))
    }

    #[test]
    fn grows_and_shrinks_and_keeps_every_key() {
        let mut d = new_resizable(64);
        let n = 20_000u32;
        for k in 0..n {
            d.insert_new(scramble32(k), k).unwrap();
            if k % 997 == 0 {
                d.check_invariants();
            }
        }
        assert!(d.stats().resizes >= 8, "growing from 64 to 20 000 needs several doublings: {:?}", d.stats());
        assert!(d.current_capacity() >= n as usize);
        for k in 0..n {
            assert_eq!(d.get(scramble32(k)), Some(k));
        }
        for k in 0..n - 100 {
            assert_eq!(d.remove(scramble32(k)), Some(k));
            if k % 997 == 0 {
                d.check_invariants();
            }
        }
        d.check_invariants();
        assert_eq!(d.len(), 100);
        // finish any resize that is still in progress with a few more operations
        for _ in 0..50 {
            d.get(0);
            d.insert(1, 1).unwrap();
            d.remove(1);
        }
        assert!(d.current_capacity() <= 1024, "shrank back: capacity {}", d.current_capacity());
        for k in n - 100..n {
            assert_eq!(d.get(scramble32(k)), Some(k));
        }
    }

    #[test]
    fn matches_the_model_through_many_resizes() {
        let mut d = new_resizable(32);
        let mut model: HashMap<u32, u32> = HashMap::new();
        let mut rng = SplitMix64::new(31);
        for step in 0..120_000 {
            // alternate between growth and shrinkage phases so that the table keeps resizing
            let growing = (step / 15_000) % 2 == 0;
            let key = scramble32(rng.below(6_000) as u32);
            let r = rng.below(10);
            if (growing && r < 6) || (!growing && r < 2) {
                let v = rng.next_u32();
                assert_eq!(d.insert(key, v).unwrap(), model.insert(key, v), "step {step}");
            } else if r < 8 {
                assert_eq!(d.remove(key), model.remove(&key), "step {step}");
            } else {
                assert_eq!(d.get(key), model.get(&key).copied(), "step {step}");
            }
            assert_eq!(d.len(), model.len());
            if step % 40 == 0 {
                d.check_invariants();
            }
            if step % 5_000 == 0 {
                for (k, v) in &model {
                    assert_eq!(d.get(*k), Some(*v), "key {k} lost, step {step}");
                }
            }
        }
        assert!(d.stats().resizes >= 6, "{:?}", d.stats());
    }

    #[test]
    fn a_key_never_lives_in_both_tables() {
        let mut d = new_resizable(32);
        let mut live = Vec::new();
        for k in 0..3_000u32 {
            let key = scramble32(k);
            d.insert_new(key, k).unwrap();
            live.push(key);
            if d.is_resizing() {
                // overwrite and delete keys that may still be in the old table; the value must always be the newest one
                let victim = live[(k as usize * 7) % live.len()];
                d.insert(victim, 1_000_000 + k).unwrap();
                assert_eq!(d.get(victim), Some(1_000_000 + k));
            }
        }
        d.check_invariants();
    }

    #[test]
    fn slab_tables_use_memory_proportional_to_the_keys() {
        use crate::indexed::IndexedPartition;
        let n = 200_000usize;
        let keys: Vec<u32> = (0..n as u32 / 4).map(scramble32).collect();
        let mut fixed = IndexedPartition::<Packed>::new(Params::paper(n));
        let mut slab = Table::new(Params::paper(n));
        for &k in &keys {
            fixed.insert_new(k, k).unwrap();
            slab.insert_new(k, k).unwrap();
        }
        let f = fixed.space().key_slot_bits as f64 / keys.len() as f64;
        let s = slab.space().key_slot_bits as f64 / keys.len() as f64;
        // a quarter full: the fixed table still holds all its slots (>= 4 x 32 bits per key), the slab table about 32
        assert!(f > 120.0, "fixed table: {f} bits per key");
        assert!(s < 40.0, "slab table: {s} bits per key");
    }
}
