//! Section 5: constant-time insertions and deletions with high probability, by buffering.
//!
//! > "Our new data structure will make use of two k-tries, each storing up to n^(1/4) w-bit keys. ... The growing trie
//! > receives new insertion/deletion operations. The shrinking trie has items removed as they are implemented. For
//! > every insertion/deletion operation that is added to the growing tree, a constant c > 1 amount of time is spent
//! > implementing operations from the shrinking tree in the actual hash table. We will guarantee that, whenever the
//! > growing tree reaches size n^(1/4), the shrinking tree will have shrunk to size 0 -- at this point, the two tries
//! > swap roles."
//!
//! `Buffered<D>` wraps any [`Dictionary`] `D` (the paper wraps the indexed partition table, but "the technique can
//! actually be extended to apply to any hash table", Appendix A) in this scheme.
//!
//! Semantics that the thesis implementation lacked: a lookup sees *every* acknowledged operation. It asks the growing
//! trie first, then the shrinking trie, then the table, so the newest pending operation on a key wins. The thesis
//! `get` read only the table, so a key that had just been `put` was invisible until its operation happened to be
//! applied.
//!
//! What the buffer does and does not buy (measured in the benchmark, explained in the documentation): operations are
//! applied to the table in constant-size batches spread over time, but an individual table operation that triggers a
//! rebuild is not interruptible here, so the *worst-case* latency of that one operation is not reduced; the paper's
//! real-time argument assumes such a rebuild can be paused and resumed.

use crate::dict::{Dictionary, Error, Key, SpaceReport, Stats, Val};
use crate::ktrie::{KTrie, Op};

pub struct Buffered<D: Dictionary> {
    inner: D,
    growing: KTrie,
    shrinking: KTrie,
    /// `n^(1/4)`: size at which the growing trie is handed over.
    threshold: usize,
    /// The paper's constant `c > 1`: operations applied from the shrinking trie per incoming operation.
    drain_per_op: usize,
    /// Logical capacity `n`.
    capacity: usize,
    /// Number of keys in the logical dictionary (table plus pending operations).
    logical_len: usize,
    forced_flushes: u64,
    buffer_swaps: u64,
}

/// The size `n^(1/4)` of the buffers for capacity `n`.
pub fn buffer_threshold(capacity: usize) -> usize {
    ((capacity as f64).powf(0.25).ceil() as usize).max(2)
}

impl<D: Dictionary> Buffered<D> {
    /// Buffer on top of `inner`. The table underneath must have room for the keys that are in the buffers but not yet
    /// applied: build it with capacity [`Buffered::inner_capacity`]`(n)`.
    pub fn new(inner: D, capacity: usize) -> Self {
        assert!(
            inner.capacity() >= Self::inner_capacity(capacity),
            "the table under a buffer needs capacity >= n + 2 n^(1/4)"
        );
        Buffered {
            inner,
            growing: KTrie::new(),
            shrinking: KTrie::new(),
            threshold: buffer_threshold(capacity),
            drain_per_op: 2,
            capacity,
            logical_len: 0,
            forced_flushes: 0,
            buffer_swaps: 0,
        }
    }

    /// Capacity of the table under a buffer for logical capacity `n`: the table can hold up to `2 n^(1/4)` keys whose
    /// deletions are still pending in addition to the `n` live keys.
    pub fn inner_capacity(n: usize) -> usize {
        n + 2 * buffer_threshold(n) + 2
    }

    pub fn with_drain(mut self, c: usize) -> Self {
        assert!(c >= 1);
        self.drain_per_op = c;
        self
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }

    pub fn pending(&self) -> usize {
        self.growing.len() + self.shrinking.len()
    }

    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Apply every pending operation to the table now.
    pub fn flush(&mut self) -> Result<(), Error> {
        while self.drain_one()? {}
        std::mem::swap(&mut self.growing, &mut self.shrinking);
        while self.drain_one()? {}
        Ok(())
    }

    #[inline]
    fn lookup(&self, key: Key) -> Option<Val> {
        match self.growing.get(key) {
            Some(Op::Put(v)) => Some(v),
            Some(Op::Delete) => None,
            None => match self.shrinking.get(key) {
                Some(Op::Put(v)) => Some(v),
                Some(Op::Delete) => None,
                None => self.inner.get(key),
            },
        }
    }

    /// Apply one operation from the shrinking trie to the table.
    fn drain_one(&mut self) -> Result<bool, Error> {
        match self.shrinking.pop_any() {
            None => Ok(false),
            Some((k, Op::Put(v))) => {
                self.inner.insert(k, v)?;
                Ok(true)
            }
            Some((k, Op::Delete)) => {
                self.inner.remove(k);
                Ok(true)
            }
        }
    }

    fn after_op(&mut self) -> Result<(), Error> {
        for _ in 0..self.drain_per_op {
            if !self.drain_one()? {
                break;
            }
        }
        if self.growing.len() >= self.threshold {
            if !self.shrinking.is_empty() {
                // cannot happen while c >= 1 and every operation adds at most one entry, but never hide it if it does
                self.forced_flushes += 1;
                while self.drain_one()? {}
            }
            std::mem::swap(&mut self.growing, &mut self.shrinking);
            self.buffer_swaps += 1;
        }
        Ok(())
    }
}

impl<D: Dictionary> Dictionary for Buffered<D> {
    fn insert_new(&mut self, key: Key, val: Val) -> Result<(), Error> {
        if self.logical_len >= self.capacity {
            return Err(Error::CapacityExceeded);
        }
        debug_assert!(self.lookup(key).is_none(), "insert_new called with a key that is already present");
        self.growing.insert(key, Op::Put(val));
        self.logical_len += 1;
        self.after_op()
    }

    fn insert(&mut self, key: Key, val: Val) -> Result<Option<Val>, Error> {
        let old = self.lookup(key);
        if old.is_none() && self.logical_len >= self.capacity {
            return Err(Error::CapacityExceeded);
        }
        self.growing.insert(key, Op::Put(val));
        if old.is_none() {
            self.logical_len += 1;
        }
        self.after_op()?;
        Ok(old)
    }

    fn get(&self, key: Key) -> Option<Val> {
        self.lookup(key)
    }

    fn remove(&mut self, key: Key) -> Option<Val> {
        let old = self.lookup(key)?;
        self.growing.insert(key, Op::Delete);
        self.logical_len -= 1;
        // A failure here could only be a rebuild failure inside the table; surface it loudly rather than losing data.
        self.after_op().expect("applying a buffered operation to the table failed");
        Some(old)
    }

    fn len(&self) -> usize {
        self.logical_len
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn stats(&self) -> Stats {
        let mut s = self.inner.stats();
        s.forced_flushes = self.forced_flushes;
        s.buffer_swaps = self.buffer_swaps;
        s
    }

    fn space(&self) -> SpaceReport {
        let mut r = self.inner.space();
        r.items = self.logical_len as u64;
        r.buffer_bits = self.growing.space_bits() + self.shrinking.space_bits();
        r
    }

    fn check_invariants(&self) {
        self.inner.check_invariants();
        self.growing.check();
        self.shrinking.check();
        assert!(self.growing.len() < self.threshold, "the growing trie must have been handed over when it filled");
        assert!(self.shrinking.len() <= self.threshold);
        // the logical size = table size corrected by every pending operation (the newest one per key)
        let mut keys = std::collections::HashSet::new();
        for &(k, _) in self.growing.iter().chain(self.shrinking.iter()) {
            keys.insert(k);
        }
        let mut expected = self.inner.len() as isize;
        for k in keys {
            let in_table = self.inner.get(k).is_some();
            let logical = self.lookup(k).is_some();
            expected += logical as isize - in_table as isize;
        }
        assert_eq!(expected, self.logical_len as isize, "logical length out of sync with the pending operations");
        assert!(self.logical_len <= self.capacity);
    }

    fn name(&self) -> String {
        format!("k-tries({})", self.inner.name())
    }
}
