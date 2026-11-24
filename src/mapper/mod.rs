//! The *query mapper* of Section 4: a small dictionary, one per bucket, from a Θ(log log n)-bit fingerprint
//! to the Θ(log log n)-bit position where the key lives in the bucket.
//!
//! The indexed partition hash table is generic over this trait so that the very same table (and the very same
//! tests) run on top of
//!
//! | mapper                       | what it is                                                        |
//! |------------------------------|-------------------------------------------------------------------|
//! | [`SortedVecMapper`]          | sorted array + binary search: obviously correct, the test oracle  |
//! | [`StdBTreeMapper`]           | `std::collections::BTreeMap`: what the thesis did (`TreeMap`)      |
//! | [`StdHashMapper`]            | `std::collections::HashMap`: a practical O(1) baseline             |
//! | [`BTreeMapper<PackedNode>`]  | the paper's B-tree, nodes in `u64` words, tabulated node lookups   |
//! | [`BTreeMapper<TabulatedNode>`]| the paper's B-tree where **every** node operation is a table read |

pub mod btree;
pub mod sorted;
pub mod std_maps;
pub mod tabulated;

pub use btree::{BTreeMapper, PackedNode};
pub use sorted::SortedVecMapper;
pub use std_maps::{StdBTreeMapper, StdHashMapper};
pub use tabulated::{Spec3x3, Spec4x2, TabSpec, TabulatedNode, Tables};

/// Fingerprint -> position dictionary of one bucket.
pub trait QueryMapper: Sized {
    /// An empty mapper for fingerprints of at most `fp_bits` bits and positions `< 2^pos_bits`.
    fn new(fp_bits: u32, pos_bits: u32) -> Self;

    /// Largest fingerprint width this mapper can represent (48 unless the representation is narrower).
    fn max_fp_bits() -> u32 {
        crate::params::MAX_FP_BITS
    }

    /// Largest position width this mapper can represent.
    fn max_pos_bits() -> u32 {
        crate::params::MAX_POS_BITS
    }

    fn clear(&mut self);

    fn len(&self) -> usize;

    /// The position stored for `fp`.
    fn get(&self, fp: u64) -> Option<u32>;

    fn contains(&self, fp: u64) -> bool {
        self.get(fp).is_some()
    }

    /// Insert `fp -> pos`. Returns `false` and changes nothing if `fp` is already present.
    fn insert(&mut self, fp: u64, pos: u32) -> bool;

    /// Change the position of an existing fingerprint. Panics if `fp` is absent.
    fn set(&mut self, fp: u64, pos: u32);

    /// Remove `fp`, returning its position.
    fn remove(&mut self, fp: u64) -> Option<u32>;

    /// Bits used by the mapper: entries `(fp_bits + pos_bits)` plus whatever the structure adds (pointers, counts).
    fn space_bits(&self) -> u64;

    /// Panics if an internal invariant is violated.
    fn check(&self);

    fn name() -> &'static str;
}

/// Pack `(fingerprint, position)` into one `u64` whose numeric order is the fingerprint order.
#[inline]
pub(crate) fn pack(fp: u64, pos: u32, pos_bits: u32) -> u64 {
    debug_assert!(pos_bits <= 16 && (pos as u64) < (1u64 << pos_bits));
    debug_assert!(fp < (1u64 << 48));
    (fp << pos_bits) | pos as u64
}

#[inline]
pub(crate) fn unpack_fp(e: u64, pos_bits: u32) -> u64 {
    e >> pos_bits
}

#[inline]
pub(crate) fn unpack_pos(e: u64, pos_bits: u32) -> u32 {
    (e & ((1u64 << pos_bits) - 1)) as u32
}

#[cfg(test)]
pub(crate) mod conformance {
    //! Model-based test every mapper must pass: random operations against a `BTreeMap` oracle.
    use super::*;
    use crate::rng::SplitMix64;
    use std::collections::BTreeMap;

    pub fn run<M: QueryMapper>(fp_bits: u32, pos_bits: u32, ops: usize, universe: u64, seed: u64) {
        let fp_bits = fp_bits.min(M::max_fp_bits());
        let pos_bits = pos_bits.min(M::max_pos_bits());
        let universe = universe.min(1u64 << fp_bits);
        let pos_mod = 1u64 << pos_bits;
        let mut m = M::new(fp_bits, pos_bits);
        let mut model: BTreeMap<u64, u32> = BTreeMap::new();
        let mut rng = SplitMix64::new(seed);
        for step in 0..ops {
            let fp = rng.below(universe);
            match rng.below(4) {
                0 | 1 => {
                    let pos = rng.below(pos_mod) as u32;
                    let expect_new = !model.contains_key(&fp);
                    assert_eq!(m.insert(fp, pos), expect_new, "insert result, step {step}");
                    if expect_new {
                        model.insert(fp, pos);
                    }
                }
                2 => {
                    assert_eq!(m.remove(fp), model.remove(&fp), "remove result, step {step}");
                }
                _ => {
                    if let Some(old) = model.get_mut(&fp) {
                        let pos = rng.below(pos_mod) as u32;
                        m.set(fp, pos);
                        *old = pos;
                    }
                }
            }
            assert_eq!(m.get(fp), model.get(&fp).copied(), "get, step {step}");
            assert_eq!(m.len(), model.len(), "len, step {step}");
            m.check();
            if step % 17 == 0 {
                for (k, v) in &model {
                    assert_eq!(m.get(*k), Some(*v));
                }
            }
        }
        // clear and reuse
        m.clear();
        assert_eq!(m.len(), 0);
        for k in model.keys() {
            assert_eq!(m.get(*k), None);
        }
        m.check();
    }
}
