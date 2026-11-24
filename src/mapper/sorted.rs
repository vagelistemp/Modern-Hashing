//! Sorted array + binary search. O(log m) lookups and O(m) inserts, but so simple that it is obviously
//! correct: it is the reference the more elaborate mappers are tested against.

use super::{pack, unpack_fp, unpack_pos, QueryMapper};

#[derive(Clone, Debug)]
pub struct SortedVecMapper {
    /// Entries `(fp << pos_bits) | pos`, sorted by fingerprint (the fingerprint occupies the high bits).
    entries: Vec<u64>,
    pos_bits: u32,
    fp_bits: u32,
}

impl SortedVecMapper {
    #[inline]
    fn find(&self, fp: u64) -> Result<usize, usize> {
        let pb = self.pos_bits;
        self.entries.binary_search_by(|e| unpack_fp(*e, pb).cmp(&fp))
    }
}

impl QueryMapper for SortedVecMapper {
    fn new(fp_bits: u32, pos_bits: u32) -> Self {
        SortedVecMapper { entries: Vec::new(), pos_bits, fp_bits }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn get(&self, fp: u64) -> Option<u32> {
        self.find(fp).ok().map(|i| unpack_pos(self.entries[i], self.pos_bits))
    }

    fn insert(&mut self, fp: u64, pos: u32) -> bool {
        match self.find(fp) {
            Ok(_) => false,
            Err(i) => {
                self.entries.insert(i, pack(fp, pos, self.pos_bits));
                true
            }
        }
    }

    fn set(&mut self, fp: u64, pos: u32) {
        let i = self.find(fp).expect("set: fingerprint not present");
        self.entries[i] = pack(fp, pos, self.pos_bits);
    }

    fn remove(&mut self, fp: u64) -> Option<u32> {
        match self.find(fp) {
            Ok(i) => Some(unpack_pos(self.entries.remove(i), self.pos_bits)),
            Err(_) => None,
        }
    }

    fn space_bits(&self) -> u64 {
        self.entries.len() as u64 * (self.fp_bits + self.pos_bits) as u64
    }

    fn check(&self) {
        let pb = self.pos_bits;
        for w in self.entries.windows(2) {
            assert!(unpack_fp(w[0], pb) < unpack_fp(w[1], pb), "entries not strictly sorted by fingerprint");
        }
    }

    fn name() -> &'static str {
        "sorted-vec"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapper::conformance;

    #[test]
    fn conforms_small_universe() {
        conformance::run::<SortedVecMapper>(6, 6, 20_000, 64, 1);
    }

    #[test]
    fn conforms_wide_fingerprints() {
        conformance::run::<SortedVecMapper>(39, 14, 20_000, 400, 2);
    }
}
