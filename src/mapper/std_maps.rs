//! Query mappers built on the standard library collections.
//!
//! [`StdBTreeMapper`] is the closest relative of the thesis implementation, which used Java's `TreeMap`
//! (a red-black tree) instead of the paper's B-tree. [`StdHashMapper`] is a plain hash map, i.e. exactly the
//! thing the paper is *replacing* with a B-tree of tiny nodes; it is only a practical performance baseline.

use super::QueryMapper;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug)]
pub struct StdBTreeMapper {
    map: BTreeMap<u64, u32>,
    fp_bits: u32,
    pos_bits: u32,
}

impl QueryMapper for StdBTreeMapper {
    fn new(fp_bits: u32, pos_bits: u32) -> Self {
        StdBTreeMapper { map: BTreeMap::new(), fp_bits, pos_bits }
    }

    fn clear(&mut self) {
        self.map.clear();
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn get(&self, fp: u64) -> Option<u32> {
        self.map.get(&fp).copied()
    }

    fn insert(&mut self, fp: u64, pos: u32) -> bool {
        use std::collections::btree_map::Entry;
        match self.map.entry(fp) {
            Entry::Occupied(_) => false,
            Entry::Vacant(v) => {
                v.insert(pos);
                true
            }
        }
    }

    fn set(&mut self, fp: u64, pos: u32) {
        *self.map.get_mut(&fp).expect("set: fingerprint not present") = pos;
    }

    fn remove(&mut self, fp: u64) -> Option<u32> {
        self.map.remove(&fp)
    }

    fn space_bits(&self) -> u64 {
        // payload only; the std B-tree adds node headers and spare capacity that we do not try to account for
        self.map.len() as u64 * (self.fp_bits + self.pos_bits) as u64
    }

    fn check(&self) {}

    fn name() -> &'static str {
        "std-btree"
    }
}

#[derive(Clone, Debug)]
pub struct StdHashMapper {
    map: HashMap<u64, u32>,
    fp_bits: u32,
    pos_bits: u32,
}

impl QueryMapper for StdHashMapper {
    fn new(fp_bits: u32, pos_bits: u32) -> Self {
        StdHashMapper { map: HashMap::new(), fp_bits, pos_bits }
    }

    fn clear(&mut self) {
        self.map.clear();
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn get(&self, fp: u64) -> Option<u32> {
        self.map.get(&fp).copied()
    }

    fn insert(&mut self, fp: u64, pos: u32) -> bool {
        use std::collections::hash_map::Entry;
        match self.map.entry(fp) {
            Entry::Occupied(_) => false,
            Entry::Vacant(v) => {
                v.insert(pos);
                true
            }
        }
    }

    fn set(&mut self, fp: u64, pos: u32) {
        *self.map.get_mut(&fp).expect("set: fingerprint not present") = pos;
    }

    fn remove(&mut self, fp: u64) -> Option<u32> {
        self.map.remove(&fp)
    }

    fn space_bits(&self) -> u64 {
        self.map.len() as u64 * (self.fp_bits + self.pos_bits) as u64
    }

    fn check(&self) {}

    fn name() -> &'static str {
        "std-hash"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapper::conformance;

    #[test]
    fn std_btree_conforms() {
        conformance::run::<StdBTreeMapper>(20, 10, 10_000, 300, 3);
    }

    #[test]
    fn std_hash_conforms() {
        conformance::run::<StdHashMapper>(20, 10, 10_000, 300, 4);
    }
}
