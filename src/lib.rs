//! Partition hashing, after Bender, Farach-Colton, J. Kuszmaul and W. Kuszmaul,
//! *Modern Hashing Made Simple* (SOSA 2024).
//!
//! The paper builds a hash table with constant-time operations from three ingredients that are taught in any
//! advanced data-structures course. This crate implements each ingredient on its own and the three tables on top:
//!
//! | paper     | structure                         | here                                                              |
//! |-----------|-----------------------------------|-------------------------------------------------------------------|
//! | Section 3 | Slow Partition Hash Table         | [`SlowPartition`]                                                  |
//! | Section 4 | Indexed Partition Hash Table      | [`IndexedPartition`] over a [`mapper::QueryMapper`]                |
//! | Section 4 | query mapper = B-tree + Four Russians | [`mapper::BTreeMapper`] with [`mapper::PackedNode`] / [`mapper::TabulatedNode`] |
//! | Section 5 | K-Tries: buffered insert/delete   | [`KTriesPartition`] = [`Buffered`] over [`ktrie::KTrie`]           |
//! | Section 6 | resizing                          | [`slab::SlabVec`] (local) and [`Resizable`] (global)               |
//!
//! Keys and values are `u32`. Every table implements [`Dictionary`].

pub mod buffered;
pub mod dict;
pub mod hash;
pub mod indexed;
pub mod ktrie;
pub mod mapper;
pub mod params;
pub mod partition;
pub mod resizable;
pub mod rng;
pub mod slab;
pub mod slow;

pub use buffered::{buffer_threshold, Buffered};
pub use dict::{Dictionary, Error, Key, SpaceReport, Stats, Val};
pub use indexed::{IndexedBucket, IndexedPartition, SlabIndexedPartition};
pub use mapper::QueryMapper;
pub use params::Params;
pub use partition::Partition;
pub use resizable::Resizable;
pub use slow::{SlowBucket, SlowPartition};

/// Section 5: the indexed partition table behind a pair of tries.
pub type KTriesPartition<M> = Buffered<IndexedPartition<M>>;

/// A Slow Partition table with the paper's geometry for capacity `n`.
pub fn slow(n: usize) -> SlowPartition {
    SlowPartition::new(Params::paper(n))
}

/// An Indexed Partition table with the paper's geometry for capacity `n`, using mapper `M`.
pub fn indexed<M: QueryMapper>(n: usize) -> IndexedPartition<M> {
    IndexedPartition::<M>::new(Params::paper(n))
}

/// A K-Tries table of logical capacity `n` (the table underneath gets the head-room for pending deletions).
pub fn ktries<M: QueryMapper>(n: usize) -> KTriesPartition<M> {
    let inner_cap = Buffered::<IndexedPartition<M>>::inner_capacity(n);
    let mut p = Params::paper(n);
    p.capacity = inner_cap;
    Buffered::new(IndexedPartition::<M>::new(p), n)
}
