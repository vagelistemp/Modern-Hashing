//! Model-based tests: every dictionary of the crate, on every query mapper, against a `HashMap` oracle.

mod common;

use common::{run_model, ModelCfg};
use modern_hashing::mapper::{
    BTreeMapper, PackedNode, SortedVecMapper, Spec3x3, Spec4x2, StdBTreeMapper, StdHashMapper, TabulatedNode,
};
use modern_hashing::{Buffered, Dictionary, IndexedPartition, Params, QueryMapper, SlowPartition};

type Packed = BTreeMapper<PackedNode>;
type Tab4x2 = BTreeMapper<TabulatedNode<Spec4x2>>;
type Tab3x3 = BTreeMapper<TabulatedNode<Spec3x3>>;

/// A small table whose geometry leaves room (mean load 100 of 160 slots per bucket) so that rebuilds are rare.
fn roomy(seed: u64) -> Params {
    Params::custom(400, 4, 160, 24).with_seed(seed)
}

#[test]
fn slow_partition_matches_the_model() {
    run_model(SlowPartition::new(roomy(1)), ModelCfg::new(700, 12_000, 1));
}

#[test]
fn indexed_sorted_vec_matches_the_model() {
    run_model(IndexedPartition::<SortedVecMapper>::new(roomy(2)), ModelCfg::new(700, 12_000, 2));
}

#[test]
fn indexed_std_btree_matches_the_model() {
    run_model(IndexedPartition::<StdBTreeMapper>::new(roomy(3)), ModelCfg::new(700, 12_000, 3));
}

#[test]
fn indexed_std_hash_matches_the_model() {
    run_model(IndexedPartition::<StdHashMapper>::new(roomy(4)), ModelCfg::new(700, 12_000, 4));
}

#[test]
fn indexed_packed_btree_matches_the_model() {
    // buckets of ~100 keys: the B-tree has 3-4 levels, so splits, borrows and merges all happen
    run_model(IndexedPartition::<Packed>::new(roomy(5)), ModelCfg::new(700, 12_000, 5));
}

#[test]
fn indexed_packed_btree_with_tiny_fingerprints_matches_the_model() {
    // 7-bit fingerprints for ~100 keys per bucket: collisions are constant, so every insertion path that rebuilds a
    // bucket's fingerprints (and, after repeated failures, widens them) is exercised
    let p = Params::custom(400, 4, 160, 7).with_seed(6);
    let d = run_model(IndexedPartition::<Packed>::new(p), ModelCfg::new(700, 6_000, 6));
    let s = d.stats();
    assert!(s.bucket_rebuilds > 10, "expected fingerprint rebuilds, got {s:?}");
    assert!(s.fp_widenings > 0, "expected the fingerprint width to have been increased, got {s:?}");
}

#[test]
fn indexed_tabulated_4x2_matches_the_model() {
    // 16 buckets of 4 slots, 4-bit fingerprints: every node operation is a table read, collisions are constant and
    // buckets overflow regularly, which forces whole-table rebuilds
    let p = Params::custom(40, 16, 4, 4).with_seed(7);
    let d = run_model(IndexedPartition::<Tab4x2>::new(p), ModelCfg::new(60, 20_000, 7));
    let s = d.stats();
    assert!(s.bucket_rebuilds > 50, "{s:?}");
    assert!(s.global_rebuilds > 5, "{s:?}");
}

#[test]
fn indexed_tabulated_3x3_matches_the_model() {
    let p = Params::custom(24, 8, 4, 3).with_seed(8);
    let d = run_model(IndexedPartition::<Tab3x3>::new(p), ModelCfg::new(40, 20_000, 8));
    assert!(d.stats().bucket_rebuilds > 50);
}

fn buffered<M: QueryMapper>(n: usize, seed: u64) -> Buffered<IndexedPartition<M>> {
    let mut p = Params::paper(n).with_seed(seed);
    p.capacity = Buffered::<IndexedPartition<M>>::inner_capacity(n);
    Buffered::new(IndexedPartition::<M>::new(p), n)
}

#[test]
fn ktries_over_packed_btree_matches_the_model() {
    let d = run_model(buffered::<Packed>(300, 9), ModelCfg::new(500, 15_000, 9));
    let s = d.stats();
    assert!(s.buffer_swaps > 50, "the tries should have exchanged roles many times: {s:?}");
    assert_eq!(s.forced_flushes, 0, "the shrinking trie must always be empty when the growing one fills: {s:?}");
}

#[test]
fn ktries_over_sorted_vec_matches_the_model() {
    run_model(buffered::<SortedVecMapper>(300, 10), ModelCfg::new(500, 15_000, 10));
}

#[test]
fn ktries_with_a_single_drain_per_operation_still_never_force_a_flush() {
    let d = run_model(buffered::<Packed>(400, 11).with_drain(1), ModelCfg::new(600, 15_000, 11));
    assert_eq!(d.stats().forced_flushes, 0);
}

#[test]
fn ktries_over_tabulated_nodes_match_the_model() {
    let n = 30;
    let mut p = Params::custom(Buffered::<IndexedPartition<Tab4x2>>::inner_capacity(n), 16, 4, 4).with_seed(12);
    p.capacity = Buffered::<IndexedPartition<Tab4x2>>::inner_capacity(n);
    let d = Buffered::new(IndexedPartition::<Tab4x2>::new(p), n);
    run_model(d, ModelCfg::new(50, 15_000, 12));
}

#[test]
fn ktries_over_slow_partition_match_the_model() {
    // Appendix A: the buffering technique applies to any hash table
    let n = 300;
    let mut p = Params::paper(n).with_seed(13);
    p.capacity = Buffered::<SlowPartition>::inner_capacity(n);
    run_model(Buffered::new(SlowPartition::new(p), n), ModelCfg::new(500, 15_000, 13));
}

#[test]
fn paper_sized_tables_survive_a_longer_random_run() {
    // paper-sized geometry for n = 3000: 1 bucket of ~1000+ slots, 28-bit fingerprints
    for seed in 0..3u64 {
        let n = 3000;
        let cfg = ModelCfg { universe: 4500, ops: 6_000, seed: 100 + seed, full_compare_every: 200 };
        run_model(IndexedPartition::<Packed>::new(Params::paper(n).with_seed(seed)), cfg);
        run_model(SlowPartition::new(Params::paper(n).with_seed(seed)), cfg);
    }
}
