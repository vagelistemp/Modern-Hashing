//! One named regression test per defect found in the thesis implementation (see the recap PDF, section 4,
//! items B1-B7). Each test fails for the Java semantics and passes for this crate.

mod common;

use modern_hashing::mapper::{
    BTreeMapper, PackedNode, SortedVecMapper, Spec3x3, Spec4x2, StdBTreeMapper, StdHashMapper, TabulatedNode,
};
use modern_hashing::rng::scramble32;
use modern_hashing::{Buffered, Dictionary, Error, IndexedPartition, Params, QueryMapper, SlowPartition};
use std::cell::Cell;

type Packed = BTreeMapper<PackedNode>;
type Tab4x2 = BTreeMapper<TabulatedNode<Spec4x2>>;
type Tab3x3 = BTreeMapper<TabulatedNode<Spec3x3>>;

// ------------------------------------------------------------------------------------------------------------------
// B1. Slow Partition: the insertion of a new key must not scan the bucket (the thesis `insert` did, O(bucket fill)).
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn b1_slow_insert_new_does_not_scan_the_bucket() {
    let mut t = SlowPartition::new(Params::paper(20_000).with_seed(1));
    for k in 0..14_000u32 {
        t.insert_new(scramble32(k), k).unwrap();
    }
    assert_eq!(t.slots_scanned(), 0, "insert_new must be O(1): it may not examine existing slots");
    assert_eq!(t.len(), 14_000);
    // the upsert has to look for an existing key and therefore scans the bucket (thousands of slots here)
    for k in 14_000..14_100u32 {
        assert_eq!(t.insert(scramble32(k), k), Ok(None));
    }
    assert!(t.slots_scanned() > 100 * 1_000, "upsert should scan: {}", t.slots_scanned());
    t.check_invariants();
}

// ------------------------------------------------------------------------------------------------------------------
// B2. Indexed: get and delete of ABSENT keys must verify the stored key (the thesis returned / deleted another key
//     whenever the fingerprints collided).
// ------------------------------------------------------------------------------------------------------------------

fn absent_keys_are_harmless<D: Dictionary>(mut d: D, present: u32, absent: u32) {
    for k in 0..present {
        d.insert_new(scramble32(k), k ^ 0x5555).unwrap();
    }
    let len = d.len();
    // scramble32 is a bijection, so these keys are guaranteed to be absent
    for k in present..present + absent {
        let key = scramble32(k);
        assert_eq!(d.get(key), None, "absent key {key} was reported present");
        assert!(!d.contains(key));
    }
    for k in present..present + absent {
        let key = scramble32(k);
        assert_eq!(d.remove(key), None, "removing the absent key {key} removed something");
    }
    assert_eq!(d.len(), len);
    for k in 0..present {
        assert_eq!(d.get(scramble32(k)), Some(k ^ 0x5555), "present key {} was damaged by absent-key operations", scramble32(k));
    }
    d.check_invariants();
}

#[test]
fn b2_absent_keys_do_not_disturb_present_keys_sorted_vec() {
    // 6-bit fingerprints and ~100 keys per bucket: every absent key collides with a present fingerprint
    absent_keys_are_harmless(IndexedPartition::<SortedVecMapper>::new(Params::custom(400, 4, 160, 6)), 300, 2_000);
}

#[test]
fn b2_absent_keys_do_not_disturb_present_keys_std_maps() {
    absent_keys_are_harmless(IndexedPartition::<StdBTreeMapper>::new(Params::custom(400, 4, 160, 6)), 300, 2_000);
    absent_keys_are_harmless(IndexedPartition::<StdHashMapper>::new(Params::custom(400, 4, 160, 6)), 300, 2_000);
}

#[test]
fn b2_absent_keys_do_not_disturb_present_keys_packed_btree() {
    absent_keys_are_harmless(IndexedPartition::<Packed>::new(Params::custom(400, 4, 160, 6)), 300, 2_000);
    absent_keys_are_harmless(IndexedPartition::<Packed>::new(Params::paper(4_000)), 3_000, 3_000);
}

#[test]
fn b2_absent_keys_do_not_disturb_present_keys_tabulated() {
    absent_keys_are_harmless(IndexedPartition::<Tab4x2>::new(Params::custom(40, 16, 4, 4)), 30, 500);
    absent_keys_are_harmless(IndexedPartition::<Tab3x3>::new(Params::custom(24, 8, 4, 3)), 18, 500);
}

#[test]
fn b2_absent_keys_do_not_disturb_present_keys_slow_and_ktries() {
    absent_keys_are_harmless(SlowPartition::new(Params::custom(400, 4, 160, 6)), 300, 500);
    let n = 400;
    let mut p = Params::custom(400, 4, 160, 6);
    p.capacity = Buffered::<IndexedPartition<Packed>>::inner_capacity(n);
    absent_keys_are_harmless(Buffered::new(IndexedPartition::<Packed>::new(p), n), 300, 2_000);
}

// ------------------------------------------------------------------------------------------------------------------
// B3 / B4. Indexed: delete moves ONE item and updates ONE mapper entry; a query is ONE mapper lookup.
//          (The thesis shifted the whole bucket and renumbered every mapper entry; its linked list made each slot
//          access O(position).)
// ------------------------------------------------------------------------------------------------------------------

thread_local! {
    static GETS: Cell<usize> = const { Cell::new(0) };
    static SETS: Cell<usize> = const { Cell::new(0) };
    static REMOVES: Cell<usize> = const { Cell::new(0) };
    static INSERTS: Cell<usize> = const { Cell::new(0) };
}

fn reset_counters() {
    GETS.with(|c| c.set(0));
    SETS.with(|c| c.set(0));
    REMOVES.with(|c| c.set(0));
    INSERTS.with(|c| c.set(0));
}

fn counters() -> (usize, usize, usize, usize) {
    (GETS.with(|c| c.get()), SETS.with(|c| c.get()), REMOVES.with(|c| c.get()), INSERTS.with(|c| c.get()))
}

/// A mapper that counts the operations the table performs on it.
struct Counting(StdBTreeMapper);

impl QueryMapper for Counting {
    fn new(fp_bits: u32, pos_bits: u32) -> Self {
        Counting(StdBTreeMapper::new(fp_bits, pos_bits))
    }
    fn clear(&mut self) {
        self.0.clear()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, fp: u64) -> Option<u32> {
        GETS.with(|c| c.set(c.get() + 1));
        self.0.get(fp)
    }
    fn insert(&mut self, fp: u64, pos: u32) -> bool {
        INSERTS.with(|c| c.set(c.get() + 1));
        self.0.insert(fp, pos)
    }
    fn set(&mut self, fp: u64, pos: u32) {
        SETS.with(|c| c.set(c.get() + 1));
        self.0.set(fp, pos)
    }
    fn remove(&mut self, fp: u64) -> Option<u32> {
        REMOVES.with(|c| c.set(c.get() + 1));
        self.0.remove(fp)
    }
    fn space_bits(&self) -> u64 {
        self.0.space_bits()
    }
    fn check(&self) {}
    fn name() -> &'static str {
        "counting"
    }
}

fn filled_single_bucket(keys: usize) -> IndexedPartition<Counting> {
    // one bucket, 40-bit fingerprints: no fingerprint collisions, so the counters see only the operation under test
    let mut t = IndexedPartition::<Counting>::new(Params::custom(keys + 10, 1, keys + 10, 40).with_seed(3));
    for k in 0..keys as u32 {
        t.insert_new(k, k).unwrap();
    }
    t
}

#[test]
fn b3_delete_updates_exactly_one_mapper_entry_whatever_the_bucket_size() {
    for &size in &[10usize, 100, 2_000] {
        // delete an item in the middle: one lookup, one removal, ONE position update (for the item moved into the hole)
        let mut t = filled_single_bucket(size);
        reset_counters();
        assert_eq!(t.remove(size as u32 / 2), Some(size as u32 / 2));
        assert_eq!(counters(), (1, 1, 1, 0), "size {size}: (get, set, remove, insert) after a mid-bucket delete");
        // delete the last item: nothing has to move, so no position update at all
        let mut t = filled_single_bucket(size);
        reset_counters();
        assert_eq!(t.remove(size as u32 - 1), Some(size as u32 - 1));
        assert_eq!(counters(), (1, 0, 1, 0), "size {size}: after deleting the last item");
        t.check_invariants();
    }
}

#[test]
fn b4_a_query_is_one_mapper_lookup_whatever_the_bucket_size() {
    for &size in &[10usize, 2_000] {
        let t = filled_single_bucket(size);
        reset_counters();
        assert_eq!(t.get(size as u32 / 3), Some(size as u32 / 3));
        assert_eq!(counters(), (1, 0, 0, 0), "size {size}: a hit costs one mapper lookup");
        reset_counters();
        assert_eq!(t.get(1_000_000), None);
        assert_eq!(counters(), (1, 0, 0, 0), "size {size}: a miss costs one mapper lookup");
    }
}

// ------------------------------------------------------------------------------------------------------------------
// B5. Rebuilds (whole table, and one bucket's fingerprints) must never lose or corrupt a key. The thesis `rebuild()`
//     inserted fingerprints with a plain `put`, silently overwriting colliding ones.
// ------------------------------------------------------------------------------------------------------------------

fn forced_global_rebuilds<M: QueryMapper>(capacity: usize, m: usize, cap: usize, fp_bits: u32, seed: u64) {
    let p = Params::custom(capacity, m, cap, fp_bits).with_seed(seed);
    let mut t = IndexedPartition::<M>::new(p);
    for k in 0..capacity as u32 {
        t.insert_new(scramble32(k), k ^ 0xABCD).unwrap_or_else(|e| panic!("insert {k} failed: {e}"));
    }
    let s = t.stats();
    assert!(s.global_rebuilds > 0, "the geometry was supposed to force whole-table rebuilds: {s:?}");
    t.check_invariants();
    for k in 0..capacity as u32 {
        assert_eq!(t.get(scramble32(k)), Some(k ^ 0xABCD), "key {k} unreachable after {} rebuilds", s.global_rebuilds);
    }
}

#[test]
fn b5_whole_table_rebuilds_lose_no_key_packed_btree() {
    // the thesis experiment: 10 000 keys in 12 buckets that are too tight, so that buckets overflow and the table is
    // rebuilt again and again; with wide, medium and tiny fingerprints
    forced_global_rebuilds::<Packed>(10_000, 12, 860, 39, 1);
    forced_global_rebuilds::<Packed>(10_000, 12, 860, 12, 2);
    forced_global_rebuilds::<Packed>(10_000, 12, 860, 8, 3);
}

#[test]
fn b5_whole_table_rebuilds_lose_no_key_other_mappers() {
    forced_global_rebuilds::<SortedVecMapper>(3_000, 8, 390, 20, 4);
    forced_global_rebuilds::<StdBTreeMapper>(3_000, 8, 390, 9, 5);
}

#[test]
fn b5_whole_table_rebuilds_lose_no_key_tabulated() {
    forced_global_rebuilds::<Tab4x2>(40, 16, 4, 4, 6);
}

#[test]
fn b5_a_failed_rebuild_loses_nothing() {
    // 20 slots in total cannot hold 100 keys: sooner or later insertion fails, and the table must stay intact
    let mut t = IndexedPartition::<Packed>::new(Params::custom(100, 2, 10, 16).with_seed(7));
    let mut stored = Vec::new();
    let mut failed = None;
    for k in 0..100u32 {
        let key = scramble32(k);
        match t.insert_new(key, k) {
            Ok(()) => stored.push((key, k)),
            Err(e) => {
                failed = Some((key, e));
                break;
            }
        }
    }
    let (failed_key, err) = failed.expect("20 slots cannot hold 100 keys");
    assert_eq!(err, Error::RebuildFailed);
    assert_eq!(t.get(failed_key), None, "the key whose insertion failed must not be half-inserted");
    assert!(stored.len() <= 20);
    for (key, v) in stored {
        assert_eq!(t.get(key), Some(v));
    }
    t.check_invariants();
}

// ------------------------------------------------------------------------------------------------------------------
// B6. Every u32 is a legal key (the thesis crashed on negative keys with an index out of bounds).
// ------------------------------------------------------------------------------------------------------------------

fn extreme_keys<D: Dictionary>(mut d: D) {
    let keys = [0u32, 1, u32::MAX, u32::MAX - 1, 0x8000_0000, 0x7FFF_FFFF, 0xFFFF_0000, 0x0000_FFFF];
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(d.insert(k, i as u32 + 100), Ok(None), "{} key {k:#x}", d.name());
        d.check_invariants();
    }
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(d.get(k), Some(i as u32 + 100), "{} key {k:#x}", d.name());
    }
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(d.remove(k), Some(i as u32 + 100));
        assert_eq!(d.get(k), None);
        d.check_invariants();
    }
    assert!(d.is_empty());
}

#[test]
fn b6_keys_zero_and_u32_max_and_the_sign_bit_work_everywhere() {
    extreme_keys(SlowPartition::new(Params::paper(1_000)));
    extreme_keys(IndexedPartition::<SortedVecMapper>::new(Params::paper(1_000)));
    extreme_keys(IndexedPartition::<StdBTreeMapper>::new(Params::paper(1_000)));
    extreme_keys(IndexedPartition::<StdHashMapper>::new(Params::paper(1_000)));
    extreme_keys(IndexedPartition::<Packed>::new(Params::paper(1_000)));
    extreme_keys(IndexedPartition::<Tab4x2>::new(Params::custom(40, 16, 4, 4)));
    extreme_keys(modern_hashing::ktries::<Packed>(1_000));
}

// ------------------------------------------------------------------------------------------------------------------
// B7. K-Tries: an acknowledged operation is visible at once, before the buffer has applied it (the thesis `get` read
//     only the table).
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn b7_put_is_visible_immediately_and_stays_in_the_buffer() {
    let mut d = modern_hashing::ktries::<Packed>(1_000);
    assert!(d.threshold() > 2);
    d.insert_new(7, 70).unwrap();
    assert_eq!(d.get(7), Some(70), "the key that was just inserted must be visible");
    assert_eq!(d.pending(), 1, "...even though it has not been applied to the table yet");
    assert_eq!(d.inner().len(), 0, "the table itself has not seen it");
    assert_eq!(d.len(), 1);
}

#[test]
fn b7_overwrite_and_delete_are_visible_immediately() {
    let mut d = modern_hashing::ktries::<Packed>(1_000);
    d.insert_new(7, 70).unwrap();
    assert_eq!(d.insert(7, 71), Ok(Some(70)));
    assert_eq!(d.get(7), Some(71));
    assert_eq!(d.remove(7), Some(71));
    assert_eq!(d.get(7), None);
    assert_eq!(d.remove(7), None);
    assert_eq!(d.len(), 0);
    d.insert_new(7, 72).unwrap();
    assert_eq!(d.get(7), Some(72));
    d.check_invariants();
}

#[test]
fn b7_visibility_holds_while_buffers_fill_swap_and_drain() {
    let n = 2_000;
    let mut d = modern_hashing::ktries::<Packed>(n);
    let threshold = d.threshold();
    // enough operations for many swaps; after every single operation, ALL keys inserted so far are visible
    let mut live: Vec<u32> = Vec::new();
    for i in 0..(10 * threshold as u32) {
        let key = scramble32(i);
        d.insert_new(key, i).unwrap();
        live.push(key);
        assert_eq!(d.get(key), Some(i));
        if i % 3 == 2 {
            let victim = live.remove((i as usize * 7) % live.len());
            assert!(d.remove(victim).is_some());
            assert_eq!(d.get(victim), None);
        }
        d.check_invariants();
        for &k in &live {
            assert!(d.get(k).is_some(), "key {k} disappeared after operation {i}");
        }
    }
    let s = d.stats();
    assert!(s.buffer_swaps >= 5, "{s:?}");
    assert_eq!(s.forced_flushes, 0);
    // flushing applies everything and changes nothing observable
    let before: Vec<_> = live.iter().map(|&k| d.get(k)).collect();
    d.flush().unwrap();
    assert_eq!(d.pending(), 0);
    let after: Vec<_> = live.iter().map(|&k| d.get(k)).collect();
    assert_eq!(before, after);
    assert_eq!(d.inner().len(), d.len());
}

// ------------------------------------------------------------------------------------------------------------------
// Contract tests that do not correspond to a thesis defect.
// ------------------------------------------------------------------------------------------------------------------

#[test]
fn capacity_is_enforced_and_freed_by_removal() {
    for mut d in [
        Box::new(SlowPartition::new(Params::paper(100))) as Box<dyn Dictionary>,
        Box::new(IndexedPartition::<Packed>::new(Params::paper(100))),
        Box::new(modern_hashing::ktries::<Packed>(100)),
    ] {
        for k in 0..100u32 {
            d.insert_new(scramble32(k), k).unwrap();
        }
        assert_eq!(d.insert(scramble32(1_000), 1), Err(Error::CapacityExceeded), "{}", d.name());
        assert_eq!(d.insert_new(scramble32(1_001), 1), Err(Error::CapacityExceeded), "{}", d.name());
        // overwriting an existing key is still fine at capacity
        assert_eq!(d.insert(scramble32(5), 55), Ok(Some(5)));
        assert_eq!(d.remove(scramble32(3)), Some(3));
        assert_eq!(d.insert_new(scramble32(1_000), 1), Ok(()), "{}", d.name());
        d.check_invariants();
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "already present")]
fn insert_new_of_an_existing_key_is_caught_in_debug_builds_slow() {
    let mut t = SlowPartition::new(Params::paper(100));
    t.insert_new(5, 1).unwrap();
    let _ = t.insert_new(5, 2);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "already present")]
fn insert_new_of_an_existing_key_is_caught_in_debug_builds_indexed() {
    let mut t = IndexedPartition::<Packed>::new(Params::paper(100));
    t.insert_new(5, 1).unwrap();
    let _ = t.insert_new(5, 2);
}
