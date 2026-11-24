//! Exhaustive subtabulation (the Method of Four Russians), exactly as Lemma 4.2 of the paper uses it.
//!
//! > "The fact that each node fits in o(log n) bits means that we can use exhaustive subtabulation in order to
//! > implement node-level operations (queries, modifications, insertions, deletions) in O(1) time."
//!
//! The whole contents of a B-tree node (how many entries it has and every `(fingerprint, position)` pair) is one
//! small integer, a *state*. There are only a few tens of thousands of states, so we **pre-compute** the outcome of
//! every operation on every state once, and from then on a node operation is a single read from a table:
//!
//! | operation            | table                                  | result                                     |
//! |----------------------|----------------------------------------|--------------------------------------------|
//! | search `fp`          | `search[state][fp]`                    | found-flag and index, or the insertion rank |
//! | insert `(fp, pos)`   | `insert[state][fp][pos]`               | new state                                   |
//! | remove entry `i`     | `remove[state][i]`                     | new state                                   |
//! | change position      | `setval[state][i][pos]`                | new state                                   |
//! | split a full node    | `split[state]`                         | left state, right state, median             |
//! | merge two nodes      | `merge[left][right][median]`           | merged state                                |
//!
//! This only works while a node is tiny: with `KB`-bit fingerprints, `VB`-bit positions and 3 entries per node the
//! number of states is `sum_{c<=3} C(2^KB, c) * 2^(VB*c)`. For the default spec (4-bit fingerprints, 2-bit positions)
//! that is 37 825 states; with the 39-bit fingerprints of a paper-sized table it would be astronomically many.
//! That is precisely why [`super::btree::PackedNode`] exists, and why the paper's argument is asymptotic (node size
//! `O(sqrt(log n) * log log n) = o(log n)` bits). The `tabulated` mapper is the faithful, verified demonstration of
//! the technique, plugged into a real (small) indexed hash table.

use super::btree::NodeOps;
use std::marker::PhantomData;
use std::sync::OnceLock;

/// Maximum entries per tabulated node (minimum degree t = 2).
pub const TK: usize = 3;
const NOPE: u32 = u32::MAX;

/// A compile-time choice of widths for a tabulated node.
pub trait TabSpec: Clone + 'static {
    /// Fingerprint width in bits.
    const KB: u32;
    /// Position width in bits.
    const VB: u32;
    fn tables() -> &'static Tables;
}

/// 4-bit fingerprints (16 values), 2-bit positions (buckets of at most 4 slots).
#[derive(Clone, Debug)]
pub struct Spec4x2;

impl TabSpec for Spec4x2 {
    const KB: u32 = 4;
    const VB: u32 = 2;
    fn tables() -> &'static Tables {
        static T: OnceLock<Tables> = OnceLock::new();
        T.get_or_init(|| Tables::build(4, 2))
    }
}

/// 3-bit fingerprints (8 values), 3-bit positions (buckets of at most 8 slots, but only 8 fingerprints).
#[derive(Clone, Debug)]
pub struct Spec3x3;

impl TabSpec for Spec3x3 {
    const KB: u32 = 3;
    const VB: u32 = 3;
    fn tables() -> &'static Tables {
        static T: OnceLock<Tables> = OnceLock::new();
        T.get_or_init(|| Tables::build(3, 3))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SplitEntry {
    pub left: u32,
    pub right: u32,
    pub median_key: u32,
    pub median_val: u32,
}

/// All pre-computed tables for one `(KB, VB)` pair.
pub struct Tables {
    pub kb: u32,
    pub vb: u32,
    pub nkeys: usize,
    pub nvals: usize,
    /// Number of states in total.
    pub num_states: usize,
    /// `count_start[c]` = first state id with `c` entries (`count_start[TK + 1]` = number of states).
    pub count_start: [usize; TK + 2],
    /// state id -> packed contents (count in the low 2 bits, then `(key | val << kb)` per entry).
    pub decode: Vec<u32>,
    search: Vec<u8>,
    insert: Vec<u32>,
    remove: Vec<u32>,
    setval: Vec<u32>,
    split: Vec<SplitEntry>,
    merge: Vec<u32>,
}

impl Tables {
    fn entry_bits(kb: u32, vb: u32) -> u32 {
        kb + vb
    }

    /// Pack a sorted list of `(key, val)` into the canonical raw state code.
    pub fn encode(kb: u32, vb: u32, entries: &[(u32, u32)]) -> u32 {
        assert!(entries.len() <= TK);
        let eb = Self::entry_bits(kb, vb);
        let mut raw = entries.len() as u32;
        for (i, &(k, v)) in entries.iter().enumerate() {
            debug_assert!(k < (1 << kb) && v < (1 << vb));
            raw |= (k | (v << kb)) << (2 + i as u32 * eb);
        }
        raw
    }

    pub fn raw_entries(kb: u32, vb: u32, raw: u32) -> Vec<(u32, u32)> {
        let eb = Self::entry_bits(kb, vb);
        let c = (raw & 3) as usize;
        (0..c)
            .map(|i| {
                let e = raw >> (2 + i as u32 * eb);
                (e & ((1 << kb) - 1), (e >> kb) & ((1 << vb) - 1))
            })
            .collect()
    }

    /// Enumerate every valid node (sorted, distinct keys, `0..=TK` entries) and pre-compute every operation.
    pub fn build(kb: u32, vb: u32) -> Tables {
        let nkeys = 1usize << kb;
        let nvals = 1usize << vb;
        // ---- states, grouped by entry count ----
        let mut decode: Vec<u32> = Vec::new();
        let mut count_start = [0usize; TK + 2];
        let mut cur: Vec<(u32, u32)> = Vec::new();
        for c in 0..=TK {
            count_start[c] = decode.len();
            fn rec(
                kb: u32,
                vb: u32,
                c: usize,
                next_key: u32,
                cur: &mut Vec<(u32, u32)>,
                out: &mut Vec<u32>,
            ) {
                if cur.len() == c {
                    out.push(Tables::encode(kb, vb, cur));
                    return;
                }
                for k in next_key..(1u32 << kb) {
                    for v in 0..(1u32 << vb) {
                        cur.push((k, v));
                        rec(kb, vb, c, k + 1, cur, out);
                        cur.pop();
                    }
                }
            }
            rec(kb, vb, c, 0, &mut cur, &mut decode);
        }
        count_start[TK + 1] = decode.len();
        let num_states = decode.len();
        let id_of: std::collections::HashMap<u32, u32> =
            decode.iter().enumerate().map(|(i, &r)| (r, i as u32)).collect();
        let id = |entries: &[(u32, u32)]| -> u32 { id_of[&Tables::encode(kb, vb, entries)] };

        // ---- search ----
        let mut search = vec![0u8; num_states * nkeys];
        for s in 0..num_states {
            let es = Self::raw_entries(kb, vb, decode[s]);
            for fp in 0..nkeys as u32 {
                search[s * nkeys + fp as usize] = match es.binary_search_by(|e| e.0.cmp(&fp)) {
                    Ok(i) => 0x80 | i as u8,
                    Err(i) => i as u8,
                };
            }
        }
        // ---- insert (only states with at most TK-1 entries can accept an entry) ----
        let insertable = count_start[TK];
        let mut insert = vec![NOPE; insertable * nkeys * nvals];
        for s in 0..insertable {
            let es = Self::raw_entries(kb, vb, decode[s]);
            for fp in 0..nkeys as u32 {
                if let Err(rank) = es.binary_search_by(|e| e.0.cmp(&fp)) {
                    for pos in 0..nvals as u32 {
                        let mut e2 = es.clone();
                        e2.insert(rank, (fp, pos));
                        insert[(s * nkeys + fp as usize) * nvals + pos as usize] = id(&e2);
                    }
                }
            }
        }
        // ---- remove / setval ----
        let mut remove = vec![NOPE; num_states * TK];
        let mut setval = vec![NOPE; num_states * TK * nvals];
        for s in 0..num_states {
            let es = Self::raw_entries(kb, vb, decode[s]);
            for i in 0..es.len() {
                let mut e2 = es.clone();
                e2.remove(i);
                remove[s * TK + i] = id(&e2);
                for v in 0..nvals as u32 {
                    let mut e3 = es.clone();
                    e3[i].1 = v;
                    setval[(s * TK + i) * nvals + v as usize] = id(&e3);
                }
            }
        }
        // ---- split (full nodes, t = 2: 1 entry left, the median, 1 entry right) ----
        let mut split = Vec::with_capacity(num_states - count_start[TK]);
        for s in count_start[TK]..num_states {
            let es = Self::raw_entries(kb, vb, decode[s]);
            split.push(SplitEntry {
                left: id(&es[0..1]),
                right: id(&es[2..3]),
                median_key: es[1].0,
                median_val: es[1].1,
            });
        }
        // ---- merge (two nodes with t - 1 = 1 entry each, around a median) ----
        let n1 = count_start[2] - count_start[1];
        let nmed = nkeys * nvals;
        let mut merge = vec![NOPE; n1 * n1 * nmed];
        for l in 0..n1 {
            let le = Self::raw_entries(kb, vb, decode[count_start[1] + l]);
            for r in 0..n1 {
                let re = Self::raw_entries(kb, vb, decode[count_start[1] + r]);
                for mk in 0..nkeys as u32 {
                    for mv in 0..nvals as u32 {
                        if le[0].0 < mk && mk < re[0].0 {
                            let e = [le[0], (mk, mv), re[0]];
                            merge[(l * n1 + r) * nmed + mk as usize * nvals + mv as usize] = id(&e);
                        }
                    }
                }
            }
        }
        Tables { kb, vb, nkeys, nvals, num_states, count_start, decode, search, insert, remove, setval, split, merge }
    }

    /// Total number of table entries (for the documentation / benchmark report).
    pub fn table_entries(&self) -> usize {
        self.search.len() + self.insert.len() + self.remove.len() + self.setval.len() + self.split.len() + self.merge.len()
    }

    // ---- the O(1) operations: each is a single indexed read ----

    #[inline]
    pub fn len_of(&self, s: u32) -> usize {
        (self.decode[s as usize] & 3) as usize
    }

    #[inline]
    pub fn key_of(&self, s: u32, i: usize) -> u32 {
        let eb = self.kb + self.vb;
        (self.decode[s as usize] >> (2 + i as u32 * eb)) & ((1 << self.kb) - 1)
    }

    #[inline]
    pub fn val_of(&self, s: u32, i: usize) -> u32 {
        let eb = self.kb + self.vb;
        (self.decode[s as usize] >> (2 + i as u32 * eb + self.kb)) & ((1 << self.vb) - 1)
    }

    #[inline]
    pub fn search_t(&self, s: u32, fp: u32) -> u8 {
        self.search[s as usize * self.nkeys + fp as usize]
    }

    #[inline]
    pub fn insert_t(&self, s: u32, fp: u32, pos: u32) -> u32 {
        self.insert[(s as usize * self.nkeys + fp as usize) * self.nvals + pos as usize]
    }

    #[inline]
    pub fn remove_t(&self, s: u32, i: usize) -> u32 {
        self.remove[s as usize * TK + i]
    }

    #[inline]
    pub fn setval_t(&self, s: u32, i: usize, v: u32) -> u32 {
        self.setval[(s as usize * TK + i) * self.nvals + v as usize]
    }

    #[inline]
    pub fn split_t(&self, s: u32) -> SplitEntry {
        self.split[s as usize - self.count_start[TK]]
    }

    #[inline]
    pub fn merge_t(&self, left: u32, right: u32, mk: u32, mv: u32) -> u32 {
        let n1 = self.count_start[2] - self.count_start[1];
        let l = left as usize - self.count_start[1];
        let r = right as usize - self.count_start[1];
        self.merge[(l * n1 + r) * (self.nkeys * self.nvals) + mk as usize * self.nvals + mv as usize]
    }
}

/// A B-tree node whose entire state is one `u32` index into the pre-computed tables.
#[derive(Clone, Debug)]
pub struct TabulatedNode<S: TabSpec> {
    state: u32,
    _spec: PhantomData<S>,
}

impl<S: TabSpec> TabulatedNode<S> {
    /// Raw state id (exposed for tests).
    pub fn state(&self) -> u32 {
        self.state
    }
}

impl<S: TabSpec> NodeOps for TabulatedNode<S> {
    const K: usize = TK;

    fn empty(_fp_bits: u32, _pos_bits: u32) -> Self {
        TabulatedNode { state: 0, _spec: PhantomData }
    }

    #[inline]
    fn len(&self) -> usize {
        S::tables().len_of(self.state)
    }

    #[inline]
    fn search(&self, fp: u64) -> Result<usize, usize> {
        debug_assert!(fp < (1u64 << S::KB), "fingerprint wider than the tabulated node supports");
        let v = S::tables().search_t(self.state, fp as u32);
        if v & 0x80 != 0 {
            Ok((v & 0x7f) as usize)
        } else {
            Err(v as usize)
        }
    }

    #[inline]
    fn key(&self, i: usize) -> u64 {
        S::tables().key_of(self.state, i) as u64
    }

    #[inline]
    fn val(&self, i: usize) -> u32 {
        S::tables().val_of(self.state, i)
    }

    fn set_val(&mut self, i: usize, pos: u32) {
        let n = S::tables().setval_t(self.state, i, pos);
        debug_assert_ne!(n, NOPE);
        self.state = n;
    }

    fn insert_at(&mut self, i: usize, fp: u64, pos: u32) {
        let t = S::tables();
        debug_assert_eq!(t.search_t(self.state, fp as u32) as usize, i, "insert rank mismatch");
        let n = t.insert_t(self.state, fp as u32, pos);
        assert_ne!(n, NOPE, "insert into a full node or a duplicate key");
        self.state = n;
    }

    fn remove_at(&mut self, i: usize) -> (u64, u32) {
        let t = S::tables();
        let kv = (t.key_of(self.state, i) as u64, t.val_of(self.state, i));
        let n = t.remove_t(self.state, i);
        assert_ne!(n, NOPE, "remove index out of range");
        self.state = n;
        kv
    }

    fn split(&mut self) -> ((u64, u32), Self) {
        let e = S::tables().split_t(self.state);
        self.state = e.left;
        ((e.median_key as u64, e.median_val), TabulatedNode { state: e.right, _spec: PhantomData })
    }

    fn merge(&mut self, median: (u64, u32), right: &Self) {
        let n = S::tables().merge_t(self.state, right.state, median.0 as u32, median.1);
        assert_ne!(n, NOPE, "merge of nodes that are not adjacent around the median");
        self.state = n;
    }

    fn max_fp_bits() -> u32 {
        S::KB
    }

    fn max_pos_bits() -> u32 {
        S::VB
    }

    fn name() -> &'static str {
        match (S::KB, S::VB) {
            (4, 2) => "btree-tabulated-4x2",
            (3, 3) => "btree-tabulated-3x3",
            _ => "btree-tabulated",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapper::{conformance, BTreeMapper, QueryMapper};

    fn expected_state_count(kb: u32, vb: u32) -> usize {
        fn binom(n: usize, k: usize) -> usize {
            (0..k).fold(1usize, |acc, i| acc * (n - i) / (i + 1))
        }
        let nk = 1usize << kb;
        let nv = 1usize << vb;
        (0..=TK).map(|c| binom(nk, c) * nv.pow(c as u32)).sum()
    }

    #[test]
    fn state_counts_match_the_formula() {
        assert_eq!(Spec4x2::tables().num_states, expected_state_count(4, 2));
        assert_eq!(Spec4x2::tables().num_states, 37_825);
        assert_eq!(Spec3x3::tables().num_states, expected_state_count(3, 3));
    }

    #[test]
    fn encode_decode_roundtrip_for_every_state() {
        for t in [Spec4x2::tables(), Spec3x3::tables()] {
            for s in 0..t.num_states {
                let es = Tables::raw_entries(t.kb, t.vb, t.decode[s]);
                assert_eq!(Tables::encode(t.kb, t.vb, &es), t.decode[s]);
                for (i, e) in es.iter().enumerate() {
                    assert_eq!((t.key_of(s as u32, i), t.val_of(s as u32, i)), *e);
                }
                assert_eq!(t.len_of(s as u32), es.len());
                assert!(es.windows(2).all(|w| w[0].0 < w[1].0), "keys of a state must be strictly increasing");
            }
        }
    }

    /// The Four Russians check: for EVERY node state and EVERY operation, the table read gives exactly what a naive
    /// sorted-vector implementation computes.
    fn exhaustive_check<S: TabSpec>() {
        let t = S::tables();
        let mut checked = 0u64;
        for s in 0..t.num_states as u32 {
            let es = Tables::raw_entries(t.kb, t.vb, t.decode[s as usize]);
            let node = TabulatedNode::<S> { state: s, _spec: PhantomData };
            // search: all fingerprints
            for fp in 0..(1u64 << S::KB) {
                assert_eq!(node.search(fp), es.binary_search_by(|e| (e.0 as u64).cmp(&fp)));
                checked += 1;
            }
            // insert: every absent fingerprint, every position, when there is room
            if es.len() < TK {
                for fp in 0..(1u32 << S::KB) {
                    if let Err(rank) = es.binary_search_by(|e| e.0.cmp(&fp)) {
                        for pos in 0..(1u32 << S::VB) {
                            let mut n2 = node.clone();
                            n2.insert_at(rank, fp as u64, pos);
                            let mut e2 = es.clone();
                            e2.insert(rank, (fp, pos));
                            assert_eq!(Tables::raw_entries(t.kb, t.vb, t.decode[n2.state as usize]), e2);
                            checked += 1;
                        }
                    }
                }
            }
            // remove and set_val for every index
            for i in 0..es.len() {
                let mut n2 = node.clone();
                let kv = n2.remove_at(i);
                let mut e2 = es.clone();
                let rem = e2.remove(i);
                assert_eq!(kv, (rem.0 as u64, rem.1));
                assert_eq!(Tables::raw_entries(t.kb, t.vb, t.decode[n2.state as usize]), e2);
                for v in 0..(1u32 << S::VB) {
                    let mut n3 = node.clone();
                    n3.set_val(i, v);
                    let mut e3 = es.clone();
                    e3[i].1 = v;
                    assert_eq!(Tables::raw_entries(t.kb, t.vb, t.decode[n3.state as usize]), e3);
                    checked += 1;
                }
                checked += 1;
            }
            // split of every full node
            if es.len() == TK {
                let mut left = node.clone();
                let (med, right) = left.split();
                assert_eq!(Tables::raw_entries(t.kb, t.vb, t.decode[left.state as usize]), es[0..1].to_vec());
                assert_eq!(Tables::raw_entries(t.kb, t.vb, t.decode[right.state as usize]), es[2..3].to_vec());
                assert_eq!(med, (es[1].0 as u64, es[1].1));
                checked += 1;
            }
        }
        // merge: every pair of one-entry nodes and every median in between
        let n1 = t.count_start[2] - t.count_start[1];
        for l in 0..n1 {
            for r in 0..n1 {
                let ls = (t.count_start[1] + l) as u32;
                let rs = (t.count_start[1] + r) as u32;
                let le = Tables::raw_entries(t.kb, t.vb, t.decode[ls as usize]);
                let re = Tables::raw_entries(t.kb, t.vb, t.decode[rs as usize]);
                for mk in 0..(1u32 << S::KB) {
                    for mv in 0..(1u32 << S::VB) {
                        let got = t.merge_t(ls, rs, mk, mv);
                        if le[0].0 < mk && mk < re[0].0 {
                            assert_eq!(
                                Tables::raw_entries(t.kb, t.vb, t.decode[got as usize]),
                                vec![le[0], (mk, mv), re[0]]
                            );
                        } else {
                            assert_eq!(got, NOPE);
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn exhaustive_4x2() {
        exhaustive_check::<Spec4x2>();
    }

    #[test]
    fn exhaustive_3x3() {
        exhaustive_check::<Spec3x3>();
    }

    #[test]
    fn tabulated_btree_conforms_4x2() {
        conformance::run::<BTreeMapper<TabulatedNode<Spec4x2>>>(4, 2, 30_000, 16, 21);
    }

    #[test]
    fn tabulated_btree_conforms_3x3() {
        conformance::run::<BTreeMapper<TabulatedNode<Spec3x3>>>(3, 3, 30_000, 8, 22);
    }

    #[test]
    fn tabulated_btree_fills_and_empties_completely() {
        let mut m = BTreeMapper::<TabulatedNode<Spec4x2>>::new(4, 2);
        for fp in 0..16u64 {
            assert!(m.insert(fp, (fp % 4) as u32));
            m.check();
        }
        assert_eq!(m.len(), 16);
        assert!(m.depth() >= 3);
        for fp in (0..16u64).rev() {
            assert_eq!(m.remove(fp), Some((fp % 4) as u32));
            m.check();
        }
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn table_sizes_are_reported() {
        let t = Spec4x2::tables();
        // a handful of hundred thousand entries replace all comparisons and shifts inside the nodes
        assert!(t.table_entries() > 100_000);
        assert!(t.table_entries() < 5_000_000, "{}", t.table_entries());
    }
}
