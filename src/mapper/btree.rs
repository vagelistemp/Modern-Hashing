//! The paper's query mapper (Lemma 4.2): a B-tree whose nodes are so small that every operation *inside* a node
//! costs O(1), so the whole tree costs O(depth) = O(1).
//!
//! The tree skeleton here is the textbook B-tree of Cormen, Leiserson, Rivest and Stein (minimum degree `t`,
//! at most `2t - 1` entries per node, top-down insertion that splits full nodes on the way down, top-down deletion
//! that fills thin nodes on the way down), written once and generic over a [`NodeOps`] implementation. The node
//! implementations are where the Method of Four Russians lives:
//!
//! * [`PackedNode`]: 7 entries `(fingerprint, position)` packed into `u64` words, searched with a branchless
//!   compare that produces a 7-bit mask which is then turned into an index by **table lookup**
//!   (`RANK_TABLE`, `EQ_INDEX_TABLE`: a 128-entry table is an exhaustive pre-computation over all 2^7 masks).
//!   This is the variant that scales to the sizes of the experiments.
//! * `TabulatedNode` (see `tabulated.rs`): the *whole* node is a small integer and **every** operation
//!   (search, insert, delete, split, merge) is a read from a table that was pre-computed over all possible nodes.
//!   This is the construction exactly as the paper describes it, which only works for tiny key and position widths.

use super::{pack, unpack_fp, unpack_pos, QueryMapper};

/// "No child" marker.
pub const NIL: u16 = u16::MAX;
/// Largest fan-out supported by the fixed-size child arrays (`K + 1` for `K = 7`).
pub const MAX_CHILDREN: usize = 8;

/// Operations on one B-tree node. A node holds up to `K = 2t - 1` entries `(fingerprint, position)` in increasing
/// fingerprint order. All operations must take O(1) time (K is a constant).
pub trait NodeOps: Clone {
    /// Maximum number of entries, `2t - 1`.
    const K: usize;

    fn empty(fp_bits: u32, pos_bits: u32) -> Self;

    fn len(&self) -> usize;

    /// `Ok(i)` if entry `i` has fingerprint `fp`; otherwise `Err(i)` where `i` is the number of entries with a
    /// smaller fingerprint (the insertion rank, and the index of the child to descend into).
    fn search(&self, fp: u64) -> Result<usize, usize>;

    fn key(&self, i: usize) -> u64;
    fn val(&self, i: usize) -> u32;
    fn set_val(&mut self, i: usize, pos: u32);

    /// Insert at rank `i` (the node must not be full and must not contain `fp`).
    fn insert_at(&mut self, i: usize, fp: u64, pos: u32);

    fn remove_at(&mut self, i: usize) -> (u64, u32);

    /// Split a *full* node: `self` keeps the `t - 1` smallest entries, the median is returned together with a new
    /// node holding the `t - 1` largest entries.
    fn split(&mut self) -> ((u64, u32), Self);

    /// Inverse of [`NodeOps::split`]: `self` (t - 1 entries) ++ `median` ++ `right` (t - 1 entries).
    fn merge(&mut self, median: (u64, u32), right: &Self);

    fn max_fp_bits() -> u32 {
        crate::params::MAX_FP_BITS
    }

    fn max_pos_bits() -> u32 {
        crate::params::MAX_POS_BITS
    }

    fn name() -> &'static str;
}

#[derive(Clone)]
struct TreeNode<N> {
    node: N,
    /// `children[i]` is the child to the left of entry `i`; `children[len]` is the rightmost child.
    children: [u16; MAX_CHILDREN],
    leaf: bool,
}

/// B-tree mapping fingerprints to positions, generic over the node representation.
#[derive(Clone)]
pub struct BTreeMapper<N: NodeOps> {
    nodes: Vec<TreeNode<N>>,
    free: Vec<u16>,
    root: u16,
    len: usize,
    fp_bits: u32,
    pos_bits: u32,
}

impl<N: NodeOps> BTreeMapper<N> {
    /// Minimum degree `t`.
    #[inline]
    fn t() -> usize {
        (N::K + 1) / 2
    }

    fn alloc(&mut self, leaf: bool) -> u16 {
        let blank = TreeNode { node: N::empty(self.fp_bits, self.pos_bits), children: [NIL; MAX_CHILDREN], leaf };
        if let Some(i) = self.free.pop() {
            self.nodes[i as usize] = blank;
            i
        } else {
            assert!(self.nodes.len() < NIL as usize, "query mapper has too many nodes for 16-bit pointers");
            self.nodes.push(blank);
            (self.nodes.len() - 1) as u16
        }
    }

    fn release(&mut self, i: u16) {
        self.free.push(i);
    }

    /// Number of nodes on the longest root-to-leaf path (0 for an empty tree).
    pub fn depth(&self) -> usize {
        let mut d = 0;
        let mut x = self.root;
        while x != NIL {
            d += 1;
            let n = &self.nodes[x as usize];
            x = if n.leaf { NIL } else { n.children[0] };
        }
        d
    }

    /// Number of live nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    fn split_child(&mut self, p: u16, i: usize) {
        let t = Self::t();
        let y = self.nodes[p as usize].children[i];
        let y_leaf = self.nodes[y as usize].leaf;
        let (median, right_node) = self.nodes[y as usize].node.split();
        let z = self.alloc(y_leaf);
        self.nodes[z as usize].node = right_node;
        if !y_leaf {
            for j in 0..t {
                self.nodes[z as usize].children[j] = self.nodes[y as usize].children[t + j];
                self.nodes[y as usize].children[t + j] = NIL;
            }
        }
        let plen = self.nodes[p as usize].node.len();
        for j in ((i + 1)..=plen).rev() {
            self.nodes[p as usize].children[j + 1] = self.nodes[p as usize].children[j];
        }
        self.nodes[p as usize].children[i + 1] = z;
        self.nodes[p as usize].node.insert_at(i, median.0, median.1);
    }

    fn insert_nonfull(&mut self, mut x: u16, fp: u64, pos: u32) {
        loop {
            let i = match self.nodes[x as usize].node.search(fp) {
                Err(i) => i,
                Ok(_) => unreachable!("insert_nonfull: fingerprint already present"),
            };
            if self.nodes[x as usize].leaf {
                self.nodes[x as usize].node.insert_at(i, fp, pos);
                return;
            }
            let mut i = i;
            let c = self.nodes[x as usize].children[i];
            if self.nodes[c as usize].node.len() == N::K {
                self.split_child(x, i);
                if fp > self.nodes[x as usize].node.key(i) {
                    i += 1;
                }
            }
            x = self.nodes[x as usize].children[i];
        }
    }

    fn min_entry(&self, mut x: u16) -> (u64, u32) {
        loop {
            let n = &self.nodes[x as usize];
            if n.leaf {
                return (n.node.key(0), n.node.val(0));
            }
            x = n.children[0];
        }
    }

    fn max_entry(&self, mut x: u16) -> (u64, u32) {
        loop {
            let n = &self.nodes[x as usize];
            let l = n.node.len();
            if n.leaf {
                return (n.node.key(l - 1), n.node.val(l - 1));
            }
            x = n.children[l];
        }
    }

    /// Merge child `j` and child `j + 1` of `x` around key `j` of `x`.
    fn merge_children(&mut self, x: u16, j: usize) {
        let t = Self::t();
        let l = self.nodes[x as usize].children[j];
        let r = self.nodes[x as usize].children[j + 1];
        let median = self.nodes[x as usize].node.remove_at(j);
        let right_node = self.nodes[r as usize].node.clone();
        self.nodes[l as usize].node.merge(median, &right_node);
        if !self.nodes[l as usize].leaf {
            for q in 0..t {
                self.nodes[l as usize].children[t + q] = self.nodes[r as usize].children[q];
            }
        }
        let xl_after = self.nodes[x as usize].node.len();
        for q in (j + 1)..=xl_after {
            self.nodes[x as usize].children[q] = self.nodes[x as usize].children[q + 1];
        }
        self.nodes[x as usize].children[xl_after + 1] = NIL;
        self.release(r);
    }

    /// Child `i` of `x` has only `t - 1` entries: borrow one from the left sibling.
    fn borrow_from_prev(&mut self, x: u16, i: usize) {
        let c = self.nodes[x as usize].children[i];
        let s = self.nodes[x as usize].children[i - 1];
        let s_len = self.nodes[s as usize].node.len();
        let c_len = self.nodes[c as usize].node.len();
        let (ks, vs) = self.nodes[s as usize].node.remove_at(s_len - 1);
        let (kx, vx) = self.nodes[x as usize].node.remove_at(i - 1);
        self.nodes[x as usize].node.insert_at(i - 1, ks, vs);
        self.nodes[c as usize].node.insert_at(0, kx, vx);
        if !self.nodes[c as usize].leaf {
            for j in (0..=c_len).rev() {
                self.nodes[c as usize].children[j + 1] = self.nodes[c as usize].children[j];
            }
            self.nodes[c as usize].children[0] = self.nodes[s as usize].children[s_len];
            self.nodes[s as usize].children[s_len] = NIL;
        }
    }

    /// Child `i` of `x` has only `t - 1` entries: borrow one from the right sibling.
    fn borrow_from_next(&mut self, x: u16, i: usize) {
        let c = self.nodes[x as usize].children[i];
        let s = self.nodes[x as usize].children[i + 1];
        let s_len = self.nodes[s as usize].node.len();
        let c_len = self.nodes[c as usize].node.len();
        let (ks, vs) = self.nodes[s as usize].node.remove_at(0);
        let (kx, vx) = self.nodes[x as usize].node.remove_at(i);
        self.nodes[x as usize].node.insert_at(i, ks, vs);
        self.nodes[c as usize].node.insert_at(c_len, kx, vx);
        if !self.nodes[c as usize].leaf {
            self.nodes[c as usize].children[c_len + 1] = self.nodes[s as usize].children[0];
            for j in 0..s_len {
                self.nodes[s as usize].children[j] = self.nodes[s as usize].children[j + 1];
            }
            self.nodes[s as usize].children[s_len] = NIL;
        }
    }

    /// Make sure child `i` of `x` has at least `t` entries; returns the (possibly changed) index of that child.
    fn fill(&mut self, x: u16, i: usize) -> usize {
        let t = Self::t();
        let xl = self.nodes[x as usize].node.len();
        if i > 0 {
            let left = self.nodes[x as usize].children[i - 1];
            if self.nodes[left as usize].node.len() >= t {
                self.borrow_from_prev(x, i);
                return i;
            }
        }
        if i < xl {
            let right = self.nodes[x as usize].children[i + 1];
            if self.nodes[right as usize].node.len() >= t {
                self.borrow_from_next(x, i);
                return i;
            }
            self.merge_children(x, i);
            i
        } else {
            self.merge_children(x, i - 1);
            i - 1
        }
    }

    fn remove_rec(&mut self, x: u16, fp: u64) -> Option<u32> {
        let t = Self::t();
        match self.nodes[x as usize].node.search(fp) {
            Ok(i) => {
                if self.nodes[x as usize].leaf {
                    let (_, v) = self.nodes[x as usize].node.remove_at(i);
                    return Some(v);
                }
                let old = self.nodes[x as usize].node.val(i);
                let y = self.nodes[x as usize].children[i];
                let z = self.nodes[x as usize].children[i + 1];
                if self.nodes[y as usize].node.len() >= t {
                    let (pk, pv) = self.max_entry(y);
                    self.nodes[x as usize].node.remove_at(i);
                    self.nodes[x as usize].node.insert_at(i, pk, pv);
                    self.remove_rec(y, pk);
                } else if self.nodes[z as usize].node.len() >= t {
                    let (sk, sv) = self.min_entry(z);
                    self.nodes[x as usize].node.remove_at(i);
                    self.nodes[x as usize].node.insert_at(i, sk, sv);
                    self.remove_rec(z, sk);
                } else {
                    self.merge_children(x, i);
                    let y2 = self.nodes[x as usize].children[i];
                    self.remove_rec(y2, fp);
                }
                Some(old)
            }
            Err(mut i) => {
                if self.nodes[x as usize].leaf {
                    return None;
                }
                let c = self.nodes[x as usize].children[i];
                if self.nodes[c as usize].node.len() < t {
                    i = self.fill(x, i);
                }
                let c = self.nodes[x as usize].children[i];
                self.remove_rec(c, fp)
            }
        }
    }

    fn locate(&self, fp: u64) -> Option<(u16, usize)> {
        let mut x = self.root;
        while x != NIL {
            let n = &self.nodes[x as usize];
            match n.node.search(fp) {
                Ok(i) => return Some((x, i)),
                Err(i) => {
                    if n.leaf {
                        return None;
                    }
                    x = n.children[i];
                }
            }
        }
        None
    }
}

impl<N: NodeOps> QueryMapper for BTreeMapper<N> {
    fn new(fp_bits: u32, pos_bits: u32) -> Self {
        assert!(fp_bits <= N::max_fp_bits(), "fingerprint too wide for this node type");
        assert!(pos_bits <= N::max_pos_bits(), "position too wide for this node type");
        BTreeMapper { nodes: Vec::new(), free: Vec::new(), root: NIL, len: 0, fp_bits, pos_bits }
    }

    fn max_fp_bits() -> u32 {
        N::max_fp_bits()
    }

    fn max_pos_bits() -> u32 {
        N::max_pos_bits()
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.free.clear();
        self.root = NIL;
        self.len = 0;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, fp: u64) -> Option<u32> {
        self.locate(fp).map(|(x, i)| self.nodes[x as usize].node.val(i))
    }

    fn insert(&mut self, fp: u64, pos: u32) -> bool {
        if self.locate(fp).is_some() {
            return false;
        }
        if self.root == NIL {
            let r = self.alloc(true);
            self.nodes[r as usize].node.insert_at(0, fp, pos);
            self.root = r;
            self.len = 1;
            return true;
        }
        if self.nodes[self.root as usize].node.len() == N::K {
            let old = self.root;
            let new_root = self.alloc(false);
            self.nodes[new_root as usize].children[0] = old;
            self.root = new_root;
            self.split_child(new_root, 0);
        }
        self.insert_nonfull(self.root, fp, pos);
        self.len += 1;
        true
    }

    fn set(&mut self, fp: u64, pos: u32) {
        let (x, i) = self.locate(fp).expect("set: fingerprint not present");
        self.nodes[x as usize].node.set_val(i, pos);
    }

    fn remove(&mut self, fp: u64) -> Option<u32> {
        if self.root == NIL {
            return None;
        }
        let r = self.remove_rec(self.root, fp);
        if r.is_some() {
            self.len -= 1;
        }
        // The top-down deletion may merge the root's only two children even when the key turns out to be absent,
        // so the root has to be re-examined whatever the outcome.
        let root = self.root;
        if self.nodes[root as usize].node.len() == 0 {
            if self.nodes[root as usize].leaf {
                self.release(root);
                self.root = NIL;
            } else {
                self.root = self.nodes[root as usize].children[0];
                self.release(root);
            }
        }
        r
    }

    fn space_bits(&self) -> u64 {
        // Count the live nodes by walking the tree: entries are (fp_bits + pos_bits) bits each, an internal node
        // additionally stores (len + 1) 16-bit child pointers, every node a 4-bit entry count.
        let entry_bits = (self.fp_bits + self.pos_bits) as u64;
        let mut bits = 0u64;
        let mut stack = Vec::new();
        if self.root != NIL {
            stack.push(self.root);
        }
        while let Some(x) = stack.pop() {
            let n = &self.nodes[x as usize];
            let l = n.node.len();
            bits += l as u64 * entry_bits + 4;
            if !n.leaf {
                bits += (l as u64 + 1) * 16;
                for c in 0..=l {
                    stack.push(n.children[c]);
                }
            }
        }
        bits
    }

    fn check(&self) {
        let t = Self::t();
        if self.root == NIL {
            assert_eq!(self.len, 0);
            assert_eq!(self.node_count(), 0, "all nodes must be free in an empty tree");
            return;
        }
        // in-order traversal collecting keys; all leaves at the same depth; occupancy bounds
        let mut keys: Vec<u64> = Vec::with_capacity(self.len);
        let mut leaf_depth: Option<usize> = None;
        let mut reachable = 0usize;
        fn walk<N: NodeOps>(
            m: &BTreeMapper<N>,
            x: u16,
            depth: usize,
            t: usize,
            is_root: bool,
            keys: &mut Vec<u64>,
            leaf_depth: &mut Option<usize>,
            reachable: &mut usize,
        ) {
            *reachable += 1;
            let n = &m.nodes[x as usize];
            let l = n.node.len();
            assert!(l <= N::K, "node overflow");
            if is_root {
                assert!(l >= 1, "empty root");
            } else {
                assert!(l >= t - 1, "node underflow: {} < {}", l, t - 1);
            }
            for i in 0..l {
                if i > 0 {
                    assert!(n.node.key(i - 1) < n.node.key(i), "node keys not strictly increasing");
                }
            }
            if n.leaf {
                match leaf_depth {
                    None => *leaf_depth = Some(depth),
                    Some(d) => assert_eq!(*d, depth, "leaves at different depths"),
                }
                for i in 0..l {
                    keys.push(n.node.key(i));
                }
                for c in &n.children {
                    assert_eq!(*c, NIL, "leaf with a child pointer");
                }
            } else {
                for i in 0..=l {
                    assert_ne!(n.children[i], NIL, "internal node missing child {i}");
                    walk(m, n.children[i], depth + 1, t, false, keys, leaf_depth, reachable);
                    if i < l {
                        keys.push(n.node.key(i));
                    }
                }
                for i in (l + 1)..MAX_CHILDREN {
                    assert_eq!(n.children[i], NIL, "stale child pointer beyond the last child");
                }
            }
        }
        walk(self, self.root, 0, t, true, &mut keys, &mut leaf_depth, &mut reachable);
        assert_eq!(keys.len(), self.len, "entry count mismatch");
        for w in keys.windows(2) {
            assert!(w[0] < w[1], "in-order traversal not strictly increasing");
        }
        assert_eq!(reachable, self.node_count(), "unreachable or double-freed nodes");
    }

    fn name() -> &'static str {
        N::name()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// PackedNode: 7 entries in u64 words, searched by a branchless compare + table lookup.
// ---------------------------------------------------------------------------------------------------------------

const PK: usize = 7;
const SENTINEL: u64 = u64::MAX;

/// `RANK_TABLE[mask]`: how many of the compared lanes are smaller than the search key.
/// A pre-computation over all 2^7 = 128 compare masks (the Method of Four Russians in its smallest form).
pub const RANK_TABLE: [u8; 128] = build_rank_table();
/// `EQ_INDEX_TABLE[mask]`: the lane that compared equal (lowest set bit; 7 if none).
pub const EQ_INDEX_TABLE: [u8; 128] = build_eq_index_table();

const fn build_rank_table() -> [u8; 128] {
    let mut t = [0u8; 128];
    let mut m = 0;
    while m < 128 {
        let mut c = 0u8;
        let mut b = 0;
        while b < 7 {
            if (m >> b) & 1 == 1 {
                c += 1;
            }
            b += 1;
        }
        t[m] = c;
        m += 1;
    }
    t
}

const fn build_eq_index_table() -> [u8; 128] {
    let mut t = [7u8; 128];
    let mut m = 0;
    while m < 128 {
        let mut b = 0;
        while b < 7 {
            if (m >> b) & 1 == 1 {
                t[m] = b as u8;
                break;
            }
            b += 1;
        }
        m += 1;
    }
    t
}

/// A B-tree node with `K = 7` entries kept sorted in `[u64; 7]` (entry = `(fingerprint << pos_bits) | position`).
/// Unused lanes hold `u64::MAX`, whose fingerprint field exceeds every legal fingerprint, so a search needs no
/// special case for partially filled nodes. 58 bytes of payload: it fits in one cache line.
#[derive(Clone, Debug)]
pub struct PackedNode {
    entries: [u64; PK],
    n: u8,
    pos_bits: u8,
}

impl PackedNode {
    /// The two 7-bit compare masks of `fp` against the lanes: `lt` (lane key < fp) and `eq` (lane key == fp).
    #[inline]
    pub fn compare_masks(&self, fp: u64) -> (usize, usize) {
        let pb = self.pos_bits as u32;
        let mut lt = 0usize;
        let mut eq = 0usize;
        for i in 0..PK {
            let k = self.entries[i] >> pb;
            lt |= ((k < fp) as usize) << i;
            eq |= ((k == fp) as usize) << i;
        }
        (lt, eq)
    }
}

impl NodeOps for PackedNode {
    const K: usize = PK;

    fn empty(_fp_bits: u32, pos_bits: u32) -> Self {
        PackedNode { entries: [SENTINEL; PK], n: 0, pos_bits: pos_bits as u8 }
    }

    #[inline]
    fn len(&self) -> usize {
        self.n as usize
    }

    #[inline]
    fn search(&self, fp: u64) -> Result<usize, usize> {
        let (lt, eq) = self.compare_masks(fp);
        if eq != 0 {
            Ok(EQ_INDEX_TABLE[eq] as usize)
        } else {
            Err(RANK_TABLE[lt] as usize)
        }
    }

    #[inline]
    fn key(&self, i: usize) -> u64 {
        unpack_fp(self.entries[i], self.pos_bits as u32)
    }

    #[inline]
    fn val(&self, i: usize) -> u32 {
        unpack_pos(self.entries[i], self.pos_bits as u32)
    }

    fn set_val(&mut self, i: usize, pos: u32) {
        let fp = self.key(i);
        self.entries[i] = pack(fp, pos, self.pos_bits as u32);
    }

    fn insert_at(&mut self, i: usize, fp: u64, pos: u32) {
        let n = self.n as usize;
        assert!(n < PK && i <= n);
        self.entries.copy_within(i..n, i + 1);
        self.entries[i] = pack(fp, pos, self.pos_bits as u32);
        self.n += 1;
    }

    fn remove_at(&mut self, i: usize) -> (u64, u32) {
        let n = self.n as usize;
        assert!(i < n);
        let e = self.entries[i];
        self.entries.copy_within(i + 1..n, i);
        self.entries[n - 1] = SENTINEL;
        self.n -= 1;
        (unpack_fp(e, self.pos_bits as u32), unpack_pos(e, self.pos_bits as u32))
    }

    fn split(&mut self) -> ((u64, u32), Self) {
        assert_eq!(self.n as usize, PK, "split needs a full node");
        let t = (PK + 1) / 2; // 4
        let pb = self.pos_bits as u32;
        let med = self.entries[t - 1];
        let mut right = PackedNode::empty(0, pb);
        for j in 0..(t - 1) {
            right.entries[j] = self.entries[t + j];
        }
        right.n = (t - 1) as u8;
        for j in (t - 1)..PK {
            self.entries[j] = SENTINEL;
        }
        self.n = (t - 1) as u8;
        ((unpack_fp(med, pb), unpack_pos(med, pb)), right)
    }

    fn merge(&mut self, median: (u64, u32), right: &Self) {
        let t = (PK + 1) / 2;
        assert_eq!(self.n as usize, t - 1);
        assert_eq!(right.n as usize, t - 1);
        self.entries[t - 1] = pack(median.0, median.1, self.pos_bits as u32);
        for j in 0..(t - 1) {
            self.entries[t + j] = right.entries[j];
        }
        self.n = PK as u8;
    }

    fn name() -> &'static str {
        "btree-packed7"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapper::conformance;
    use crate::rng::SplitMix64;

    #[test]
    fn tables_agree_with_hardware_instructions() {
        // The lookup tables are a pre-computation over every possible mask; check them exhaustively
        // against popcount / count-trailing-zeros, which is what a modern CPU does in one instruction.
        for m in 0..128usize {
            assert_eq!(RANK_TABLE[m] as u32, (m as u32).count_ones(), "rank table, mask {m:07b}");
            let expect = if m == 0 { 7 } else { (m as u32).trailing_zeros() };
            assert_eq!(EQ_INDEX_TABLE[m] as u32, expect, "eq-index table, mask {m:07b}");
        }
    }

    #[test]
    fn packed_node_search_matches_binary_search() {
        let mut rng = SplitMix64::new(9);
        for trial in 0..2000 {
            let n = rng.below(8) as usize;
            let mut keys: Vec<u64> = Vec::new();
            while keys.len() < n {
                let k = rng.below(40);
                if !keys.contains(&k) {
                    keys.push(k);
                }
            }
            keys.sort();
            let mut node = PackedNode::empty(6, 5);
            for (i, k) in keys.iter().enumerate() {
                node.insert_at(i, *k, (i as u32) & 31);
            }
            for probe in 0..45u64 {
                let expect = keys.binary_search(&probe);
                assert_eq!(node.search(probe), expect, "trial {trial}, keys {keys:?}, probe {probe}");
            }
        }
    }

    #[test]
    fn packed_node_split_merge_roundtrip() {
        let mut node = PackedNode::empty(8, 4);
        for i in 0..7 {
            node.insert_at(i, (i as u64) * 3 + 1, i as u32);
        }
        let before: Vec<(u64, u32)> = (0..7).map(|i| (node.key(i), node.val(i))).collect();
        let (med, right) = node.split();
        assert_eq!(node.len(), 3);
        assert_eq!(right.len(), 3);
        assert_eq!(med, before[3]);
        let mut merged = node.clone();
        merged.merge(med, &right);
        let after: Vec<(u64, u32)> = (0..7).map(|i| (merged.key(i), merged.val(i))).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn packed_btree_conforms_small() {
        conformance::run::<BTreeMapper<PackedNode>>(8, 5, 30_000, 31, 11);
    }

    #[test]
    fn packed_btree_conforms_deep() {
        conformance::run::<BTreeMapper<PackedNode>>(39, 14, 60_000, 2_500, 12);
    }

    #[test]
    fn packed_btree_sequential_and_reverse_inserts_then_deletes() {
        for &reverse in &[false, true] {
            let mut m = BTreeMapper::<PackedNode>::new(30, 13);
            let n = 5000u64;
            for i in 0..n {
                let k = if reverse { n - 1 - i } else { i };
                assert!(m.insert(k, (k % 8000) as u32));
                if i % 97 == 0 {
                    m.check();
                }
            }
            m.check();
            assert_eq!(m.len(), n as usize);
            assert!(m.depth() >= 4, "5000 entries with fan-out 8 need several levels");
            for i in 0..n {
                let k = if reverse { i } else { n - 1 - i };
                assert_eq!(m.remove(k), Some((k % 8000) as u32));
                if i % 97 == 0 {
                    m.check();
                }
            }
            m.check();
            assert_eq!(m.len(), 0);
            assert_eq!(m.depth(), 0);
        }
    }
}
