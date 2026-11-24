//! A real `k`-trie for the buffers of Section 5 (Lemma 2.2).
//!
//! > "For any power-of-two k, a k-trie on n w-bit keys is a k-ary search tree where the pivots are evenly spaced in
//! > the universe of possible keys. For any key x in the tree, the root-to-leaf path for x can be computed by:
//! > using the first (i.e., highest-order) log k bits to navigate the root node, using the next log k bits to
//! > navigate the level-2 node, ... for a total of w / log k levels."
//!
//! Here `k = 256` and `w = 32`, so a key is four bytes and the trie has four levels: lookup, insertion and removal
//! of a key touch exactly four nodes, i.e. O(1). The trie stores *pending operations* keyed by the key they apply to
//! ("put this value" / "delete"), at most one per key (a newer operation on the same key replaces the older one).
//! Nodes are fixed arrays of 256 words, zero meaning "empty": the assumption of Lemma 2.2 that "the arrays used to
//! allocate nodes are already pre-allocated and initialized to zero". A node whose last slot is cleared is returned to
//! a free list (it is all zeros again, so it can be reused without touching it), which keeps the space proportional to
//! the number of pending operations: at most `3 * entries + 1` nodes of 1 KiB each.
//!
//! The thesis simulated these tries with `ArrayList<Operation>`: a list cannot be searched in O(1), which is why its
//! `get` ignored the buffers and could not see operations that had not been applied yet.

use crate::dict::{Key, Val};

/// An operation waiting in a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Put(Val),
    Delete,
}

const FANOUT: usize = 256;
const LEVELS: usize = 4;

#[inline]
fn digit(key: Key, level: usize) -> usize {
    ((key >> (24 - 8 * level)) & 0xFF) as usize
}

#[derive(Debug)]
pub struct KTrie {
    /// Node `0` is the root. Levels 0..=2 hold `node index + 1` per digit, level 3 holds `entry index + 1`.
    nodes: Vec<[u32; FANOUT]>,
    /// Number of non-zero slots of each node.
    counts: Vec<u16>,
    /// Nodes that are all zeros and not part of the trie.
    free: Vec<u32>,
    /// The pending operations, densely packed so that "take any one" is O(1).
    entries: Vec<(Key, Op)>,
}

impl Default for KTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl KTrie {
    pub fn new() -> Self {
        KTrie { nodes: vec![[0; FANOUT]], counts: vec![0], free: Vec::new(), entries: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of nodes in the trie (the root included).
    pub fn node_count(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    /// The pending operation for `key`.
    pub fn get(&self, key: Key) -> Option<Op> {
        let mut n = 0usize;
        for level in 0..LEVELS - 1 {
            let c = self.nodes[n][digit(key, level)];
            if c == 0 {
                return None;
            }
            n = c as usize - 1;
        }
        let e = self.nodes[n][digit(key, LEVELS - 1)];
        if e == 0 {
            None
        } else {
            Some(self.entries[e as usize - 1].1)
        }
    }

    fn new_node(&mut self) -> usize {
        if let Some(i) = self.free.pop() {
            debug_assert!(self.counts[i as usize] == 0);
            i as usize
        } else {
            self.nodes.push([0; FANOUT]);
            self.counts.push(0);
            self.nodes.len() - 1
        }
    }

    /// Set the operation for `key`; returns the operation it replaced, if any.
    pub fn insert(&mut self, key: Key, op: Op) -> Option<Op> {
        let mut n = 0usize;
        for level in 0..LEVELS - 1 {
            let d = digit(key, level);
            let c = self.nodes[n][d];
            n = if c == 0 {
                let m = self.new_node();
                self.nodes[n][d] = m as u32 + 1;
                self.counts[n] += 1;
                m
            } else {
                c as usize - 1
            };
        }
        let d = digit(key, LEVELS - 1);
        let e = self.nodes[n][d];
        if e == 0 {
            self.entries.push((key, op));
            self.nodes[n][d] = self.entries.len() as u32;
            self.counts[n] += 1;
            None
        } else {
            let old = self.entries[e as usize - 1].1;
            self.entries[e as usize - 1].1 = op;
            Some(old)
        }
    }

    /// The node visited at each level on the way to `key`'s leaf slot, if the whole path exists.
    fn path(&self, key: Key) -> Option<[usize; LEVELS]> {
        let mut p = [0usize; LEVELS];
        let mut n = 0usize;
        for level in 0..LEVELS - 1 {
            p[level] = n;
            let c = self.nodes[n][digit(key, level)];
            if c == 0 {
                return None;
            }
            n = c as usize - 1;
        }
        p[LEVELS - 1] = n;
        if self.nodes[n][digit(key, LEVELS - 1)] == 0 {
            None
        } else {
            Some(p)
        }
    }

    /// Clear the leaf slot of `key` (found through `path`) and give back every node that became empty.
    fn unlink(&mut self, key: Key, path: [usize; LEVELS]) {
        self.nodes[path[LEVELS - 1]][digit(key, LEVELS - 1)] = 0;
        self.counts[path[LEVELS - 1]] -= 1;
        for level in (1..LEVELS).rev() {
            let node = path[level];
            if self.counts[node] != 0 {
                break;
            }
            // the node is all zeros again: unhook it from its parent and put it on the free list
            let parent = path[level - 1];
            self.nodes[parent][digit(key, level - 1)] = 0;
            self.counts[parent] -= 1;
            self.free.push(node as u32);
        }
    }

    /// Remove the operation for `key`.
    pub fn remove(&mut self, key: Key) -> Option<Op> {
        let path = self.path(key)?;
        let idx = self.nodes[path[LEVELS - 1]][digit(key, LEVELS - 1)] as usize - 1;
        let removed = self.entries[idx].1;
        let last = self.entries.len() - 1;
        self.unlink(key, path);
        if idx != last {
            // the last entry moves into the hole: repoint its leaf slot
            let moved_key = self.entries[last].0;
            self.entries.swap(idx, last);
            let mp = self.path(moved_key).expect("moved entry must be reachable");
            self.nodes[mp[LEVELS - 1]][digit(moved_key, LEVELS - 1)] = idx as u32 + 1;
        }
        self.entries.pop();
        Some(removed)
    }

    /// Remove and return some pending operation (the most recently stored one).
    pub fn pop_any(&mut self) -> Option<(Key, Op)> {
        let &(key, op) = self.entries.last()?;
        let path = self.path(key).expect("entry must be reachable");
        self.unlink(key, path);
        self.entries.pop();
        Some((key, op))
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        *self = KTrie::new();
    }

    /// Iterate over the pending operations (in arbitrary order).
    pub fn iter(&self) -> impl Iterator<Item = &(Key, Op)> {
        self.entries.iter()
    }

    /// Bits occupied: the allocated 256-word nodes (high-water mark) plus the dense entry list (key, 33-bit operation).
    pub fn space_bits(&self) -> u64 {
        self.nodes.len() as u64 * FANOUT as u64 * 32 + self.entries.capacity() as u64 * (32 + 33)
    }

    pub fn check(&self) {
        let mut referenced = 0usize;
        let mut reachable_nodes = 1usize;
        let mut seen = vec![false; self.nodes.len()];
        seen[0] = true;
        let mut stack = vec![(0usize, 0usize)];
        while let Some((n, level)) = stack.pop() {
            let mut nonzero = 0u16;
            for d in 0..FANOUT {
                let c = self.nodes[n][d];
                if c == 0 {
                    continue;
                }
                nonzero += 1;
                if level < LEVELS - 1 {
                    let child = c as usize - 1;
                    assert!(child < self.nodes.len(), "pointer outside the node pool");
                    assert!(!seen[child], "node reachable twice");
                    seen[child] = true;
                    reachable_nodes += 1;
                    stack.push((child, level + 1));
                } else {
                    assert!((c as usize) <= self.entries.len(), "dangling entry index");
                    referenced += 1;
                }
            }
            assert_eq!(self.counts[n], nonzero, "slot count of node {n} is out of sync");
            if n != 0 {
                assert!(nonzero > 0, "an empty non-root node must have been returned to the free list");
            }
        }
        assert_eq!(referenced, self.entries.len(), "entries and leaf slots disagree");
        assert_eq!(reachable_nodes, self.node_count(), "leaked or double-freed nodes");
        for &f in &self.free {
            assert!(!seen[f as usize], "a free node is still reachable");
            assert_eq!(self.counts[f as usize], 0);
            assert!(self.nodes[f as usize].iter().all(|&x| x == 0), "a free node is not all zeros");
        }
        for (i, &(k, _)) in self.entries.iter().enumerate() {
            let p = self.path(k).expect("entry unreachable");
            assert_eq!(self.nodes[p[LEVELS - 1]][digit(k, LEVELS - 1)] as usize, i + 1, "leaf slot points at the wrong entry");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;
    use std::collections::HashMap;

    #[test]
    fn matches_a_hash_map_under_random_operations() {
        let mut t = KTrie::new();
        let mut model: HashMap<Key, Op> = HashMap::new();
        let mut rng = SplitMix64::new(77);
        for step in 0..60_000 {
            // few distinct high bytes so that nodes are shared, plus the extreme keys
            let key = match rng.below(10) {
                0 => 0,
                1 => u32::MAX,
                _ => ((rng.below(3) as u32) << 24) | (rng.below(4) as u32) << 16 | rng.below(300) as u32,
            };
            match rng.below(5) {
                0 | 1 => {
                    let op = if rng.below(4) == 0 { Op::Delete } else { Op::Put(rng.next_u32()) };
                    assert_eq!(t.insert(key, op), model.insert(key, op), "step {step}");
                }
                2 => assert_eq!(t.remove(key), model.remove(&key), "step {step}"),
                3 => {
                    if let Some((k, op)) = t.pop_any() {
                        assert_eq!(model.remove(&k), Some(op));
                    } else {
                        assert!(model.is_empty());
                    }
                }
                _ => assert_eq!(t.get(key), model.get(&key).copied(), "step {step}"),
            }
            assert_eq!(t.len(), model.len());
            if step % 50 == 0 {
                t.check();
            }
        }
        t.check();
        t.clear();
        assert!(t.is_empty());
        assert_eq!(t.get(0), None);
        t.check();
        t.insert(5, Op::Delete);
        assert_eq!(t.get(5), Some(Op::Delete));
    }

    #[test]
    fn random_keys_do_not_leak_nodes() {
        // fully random 32-bit keys: every entry needs 3 private nodes; emptying the trie must free all of them
        let mut t = KTrie::new();
        let mut rng = SplitMix64::new(5);
        for round in 0..200 {
            let mut keys = Vec::new();
            for _ in 0..40 {
                let k = rng.next_u32();
                t.insert(k, Op::Put(round));
                keys.push(k);
            }
            assert!(t.node_count() <= 1 + 3 * 40);
            t.check();
            while t.pop_any().is_some() {}
            assert_eq!(t.node_count(), 1, "only the root may remain once the trie is empty (round {round})");
            t.check();
        }
        // the pool never grew beyond what one round needed
        assert!(t.space_bits() < (1 + 3 * 40 + 1) as u64 * 256 * 32 + 1_000_000);
    }

    #[test]
    fn four_levels_for_u32_keys() {
        let mut t = KTrie::new();
        t.insert(0x0102_0304, Op::Put(1));
        // root + three more levels = 4 nodes in use
        assert_eq!(t.node_count(), 4);
        t.insert(0x0102_0305, Op::Put(2)); // shares the whole path except the last digit
        assert_eq!(t.node_count(), 4);
        t.insert(0x0202_0304, Op::Put(3)); // new path below the root
        assert_eq!(t.node_count(), 7);
        t.remove(0x0202_0304);
        assert_eq!(t.node_count(), 4, "the private path of a removed key is freed");
    }
}
