//! Section 6, local resizing: the slot arrays of a bucket as *resizable arrays* (Lemma 2.3).
//!
//! > "Consider an array A that grows and shrinks over time, supporting both an extension operation and a shrink
//! > operation. If n-bar is the maximum size that the array is allowed to be, and n is the current size, then the
//! > array A can be implemented to use n + O(sqrt(n-bar)) machine words while supporting constant-time operations.
//! > We break A into slabs of size sqrt(n-bar) each. These slabs are dynamically allocated and freed so that there
//! > are only ceil(n / sqrt(n-bar)) at a time. We also keep a smaller array B of size sqrt(n-bar) whose i-th entry
//! > stores a pointer to slab A_i."
//!
//! `Vec<u32>::with_capacity(cap)` (what the fixed-capacity tables of Sections 3-5 use) reserves all `cap` slots of a
//! bucket from the start, so a table that is only half full still holds all its memory. [`SlabVec`] allocates slabs on
//! demand, so a bucket with `k` items holds about `k + sqrt(cap)` words, and the whole table `n + O(n / log^1.5 n)`.

/// The operations the bucket implementations need from the array that holds their slots.
pub trait Slots: Sized {
    /// An empty array that will never hold more than `max` items.
    fn with_max(max: usize) -> Self;
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> u32;
    fn set(&mut self, i: usize, v: u32);
    fn push(&mut self, v: u32);
    /// Remove item `i` by moving the last item into its place; returns the removed item.
    fn swap_remove(&mut self, i: usize) -> u32;
    /// Index of the first item equal to `x` (a linear scan).
    fn position(&self, x: u32) -> Option<usize>;
    /// The items in order (used by rebuilds and checks).
    fn to_vec(&self) -> Vec<u32>;
    /// Slots currently allocated, used or not.
    fn allocated_slots(&self) -> usize;
    /// Bits of bookkeeping beyond the slots themselves (the pointer array of a [`SlabVec`]).
    fn overhead_bits(&self) -> u64 {
        0
    }
}

impl Slots for Vec<u32> {
    fn with_max(max: usize) -> Self {
        Vec::with_capacity(max)
    }
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }
    #[inline]
    fn get(&self, i: usize) -> u32 {
        self[i]
    }
    #[inline]
    fn set(&mut self, i: usize, v: u32) {
        self[i] = v;
    }
    #[inline]
    fn push(&mut self, v: u32) {
        Vec::push(self, v);
    }
    #[inline]
    fn swap_remove(&mut self, i: usize) -> u32 {
        Vec::swap_remove(self, i)
    }
    #[inline]
    fn position(&self, x: u32) -> Option<usize> {
        self.iter().position(|&k| k == x)
    }
    fn to_vec(&self) -> Vec<u32> {
        self.as_slice().to_vec()
    }
    fn allocated_slots(&self) -> usize {
        self.capacity()
    }
}

/// Slab size for an array of at most `max` items: the power of two at or above `sqrt(max)`, so that locating item
/// `i` is a shift and a mask.
fn slab_bits_for(max: usize) -> u32 {
    let s = (max as f64).sqrt().ceil().max(1.0) as usize;
    (usize::BITS - (s - 1).leading_zeros()).max(1)
}

/// The resizable array of Lemma 2.3 for `u32` items.
#[derive(Clone, Debug)]
pub struct SlabVec {
    /// The pointer array `B`: one entry per allocated slab.
    slabs: Vec<Box<[u32]>>,
    slab_bits: u32,
    len: usize,
}

impl SlabVec {
    #[inline]
    fn slab_size(&self) -> usize {
        1usize << self.slab_bits
    }

    /// Number of slabs currently allocated.
    pub fn slab_count(&self) -> usize {
        self.slabs.len()
    }
}

impl Slots for SlabVec {
    fn with_max(max: usize) -> Self {
        SlabVec { slabs: Vec::new(), slab_bits: slab_bits_for(max), len: 0 }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn get(&self, i: usize) -> u32 {
        debug_assert!(i < self.len);
        self.slabs[i >> self.slab_bits][i & (self.slab_size() - 1)]
    }

    #[inline]
    fn set(&mut self, i: usize, v: u32) {
        debug_assert!(i < self.len);
        let mask = self.slab_size() - 1;
        self.slabs[i >> self.slab_bits][i & mask] = v;
    }

    fn push(&mut self, v: u32) {
        if self.len == self.slabs.len() * self.slab_size() {
            // one new slab: "each growth/shrinkage of the array requires at most one slab to be allocated/freed"
            self.slabs.push(vec![0u32; self.slab_size()].into_boxed_slice());
        }
        let i = self.len;
        self.len += 1;
        let mask = self.slab_size() - 1;
        self.slabs[i >> self.slab_bits][i & mask] = v;
    }

    fn swap_remove(&mut self, i: usize) -> u32 {
        assert!(i < self.len);
        let last = self.len - 1;
        let mask = self.slab_size() - 1;
        let removed = self.slabs[i >> self.slab_bits][i & mask];
        let moved = self.slabs[last >> self.slab_bits][last & mask];
        self.slabs[i >> self.slab_bits][i & mask] = moved;
        self.len = last;
        // free the last slab once it is completely unused; keep one slab of slack so that an array that oscillates
        // around a slab boundary does not allocate and free on every operation
        while self.slabs.len() * self.slab_size() >= self.len + 2 * self.slab_size() {
            self.slabs.pop();
        }
        removed
    }

    fn position(&self, x: u32) -> Option<usize> {
        let size = self.slab_size();
        let mut remaining = self.len;
        for (si, slab) in self.slabs.iter().enumerate() {
            let used = remaining.min(size);
            if let Some(p) = slab[..used].iter().position(|&k| k == x) {
                return Some(si * size + p);
            }
            remaining -= used;
            if remaining == 0 {
                break;
            }
        }
        None
    }

    fn to_vec(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.len);
        let size = self.slab_size();
        let mut remaining = self.len;
        for slab in &self.slabs {
            let used = remaining.min(size);
            out.extend_from_slice(&slab[..used]);
            remaining -= used;
            if remaining == 0 {
                break;
            }
        }
        out
    }

    fn allocated_slots(&self) -> usize {
        self.slabs.len() * self.slab_size()
    }

    fn overhead_bits(&self) -> u64 {
        // the pointer array B: one 64-bit pointer per slab
        self.slabs.capacity() as u64 * 64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn run<S: Slots>(max: usize, ops: usize, seed: u64) {
        let mut s = S::with_max(max);
        let mut model: Vec<u32> = Vec::new();
        let mut rng = SplitMix64::new(seed);
        for step in 0..ops {
            match rng.below(5) {
                0 | 1 | 2 if model.len() < max => {
                    let v = rng.next_u32() % 50;
                    s.push(v);
                    model.push(v);
                }
                3 if !model.is_empty() => {
                    let i = rng.below(model.len() as u64) as usize;
                    assert_eq!(s.swap_remove(i), model.swap_remove(i), "step {step}");
                }
                _ if !model.is_empty() => {
                    let i = rng.below(model.len() as u64) as usize;
                    let v = rng.next_u32() % 50;
                    s.set(i, v);
                    model[i] = v;
                }
                _ => {}
            }
            assert_eq!(s.len(), model.len());
            if step % 31 == 0 {
                assert_eq!(s.to_vec(), model, "step {step}");
                for x in 0..50u32 {
                    assert_eq!(s.position(x), model.iter().position(|&k| k == x), "position of {x}, step {step}");
                }
                for (i, &v) in model.iter().enumerate() {
                    assert_eq!(s.get(i), v);
                }
            }
        }
    }

    #[test]
    fn vec_store_matches_the_model() {
        run::<Vec<u32>>(500, 20_000, 1);
    }

    #[test]
    fn slab_store_matches_the_model() {
        run::<SlabVec>(500, 20_000, 2);
        run::<SlabVec>(1, 500, 3);
        run::<SlabVec>(100_000, 30_000, 4);
    }

    #[test]
    fn slab_space_is_len_plus_order_sqrt_max() {
        let max = 1_000_000usize;
        let mut s = SlabVec::with_max(max);
        let slab = 1usize << slab_bits_for(max);
        assert_eq!(slab, 1024); // sqrt(1e6) = 1000 -> 1024
        for i in 0..max {
            s.push(i as u32);
            if i % 50_000 == 0 {
                // never more than one slab (and the slack slab) beyond the items
                assert!(s.allocated_slots() <= s.len() + slab, "len {} allocated {}", s.len(), s.allocated_slots());
            }
        }
        assert!(s.allocated_slots() <= max + slab);
        while s.len() > 10 {
            let i = s.len() / 2;
            s.swap_remove(i);
            assert!(s.allocated_slots() <= s.len() + 2 * slab, "shrinking keeps at most two slabs of slack");
        }
        assert!(s.slab_count() <= 2);
    }

    #[test]
    fn oscillating_around_a_slab_boundary_does_not_thrash() {
        let mut s = SlabVec::with_max(10_000);
        let slab = 1usize << slab_bits_for(10_000);
        for i in 0..slab {
            s.push(i as u32);
        }
        let before = s.slab_count();
        for _ in 0..1_000 {
            s.push(7); // crosses into a second slab
            s.swap_remove(s.len() - 1); // and back
        }
        assert!(s.slab_count() <= before + 1);
    }
}
