//! 2-universal hashing over the Mersenne prime p = 2^61 - 1:  h(x) = (a*x + b) mod p.
//!
//! The paper assumes *fully random* hash functions ("we assume fully random hash functions
//! throughout"). That is the standard simplification of a theory paper; in code we use the classic
//! Carter-Wegman family, which is 2-universal and is what the thesis implementation also used.
//! Unlike the thesis implementation (a*x+b computed on a signed 32-bit key mod 2^31-1, which went
//! negative for negative keys) keys here are `u32` and every value is a valid key.

use crate::rng::SplitMix64;

/// The Mersenne prime 2^61 - 1.
pub const P61: u64 = (1u64 << 61) - 1;

/// Reduce `t < 2^122` modulo `P61` without a division (Mersenne folding).
#[inline]
pub fn reduce61(t: u128) -> u64 {
    let lo = (t as u64) & P61;
    let hi = (t >> 61) as u64;
    let mut r = lo + hi; // < 2^62
    r = (r & P61) + (r >> 61);
    if r >= P61 {
        r - P61
    } else {
        r
    }
}

/// One member `h(x) = (a*x + b) mod P61` of the Carter-Wegman family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniversalHash {
    a: u64,
    b: u64,
}

impl UniversalHash {
    /// Draw a random member (`a` in `[1, P61)`, `b` in `[0, P61)`).
    pub fn random(rng: &mut SplitMix64) -> Self {
        UniversalHash { a: 1 + rng.below(P61 - 1), b: rng.below(P61) }
    }

    /// Build a function from explicit parameters (used by tests).
    pub fn from_parts(a: u64, b: u64) -> Self {
        assert!(a >= 1 && a < P61 && b < P61);
        UniversalHash { a, b }
    }

    /// The raw hash value in `[0, P61)`.
    #[inline]
    pub fn raw(&self, x: u32) -> u64 {
        reduce61(self.a as u128 * x as u128 + self.b as u128)
    }

    /// Map `x` to a bucket in `[0, m)` using multiply-shift range reduction (no modulo bias, no division).
    #[inline]
    pub fn bucket(&self, x: u32, m: usize) -> usize {
        ((self.raw(x) as u128 * m as u128) >> 61) as usize
    }

    /// A `bits`-bit fingerprint of `x` (`1 <= bits <= 48`): the low bits of the raw hash.
    #[inline]
    pub fn fingerprint(&self, x: u32, bits: u32) -> u64 {
        debug_assert!((1..=48).contains(&bits));
        self.raw(x) & ((1u64 << bits) - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce61_matches_naive_modulo() {
        let mut r = SplitMix64::new(42);
        for _ in 0..200_000 {
            let t = ((r.next_u64() as u128) << 29) ^ r.next_u64() as u128; // < 2^93
            assert_eq!(reduce61(t), (t % P61 as u128) as u64);
        }
        for t in [0u128, 1, P61 as u128 - 1, P61 as u128, P61 as u128 + 1, (P61 as u128) * 2, (1u128 << 122) - 1] {
            assert_eq!(reduce61(t), (t % P61 as u128) as u64, "t = {t}");
        }
    }

    #[test]
    fn bucket_is_in_range_for_edge_keys() {
        let mut r = SplitMix64::new(3);
        let h = UniversalHash::random(&mut r);
        for m in [1usize, 2, 3, 127, 1 << 20] {
            for x in [0u32, 1, 2, u32::MAX - 1, u32::MAX, 0x8000_0000] {
                assert!(h.bucket(x, m) < m);
            }
        }
    }

    #[test]
    fn buckets_are_roughly_balanced_for_sequential_keys() {
        let mut r = SplitMix64::new(11);
        let h = UniversalHash::random(&mut r);
        let m = 64;
        let n = 200_000u32;
        let mut cnt = vec![0u32; m];
        for x in 0..n {
            cnt[h.bucket(x, m)] += 1;
        }
        let mean = n as f64 / m as f64;
        let sd = (mean * (1.0 - 1.0 / m as f64)).sqrt();
        for &c in &cnt {
            assert!((c as f64 - mean).abs() < 8.0 * sd, "bucket count {c} too far from mean {mean}");
        }
    }

    #[test]
    fn fingerprint_fits_in_requested_bits() {
        let mut r = SplitMix64::new(5);
        let h = UniversalHash::random(&mut r);
        for bits in [1u32, 4, 16, 39, 48] {
            for x in 0..1000u32 {
                assert!(h.fingerprint(x, bits) < (1u64 << bits));
            }
        }
    }
}
