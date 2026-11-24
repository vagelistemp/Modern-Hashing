//! A tiny deterministic PRNG (SplitMix64) so that the crate needs no external dependency and
//! every test and benchmark is reproducible from a seed.

/// SplitMix64 (Steele, Lea, Flood). Passes BigCrush when used as a stand-alone generator;
/// more than enough for hash-function parameters and test workloads.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform integer in `0..bound` (Lemire's multiply-shift, bias is below 2^-32 for our bounds).
    #[inline]
    pub fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        ((self.next_u64() as u128 * bound as u128) >> 64) as u64
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A bijection on `u32`: `i -> (i * odd) mod 2^32` followed by a xor-shift. Feeding it `0, 1, 2, ...`
/// yields distinct, well-spread "random looking" keys without needing a set to deduplicate.
#[inline]
pub fn scramble32(i: u32) -> u32 {
    let mut x = i.wrapping_mul(0x9E37_79B1);
    x ^= x >> 15;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_deterministic() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn below_stays_in_range() {
        let mut r = SplitMix64::new(1);
        for bound in [1u64, 2, 3, 10, 1000, u32::MAX as u64] {
            for _ in 0..1000 {
                assert!(r.below(bound) < bound);
            }
        }
    }

    #[test]
    fn scramble32_is_injective_on_a_prefix() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..200_000u32 {
            assert!(seen.insert(scramble32(i)), "collision at {i}");
        }
    }
}
