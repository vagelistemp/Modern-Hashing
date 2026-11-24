//! Parameters of a partition hash table (Section 3 and 4 of the paper).
//!
//! The paper defines, for capacity `n` (all logarithms base 2, i.e. lengths in bits):
//!   * a bucket has `log^3 n + c log^2 n` slots,
//!   * the bucket array has `n / log^3 n` buckets,
//!   * a fingerprint is a hash into `[log^9 n]`, i.e. `9 log log n` bits (Lemma 4.1).
//!
//! The thesis code used the natural logarithm, which makes the buckets about 3x smaller. Here
//! `Params::paper` uses `log2` as the paper does; [`Params::custom`] lets tests and benchmarks pick
//! any geometry (tiny buckets force overflows and rebuilds, tiny fingerprints force collisions).

/// Largest supported fingerprint width: a mapper entry `(fingerprint << pos_bits) | position` must fit a `u64`.
pub const MAX_FP_BITS: u32 = 48;
/// Largest supported position width (buckets have at most `2^16` slots).
pub const MAX_POS_BITS: u32 = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Params {
    /// Maximum number of keys `n`.
    pub capacity: usize,
    /// Number of buckets `m`.
    pub num_buckets: usize,
    /// Slots per bucket.
    pub bucket_cap: usize,
    /// Initial fingerprint width in bits.
    pub fp_bits: u32,
    /// Bits needed to address a slot in a bucket: `ceil(log2(bucket_cap))`.
    pub pos_bits: u32,
    /// Seed of the pseudo-random generator that draws the hash functions.
    pub seed: u64,
}

/// `ceil(log2(x))` for `x >= 1`.
pub fn ceil_log2(x: usize) -> u32 {
    assert!(x >= 1);
    usize::BITS - (x - 1).leading_zeros()
}

impl Params {
    /// The geometry prescribed by the paper for capacity `n`, with `c = 2` as in the thesis.
    pub fn paper(capacity: usize) -> Params {
        Self::paper_with_c(capacity, 2.0)
    }

    pub fn paper_with_c(capacity: usize, c: f64) -> Params {
        assert!(capacity >= 2, "capacity must be at least 2");
        let lg = (capacity as f64).log2().max(2.0);
        let lg3 = lg * lg * lg;
        let num_buckets = ((capacity as f64 / lg3).floor() as usize).max(1);
        let bucket_cap = (lg3 + c * lg * lg).ceil() as usize;
        let bucket_cap = bucket_cap.min(1 << MAX_POS_BITS);
        // Lemma 4.1: range log^9 n  <=>  9 * log2(log2 n) bits.
        let fp_bits = ((9.0 * lg.log2()).ceil() as u32).clamp(8, MAX_FP_BITS);
        Params { capacity, num_buckets, bucket_cap, fp_bits, pos_bits: ceil_log2(bucket_cap), seed: 0x5EED }
    }

    /// An explicit geometry (used by tests and by experiments on the bucket size).
    pub fn custom(capacity: usize, num_buckets: usize, bucket_cap: usize, fp_bits: u32) -> Params {
        assert!(capacity >= 1 && num_buckets >= 1 && bucket_cap >= 1);
        assert!(bucket_cap <= (1 << MAX_POS_BITS), "at most 2^16 slots per bucket");
        assert!((1..=MAX_FP_BITS).contains(&fp_bits));
        Params { capacity, num_buckets, bucket_cap, fp_bits, pos_bits: ceil_log2(bucket_cap), seed: 0x5EED }
    }

    pub fn with_seed(mut self, seed: u64) -> Params {
        self.seed = seed;
        self
    }

    pub fn with_fp_bits(mut self, fp_bits: u32) -> Params {
        assert!((1..=MAX_FP_BITS).contains(&fp_bits));
        self.fp_bits = fp_bits;
        self
    }

    /// Total number of key slots `m * bucket_cap` (the paper's `n + cn/log n`).
    pub fn total_slots(&self) -> usize {
        self.num_buckets * self.bucket_cap
    }

    /// Mean number of keys per bucket when the table holds `n` keys.
    pub fn mean_load(&self) -> f64 {
        self.capacity as f64 / self.num_buckets as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_values() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(65536), 16);
    }

    #[test]
    fn paper_parameters_for_one_million() {
        let p = Params::paper(1_000_000);
        // log2(1e6) = 19.93 -> log^3 = 7918.2, c*log^2 = 794.5
        assert_eq!(p.num_buckets, 126);
        assert_eq!(p.bucket_cap, 8713);
        assert_eq!(p.fp_bits, 39);
        assert_eq!(p.pos_bits, 14);
        assert!(p.total_slots() >= p.capacity);
        // slack is c n / log n  ~ 10%
        assert!((p.total_slots() as f64 / p.capacity as f64) < 1.12);
    }

    #[test]
    fn paper_parameters_are_valid_for_all_sizes() {
        for exp in 1..=32u32 {
            let n = (1usize << exp).max(2);
            let p = Params::paper(n);
            assert!(p.bucket_cap <= 1 << MAX_POS_BITS, "n = {n}");
            assert!(p.fp_bits <= MAX_FP_BITS);
            assert!(p.fp_bits + p.pos_bits <= 64);
            assert!(p.num_buckets >= 1);
        }
    }
}
