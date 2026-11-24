main.rs is a throwaway program (not part of the crate) that repeats every e2 benchmark configuration
(same geometry Params::paper(n).with_seed(1), same keys, same insertion order, 5 sizes x 2 load factors x 2 key modes x 5 structures
= 96 runs) and prints the rebuild counters. The hash functions come from a deterministic PRNG, so the counts equal those
of the benchmark run. output.txt is its result: 0 whole-table rebuilds, 0 bucket rebuilds, 0 fingerprint widenings.
To rerun: create a crate with  modern_hashing = { path = "../.." }  (the repository root) and this main.rs as src/main.rs.
