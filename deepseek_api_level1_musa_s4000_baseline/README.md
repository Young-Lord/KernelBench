# DeepSeek API — KernelBench Level 1 MUSA baseline

- Hardware: MTT S4000 single GPU (`mp_22`)
- Backend / precision: MUSA / FP32
- Correctness trials: 5; performance trials: 100
- Timing: GPU events with L2 cache thrashing (milliseconds)
- Provenance: user-confirmed DeepSeek API outputs committed as `level1_musa`
- Original API model/version and decoding parameters: unavailable in repository

## Summary

- Evaluated: 100 / 100
- Compiled: 100 (100.0%)
- Correct / pass@1: 99 (99.0%)
- Correct with valid performance timing: 99 (99.0% of all problems; 100.0% of correct kernels)
- Geometric-mean speedup over eager (correct only): 0.12619373188126815
- Median speedup over eager (correct only): 0.16934931506849316
- Faster than eager: 7.0% of all problems

## Exceptions and interpretation

- Problem 72 is reported as incorrect because the MUSA PyTorch grouped ConvTranspose3d reference is known to be faulty; repository documentation reports the generated kernel as bit-exact against CPU.

- No generated kernel source was modified while producing this baseline.
