# ZP7 (Zach's Peppy Parallel-Prefix-Popcountin' PEXT/PDEP Polyfill)

This is a fast branchless replacement for the PEXT/PDEP instructions.
If you don't know what those are, this probably won't make much sense.

These instructions are very fast on Intel chips that support them
(3 cycles latency), but have a much slower implementation on AMD.
This code will be much slower than the native instructions on Intel
chips, but faster on AMD chips, and generally faster than a naive
loop (for all but the most trivial cases).

A detailed description of the algorithm used is in `zp7.c`.

# Usage
This is distributed as a single C file, `zp7.c`.
These two functions are drop-in replacements for `_pext_u64` and `_pdep_u64`:
```c
uint64_t zp7_pext_64(uint64_t a, uint64_t mask);
uint64_t zp7_pdep_64(uint64_t a, uint64_t mask);
```

There are also variants for precomputed masks, in case the same mask is used
across multiple calls (whether for PEXT or PDEP--the masks are the same for both).
In this case, a `zp7_masks_64_t` struct is created from the input mask using the
`zp7_ppp_64` function, and passed to the `zp7_*_pre_64` variants:
```c
zp7_masks_64_t zp7_ppp_64(uint64_t mask);
uint64_t zp7_pext_pre_64(uint64_t a, const zp7_masks_64_t *masks);
uint64_t zp7_pdep_pre_64(uint64_t a, const zp7_masks_64_t *masks);
```

Two #defines can change the instructions used, depending on the target CPU:
* `HAS_CLMUL`: whether the processor has the
[CLMUL instruction set](https://en.wikipedia.org/wiki/CLMUL_instruction_set), which
is on most x86 CPUs since ~2010.  Using CLMUL gives a fairly significant
speedup and code size reduction.
* `HAS_BZHI`: whether the processor has BZHI, which was introduced in the same [BMI2
instructions](https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets) as
PEXT/PDEP. This is only used once for PDEP, so it matters much less than CLMUL.
* `HAS_POPCNT`: whether the processor has POPCNT, which was introduced in
[SSE4a/SSE4.2](https://en.wikipedia.org/wiki/SSE4). Like BZHI, this is only used
once for PDEP, but matters more for speed, as the software POPCNT is several instructions.

This code is hardcoded to operate on 64 bits. It could easily be adapted
for 32 bits by changing `N_BITS` to 5 and replacing `uint64_t` with `uint32_t`.
This will be slightly faster and will save some memory for pre-calculated
masks.

There are also a couple optimizations that could be made for precomputed
masks for PDEP: the POPCNT/BZHI combination, as well as six shifts, depend only
on the mask, and could be precomputed. I've left this out for now in the interest
of simplicity and allowing precomputed masks to be shared between PEXT and PDEP.
