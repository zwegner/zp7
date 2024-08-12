// ZP7 (Zach's Peppy Parallel-Prefix-Popcountin' PEXT/PDEP Polyfill)
//
// Copyright (c) 2020 Zach Wegner
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdint.h>

#if defined(HAS_CLMUL) || defined(HAS_BZHI) || defined(HAS_POPCNT)
#   include <immintrin.h>
#endif

// ZP7: branchless PEXT/PDEP replacement code for non-Intel processors
//
// The PEXT/PDEP instructions are pretty cool, with various (usually arcane)
// uses, behaving like bitwise gather/scatter instructions. They were introduced
// by Intel with the BMI2 instructions on Haswell.
//
// AMD processors implement these instructions, but very slowly. PEXT/PDEP can
// take from 18 to ~300 cycles, depending on the input mask. See this table:
// https://mobile.twitter.com/InstLatX64/status/1209095219087585281
// Other processors don't have PEXT/PDEP at all. This code is a polyfill for
// these processors. It's much slower than the raw instructions on Intel chips
// (which are 3L1T), but should be faster than AMD's implementation.
//
// Description of the algorithm
// ====
//
// This code uses a "parallel prefix popcount" technique (hereafter PPP for
// brevity). What this means is that we determine, for every bit in the input
// mask, how many bits below it are set. Or rather, aren't set--we need to get
// a count of how many bits each input bit should be shifted to get to its final
// position, that is, the difference between the bit-index of its destination
// and its original bit-index. This is the same as the count of unset bits in
// the mask below each input bit.
//
// The dumb way to get this PPP would be to create a 64-element array in a loop,
// but we want to do this in a bit-parallel fashion. So we store the counts
// "vertically" across six 64-bit values: one 64-bit value holds bit 0 of each
// of the 64 counts, another holds bit 1, etc. We can compute these counts
// fairly easily using a parallel prefix XOR (XOR is equivalent to a 1-bit
// adder that wraps around and ignores carrying). Using parallel prefix XOR as
// a 1-bit adder, we can build an n-bit adder by shifting the result left by
// one and ANDing with the input bits: this computes the carry by seeing where
// an input bit causes the 1-bit sum to overflow from 1 to 0. The shift left
// is needed anyways, because we want the PPP values to represent population
// counts *below* each bit, not including the bit itself.
//
// For processors with the CLMUL instructions (most x86 CPUs since ~2010), we
// can do the parallel prefix XOR and left shift in one instruction, by
// doing a carry-less multiply by -2. This is enabled with the HAS_CLMUL define.
//
// Anyways, once we have these six 64-bit values of the PPP, we can use each
// PPP bit to shift input bits by a power of two. That is, input bits that are
// in the bit-0 PPP mask are shifted by 2**0==1, bits in the bit-1 mask get
// shifted by 2, and so on, for shifts by 4, 8, 16, and 32 bits. Out of these
// six shifts, any shift value between 0 and 63 can be composed.
//
// For PEXT, we have to perform each shift in increasing order (1, 2, ...32) so
// that input bits don't overlap in the intermediate results. The input bits
// need to be pre-masked so that only the relevant bits are being shifted
// around. This is a simple AND (input &= mask) operation.
//
// For PDEP, we perform the shifts in decreasing order (32, 16, ...1). Before
// each shift, we clear the bits to make room for the shifted bits. After
// performing the shifts, we apply the mask to clear any bits not in the mask.
//

#define N_BITS      (6)

typedef struct {
    uint64_t mask;
    uint64_t ppp_bit[N_BITS];
} zp7_masks_64_t;

#ifndef HAS_CLMUL
// If we don't have access to the CLMUL instruction, emulate it with
// shifts and XORs
static inline uint64_t prefix_sum(uint64_t x) {
    for (int i = 0; i < N_BITS; i++)
        x ^= x << (1 << i);
    return x;
}
#endif

// Parallel-prefix-popcount. This is used by both the PEXT/PDEP polyfills.
// It can also be called separately and cached, if the mask values will be used
// more than once (these can be shared across PEXT and PDEP calls if they use
// the same masks). 
zp7_masks_64_t zp7_ppp_64(uint64_t mask) {
    zp7_masks_64_t r;
    r.mask = mask;

    // Count *unset* bits
    mask = ~mask;

#ifdef HAS_CLMUL
    // Move the mask and -2 to XMM registers for CLMUL
    __m128i m = _mm_cvtsi64_si128(mask);
    __m128i neg_2 = _mm_cvtsi64_si128(-2LL);
    for (int i = 0; i < N_BITS - 1; i++) {
        // Do a 1-bit parallel prefix popcount, shifted left by 1,
        // in one carry-less multiply by -2.
        __m128i bit = _mm_clmulepi64_si128(m, neg_2, 0);
        r.ppp_bit[i] = _mm_cvtsi128_si64(bit);

        // Get the carry bit of the 1-bit parallel prefix popcount. On
        // the next iteration, we will sum this bit to get the next mask
        m = _mm_and_si128(m, bit);
    }
    // For the last iteration, we can use a regular multiply by -2 instead of a
    // carry-less one (or rather, a strength reduction of that, with
    // neg/add/etc), since there can't be any carries anyways. That is because
    // the last value of m (which has one bit set for every 32nd unset mask bit)
    // has at most two bits set in it, when mask is zero and thus there are 64
    // bits set in ~mask. If two bits are set, one of them is the top bit, which
    // gets shifted out, since we're counting bits below each mask bit.
    r.ppp_bit[N_BITS - 1] = -_mm_cvtsi128_si64(m) << 1;
#else
    for (int i = 0; i < N_BITS - 1; i++) {
        // Do a 1-bit parallel prefix popcount, shifted left by 1
        uint64_t bit = prefix_sum(mask << 1);
        r.ppp_bit[i] = bit;

        // Get the carry bit of the 1-bit parallel prefix popcount. On
        // the next iteration, we will sum this bit to get the next mask
        mask &= bit;
    }
    // The last iteration won't carry, so just use neg/shift. See the CLMUL
    // case above for justification.
    r.ppp_bit[N_BITS - 1] = -mask << 1;
#endif

    return r;
}

// PEXT

uint64_t zp7_pext_pre_64(uint64_t a, const zp7_masks_64_t *masks) {
    // Mask only the bits that are set in the input mask. Otherwise they collide
    // with input bits and screw everything up
    a &= masks->mask;

    // For each bit in the PPP, shift right only those bits that are set in
    // that bit's mask
    for (int i = 0; i < N_BITS; i++) {
        uint64_t shift = 1 << i;
        uint64_t bit = masks->ppp_bit[i];
        // Shift only the input bits that are set in
        a = (a & ~bit) | ((a & bit) >> shift);
    }
    return a;
}

uint64_t zp7_pext_64(uint64_t a, uint64_t mask) {
    zp7_masks_64_t masks = zp7_ppp_64(mask);
    return zp7_pext_pre_64(a, &masks);
}

// PDEP

uint64_t zp7_pdep_pre_64(uint64_t a, const zp7_masks_64_t *masks) {
    // For each iteration, clear the bits to accommodate the bits that are
    // being shifted in.
    for (int i = N_BITS - 1; i >= 0; i--) {
        uint64_t shift = 1 << i;
        uint64_t bit = masks->ppp_bit[i];
        a = (a & ~bit) | ((a << shift) & bit);
    }
    // Finally, apply the mask to clear out bits not in the mask.
    return a & masks->mask;
}

uint64_t zp7_pdep_64(uint64_t a, uint64_t mask) {
    zp7_masks_64_t masks = zp7_ppp_64(mask);
    return zp7_pdep_pre_64(a, &masks);
}
