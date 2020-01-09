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

#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h>

#define HAS_CLMUL
#define HAS_BZHI
#define HAS_POPCNT

#include "zp7.c"

#define ARRAY_SIZE(a)       (sizeof(a) / sizeof((a)[0]))

#define N_TESTS             (1 << 20)

// PRNG modified from the public domain RKISS by Bob Jenkins. See:
// http://www.burtleburtle.net/bob/rand/smallprng.html

typedef struct {
    uint64_t a, b, c, d; 
} rand_ctx_t;

uint64_t rotate_left(uint64_t x, uint64_t k) {
	return (x << k) | (x >> (64 - k));
}

uint64_t rand_next(rand_ctx_t *x) {
    uint64_t e = x->a - rotate_left(x->b, 7);
    x->a = x->b ^ rotate_left(x->c, 13);
    x->b = x->c + rotate_left(x->d, 37);
    x->c = x->d + e;
    x->d = e + x->a;
    return x->d;
}

void rand_init(rand_ctx_t *x) {
    x->a = 0x89ABCDEF01234567ULL, x->b = x->c = x->d = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < 1000; i++)
        (void)rand_next(x);
}

int main() {
    rand_ctx_t r[1];
    rand_init(r);
    uint64_t tests = 0;

    for (int test = 0; test < N_TESTS; test++) {
        // Create four masks with low/medium/high sparsity
        uint64_t mask = rand_next(r);
        uint64_t mask_2 = mask | rand_next(r) | rand_next(r);
        uint64_t masks[] = { mask, ~mask, mask_2, ~mask_2 };

        // For each input mask, test 32 random input values
        for (int i = 0; i < ARRAY_SIZE(masks); i++) {
            uint64_t m = masks[i];
            for (int j = 0; j < 32; j++) {
                uint64_t input = rand_next(r);

                // Test PEXT
                uint64_t e_1 = _pext_u64(input, m);
                uint64_t e_2 = zp7_pext_64(input, m);
                if (e_1 != e_2) {
                    printf("FAIL PEXT!\n");
                    printf("%016llx %016llx %016llx %016llx\n",
                            m, input, e_1, e_2);
                    exit(1);
                }
                tests++;

                // Test PDEP
                uint64_t d_1 = _pdep_u64(input, m);
                uint64_t d_2 = zp7_pdep_64(input, m);
                if (d_1 != d_2) {
                    printf("FAIL PDEP!\n");
                    printf("%016llx %016llx %016llx %016llx\n",
                            m, input, d_1, d_2);
                    exit(1);
                }
                tests++;
            }
        }
    }
    printf("Passed %llu tests.\n", tests);
    return 0;
}
