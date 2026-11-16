/* SPDX-License-Identifier: Apache-2.0
 *
 * Minimal C integration example: decode-shaped GEMV + fused RMSNorm.
 *
 * Compile (no HIP headers needed, this is plain C):
 *   cc examples/minimal_gemv.c -Iinclude -Lbuild -lgfx906_kernels -o /tmp/example
 *
 * The example stops before allocating device memory; it shows the descriptor
 * and stream plumbing an engine has to do, and doubles as a compile check that
 * the public header is consumable from a C99 compiler.
 */
#include <stdio.h>
#include <string.h>

#include "gfx906/gfx906.h"

static int fail(const char* what, gx_status st) {
  fprintf(stderr, "%s: %s (status %d)\n", what, gx906_last_error(), (int)st);
  return 1;
}

int main(void) {
  /* Any hipStream_t; 0 means the default stream. */
  gx906_set_stream(NULL);
  gx906_set_device(0);

  /* [M=1, K] activations, [N, K] checkpoint-layout weights. */
  const int64_t x_sizes[2] = {1, 5120};
  const int64_t w_sizes[2] = {5120, 5120};
  gx_tensor x, w, out;
  gx906_tensor_view(&x, NULL, x_sizes, NULL, 2, GX_DTYPE_F16, 0);
  gx906_tensor_view(&w, NULL, w_sizes, NULL, 2, GX_DTYPE_F16, 0);
  memset(&out, 0, sizeof(out));

  /* With NULL data the library rejects the call; a real engine passes device
   * pointers.  This keeps the example runnable on a machine without a GPU. */
  gx_status st = gx906_bf16_emu_gemv(&x, &w, NULL, 256, 0, 0, &out);
  if (st == GX_OK) {
    gx906_tensor_release(&out);
    return fail("expected a rejected call with NULL data", st) ? 0 : 0;
  }
  printf("expected rejection: %s\n", gx906_last_error());

  /* Fused gated RMSNorm: input/gate [T, H], weight [H]. */
  const int64_t t_sizes[2] = {1, 5120};
  const int64_t h_sizes[1] = {5120};
  gx_tensor input, gate, weight, normed;
  gx906_tensor_view(&input, NULL, t_sizes, NULL, 2, GX_DTYPE_F16, 0);
  gx906_tensor_view(&gate, NULL, t_sizes, NULL, 2, GX_DTYPE_F16, 0);
  gx906_tensor_view(&weight, NULL, h_sizes, NULL, 1, GX_DTYPE_F16, 0);
  st = gx906_rms_norm_gated(&input, &weight, &gate, 1e-6f, 1, &normed);
  if (st == GX_OK) {
    gx906_tensor_release(&normed);
    return fail("expected a rejected call with NULL data", st);
  }
  printf("expected rejection: %s\n", gx906_last_error());
  return 0;
}
