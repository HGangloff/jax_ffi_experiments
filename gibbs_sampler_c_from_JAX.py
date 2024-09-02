import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.path.insert(0, 'bazel-bin/lib/')
import time

import jax
import jax.extend as jex
import jax.numpy as jnp
import numpy as np

key = jax.random.PRNGKey(0)

import gibbs_sampler as gibbs_sampler_lib

jex.ffi.register_ffi_target("gibbs_sampler", gibbs_sampler_lib.gibbs_sampler(), platform="cpu")

# Note that JIT compilation of this function would be useless since we already
# linked to compiled code
def gibbs_sampler_cpp(rows, cols, Q, beta, n_iter):

    # No need to pre allocate res! There is no input argument to the FFI call
    # res = jnp.empty((rows, cols), jnp.int32)

    out_type = jax.ShapeDtypeStruct((rows, cols), jnp.int32)

    return jex.ffi.ffi_call(
        # The target name must be the same string as we used to register the target
        # above in `register_custom_call_target`
        "gibbs_sampler",
        out_type,
        # No args!!!
        vectorized=False,
        # Note that here we're use `numpy` (not `jax.numpy`) to specify a dtype for
        # the attribute `eps`. Our FFI function expects this to have the C++ `float`
        # type (which corresponds to numpy's `float32` type), and it must be a
        # static parameter (i.e. not a JAX array).
        rows=np.int32(rows),
        cols=np.int32(cols),
        Q=np.int32(Q),
        beta=np.float32(beta),
        iter=np.int32(n_iter),
    )


rows = 200
cols = 200
Q = 3
beta = 1
n_iter = 1000

start = time.time()
res = gibbs_sampler_cpp(rows, cols, Q, beta, n_iter)
end = time.time()

print(f"Time: {end - start} seconds")
