from functools import partial
from timeit import default_timer as timer
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.path.insert(0, 'bazel-bin/lib/')
import jax
import jax.numpy as jnp
import numpy as np


key = jax.random.PRNGKey(0)

import split_array as split_array_lib

jax.ffi.register_ffi_target("split_array", split_array_lib.split_array(),
                            api_version=1)

def split_array_cpp_fun(size_for_subarrays):

    out_types = tuple(
        jax.ShapeDtypeStruct((s,), jnp.float32)
        for s in size_for_subarrays
    )

    return jax.jit(jax.ffi.ffi_call(
        # The target name must be the same string as we used to register the target
        # above in `register_custom_call_target`
        "split_array",
        out_types,
        #input_output_aliases={i:i for i in range(len(out_types))} # not
        # possible since input_array does not have same shape as outputs
    ), static_argnames=["num_parts"])

def f_split_wrapper(sizes_cumsum):

    @partial(jax.jit)
    def f_(x):
        return jnp.split(x, sizes_cumsum)

    return f_

def time(f, key):
  for k in range(10):
    key, subkey = jax.random.split(key)
    x = jax.random.normal(key=subkey, shape=(10000,))
    start = timer()
    _ = jax.block_until_ready(f(x))
    end = timer()
    print(f'  call {k}: {end - start:0.6f} s')

def time2(f, key, s, n):
  for k in range(10):
    key, subkey = jax.random.split(key)
    x = jax.random.normal(key=subkey, shape=(10000,))
    start = timer()
    _ = jax.block_until_ready(f(x, s, num_parts=n))
    end = timer()
    print(f'  call {k}: {end - start:0.6f} s')


size_for_subarrays = jnp.array([50, 450] * 10, dtype=jnp.int32)
num_parts = np.int32(len(size_for_subarrays))

j_f_split_cpp = split_array_cpp_fun(size_for_subarrays)

j_f_split = f_split_wrapper(np.asarray(np.cumsum(size_for_subarrays)[:-1]))

print("Split c++")
time2(j_f_split_cpp, key, size_for_subarrays, num_parts)
print("jnp.split")
time(j_f_split, key)
