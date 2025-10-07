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

jax.ffi.register_ffi_target("split_array",
                            split_array_lib.split_jnp_array(),
                            api_version=1)
jax.ffi.register_ffi_target("split_array_bwd",
                            split_array_lib.split_jnp_array_bwd(),
                            api_version=1)

def split_array_cpp_fun(size_for_subarrays):

    out_types = tuple(
        jax.ShapeDtypeStruct((s,), jnp.float32)
        for s in size_for_subarrays
    )

    def fwd_(x, s, num_parts):
        return jax.ffi.ffi_call(
        # The target name must be the same string as we used to register the target
        # above in `register_custom_call_target`
        "split_array",
        out_types,
        #input_output_aliases={i:i for i in range(len(out_types))} # not
        # possible since input_array does not have same shape as outputs
        vmap_method="broadcast_all"
    )(x, s, num_parts=num_parts)

    return jax.jit(fwd_, static_argnames=['num_parts'])

def split_array_cpp_fun_fwd(size_for_subarrays):

    out_types = tuple(
        jax.ShapeDtypeStruct((s,), jnp.float32)
        for s in size_for_subarrays
    )

    def fwd_(x, s, num_parts):
        return (jax.ffi.ffi_call(
        # The target name must be the same string as we used to register the target
        # above in `register_custom_call_target`
        "split_array",
        out_types,
        # possible since input_array does not have same shape as outputs
        vmap_method="broadcast_all"
    )(x, s, num_parts=num_parts), x)

    return jax.jit(fwd_, static_argnames=['num_parts'])

def split_array_cpp_fun_bwd(size_for_subarrays, num_parts):

    out_type = jax.ShapeDtypeStruct(
        (jnp.sum(size_for_subarrays),),
        jnp.float32
    )

    def bwd_(_, __, x, ct): # _, __ for s and n, find the reason in the doc!!
        res = (jax.ffi.ffi_call(
            "split_array_bwd",
            out_type,
            vmap_method="broadcast_all"
        )(*ct, num_parts=num_parts),)
        # ref = (jnp.concatenate([ct_ for ct_ in ct]),)
        
        return res

    return jax.jit(bwd_, static_argnums=(1,))



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
    _ = jax.block_until_ready(f(x, s, n))
    end = timer()
    print(f'  call {k}: {end - start:0.6f} s')


size_for_subarrays = jnp.array([50, 450] * 20, dtype=jnp.int32)
num_parts = np.int32(len(size_for_subarrays))

j_f_split_cpp = split_array_cpp_fun(size_for_subarrays)
j_f_split_cpp_fwd = split_array_cpp_fun_fwd(size_for_subarrays)
j_f_split_cpp_bwd = split_array_cpp_fun_bwd(size_for_subarrays, num_parts)

j_f_split_cpp = jax.custom_vjp(j_f_split_cpp, nondiff_argnums=(1, 2))
j_f_split_cpp.defvjp(j_f_split_cpp_fwd, j_f_split_cpp_bwd)


j_f_split = f_split_wrapper(np.asarray(np.cumsum(size_for_subarrays)[:-1]))

print("Test if splitting works")
key, subkey = jax.random.split(key)
x = jax.random.normal(key=subkey, shape=(10000,))
e = j_f_split_cpp(x, size_for_subarrays, num_parts=num_parts)
f = j_f_split(x)
assert all(tuple(jnp.allclose(e[i], f[i]) for i in range(len(f))))
print("splitting matches!")

print("Split c++")
time2(j_f_split_cpp, key, size_for_subarrays, num_parts)
print("jnp.split")
time(j_f_split, key)

print("Test if jax.customvjp is implemented")
a = jax.jacrev(j_f_split_cpp)(x, size_for_subarrays, num_parts=num_parts)
b = jax.jacrev(j_f_split)(x)
#print(j_f_split_cpp_bwd(None, None, None, e)[0].shape)
#print(x)
#print(j_f_split_cpp_bwd(None, None, None, e)[0])
assert all(tuple(jnp.allclose(a[i], b[i]) for i in range(len(a))))
print("jax.jacrev matches!")

rev_j_f_split_cpp = jax.jacrev(j_f_split_cpp)
rev_j_f_split = jax.jacrev(j_f_split)
print("jacrev of split c++")
time2(j_f_split_cpp, key, size_for_subarrays, num_parts)
print("jacrev of jnp.split")
time(j_f_split, key)

#print("Test if higher order derivatives work")
#print(jax.hessian(j_f_split_cpp)(x, size_for_subarrays, num_parts=num_parts))

