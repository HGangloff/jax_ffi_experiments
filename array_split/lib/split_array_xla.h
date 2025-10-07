#ifndef SPLIT_ARRAY_XLA_H_
#define SPLIT_ARRAY_XLA_H_

#include <iostream>
#include <cstring>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

XLA_FFI_DECLARE_HANDLER_SYMBOL(split_jnp_array);
XLA_FFI_DECLARE_HANDLER_SYMBOL(split_jnp_array_bwd);

#endif SPLIT_ARRAY_XLA_H_
