#ifndef GIBBS_SAMPLER_XLA_H_
#define GIBBS_SAMPLER_XLA_H_

#include <cmath>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>
#include <functional>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/api.h"

#include "gibbs_sampler.h"

XLA_FFI_DECLARE_HANDLER_SYMBOL(GibbsSampler);
#endif  // GIBBS_SAMPLER_XLA_H_

