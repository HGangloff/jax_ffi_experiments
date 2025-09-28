#include <cmath>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>
#include <functional>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

extern void initialize(int* image, int rows, int cols, int Q);
extern void RunGibbsSampler(int* image, int rows, int cols, int Q, float beta, int iter);

namespace ffi = xla::ffi;

ffi::Error GibbsSamplerImpl(
    //ffi::BufferR1<ffi::DataType::S32> in_img,
    ffi::Result<ffi::BufferR2<ffi::DataType::S32>> img,
    int rows,
    int cols,
    int Q,
    float beta,
    int iter
) {
    initialize(img->typed_data(), rows, cols, Q);
    RunGibbsSampler(img->typed_data(), rows, cols, Q, beta, iter);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GibbsSampler, GibbsSamplerImpl,
    ffi::Ffi::Bind()
        // .Arg<ffi::BufferR1<ffi::DataType::S32>>()
        .Ret<ffi::BufferR2<ffi::DataType::S32>>()
        .Attr<int>("rows")
        .Attr<int>("cols")
        .Attr<int>("Q")
        .Attr<float>("beta")
        .Attr<int>("iter")
);

