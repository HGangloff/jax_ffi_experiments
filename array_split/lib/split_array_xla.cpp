#include <iostream>
#include <cstring>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace std;


// NOTE that AnyBuffer should be the prefered way since assigning a rank then
// causes problem when vmapping
ffi::Error split_jnp_array_(
    ffi::AnyBuffer input_array,
    ffi::AnyBuffer size_for_subarrays,
    ffi::RemainingRets rets,
    int num_parts
) {

    int* size_for_subarrays_ = reinterpret_cast<int *>(size_for_subarrays.untyped_data());
    float* ptr = reinterpret_cast<float *>(input_array.untyped_data());
    auto dims = input_array.dimensions();
    // std::cout << dims.front() << ", " << dims.back() << ", " << dims.size() << "\n";

    int offset;
    if (dims.size() == 1) {
        offset = 1;
    } else {
        offset = dims.front();
    }

    for (int i=0; i < num_parts; i++) {
        ffi::Result<ffi::AnyBuffer> ret = rets.get<ffi::AnyBuffer>(i).value();
        memcpy(ret->untyped_data(), ptr, size_for_subarrays_[i] * 4 * offset); // size in bytes
                                                 // (float=4bytes)
        ptr += size_for_subarrays_[i] * offset;
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    split_jnp_array,
    split_jnp_array_,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<int>("num_parts")
);

// NOTE that AnyBuffer should be the prefered way since assigning a rank then
// causes problem when vmapping
ffi::Error split_jnp_array_bwd_(
    ffi::RemainingArgs args,
    ffi::Result<ffi::AnyBuffer> ret_array,
    int num_parts
) {

    float* ptr = reinterpret_cast<float *>(ret_array->untyped_data());

    ffi::AnyBuffer arg = args.get<ffi::AnyBuffer>(0).value();
    //std::cout << ret_array->element_count() << "\n";
    auto dims = arg.dimensions();
    //std::cout << dims.front() << ", " << dims.back() << ", " << dims.size() << "\n";

    int offset;
    if (dims.size() == 1) {
        offset = 1;
    } else {
        offset = dims.front();
    }

    // for all the parts here I must consider a rank 2 buffer with batch_dim =
    // dim.front() and array dim = dim.back()
    // dim.size() == rank!
    //
    int cumulative_arg_size = 0;
    for (int i=0; i < num_parts; i++) {
        ffi::AnyBuffer arg = args.get<ffi::AnyBuffer>(i).value();
        auto dims = arg.dimensions();
        float* ptr_arg = static_cast<float *>(arg.untyped_data());

        // NOTE we need to save the return array in a row major order which
        // I cannot explain
        for (int j = 0; j < dims.front(); j += 1) {
            for (int k = 0; k < dims.back(); k += 1) {
                ptr[j + k * dims.front() + cumulative_arg_size] = *(ptr_arg + k + j * dims.back());

            }
        }
        cumulative_arg_size += arg.element_count();
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    split_jnp_array_bwd,
    split_jnp_array_bwd_,
    ffi::Ffi::Bind()
        .RemainingArgs()
        .Ret<ffi::AnyBuffer>()
        .Attr<int>("num_parts")
);

// Below this line remains some debugging

// An auxiliary function not used by the final split_jnp_array_ version
template <typename T>
void print_array(T* a, std::size_t N, std::ostream& o = std::cout)
{
  o << "{";
  for (std::size_t i = 0; i < N-1; ++i)
  {
    o << a[i] << ", ";
  }
  o << a[N-1] << "}\n";
}

// An auxiliary function not used by the final split_jnp_array_ version
void split_array(float **subarrays, float *input_array, int *parts, int num_parts) {
    float * ptr = input_array;
    for (int i=0; i< num_parts; i++) {
        memcpy(subarrays[i], ptr, parts[i] * 4); // size in bytes
                                                 // (float=4bytes)
        ptr += parts[i];
    }
}


int main () {

    int num_parts = 4;
    int N = 1000;
    int parts[4] = {250, 250, 250, 250};
    float *input_array = new float[N];
    for (int j = 0; j < N; ++j) {
        input_array[j] = j;
    }
    float **subarrays = new float*[num_parts];

    for (int i = 0; i < num_parts; i++) {
    }

    for (int i = 0; i < num_parts; ++i) {
        subarrays[i] = new float[parts[i]];
    }

    split_array(subarrays, input_array, parts, num_parts);

    print_array(input_array, N);

    for (int i = 0; i < num_parts; ++i) {
        print_array(subarrays[i], parts[i]);
    }

    delete[] input_array;

    for (int i = 0; i < num_parts; ++i) {
        delete [] subarrays[i];
    }
    delete[] subarrays;
}



///// Simpler version that does too much allocation
//ffi::Error split_jnp_array_(
//    ffi::BufferR1<ffi::DataType::F32> input_array,
//    ffi::BufferR1<ffi::DataType::S32> size_for_subarrays,
//    ffi::RemainingRets rets,
//    int num_parts
//) {
//
//    auto size_for_subarrays_ = size_for_subarrays.typed_data();
//
//    float** subarrays_ = new float*[num_parts];
//    for (int i = 0; i < num_parts; i++) {
//        subarrays_[i] = new float[size_for_subarrays_[i]];
//
//    }
//
//    split_array(subarrays_, input_array.typed_data(), size_for_subarrays.typed_data(), num_parts);
//
//    for (size_t i = 0; i < rets.size(); ++i) {
//        ffi::Result<ffi::AnyBuffer> ret = rets.get<ffi::AnyBuffer>(i).value();
//
//        memcpy(ret->untyped_data(), subarrays_[i], size_for_subarrays_[i] * 4);
//        delete [] subarrays_[i];
//    }
//
//    return ffi::Error::Success();
//}

////// More condensed approach without the auxiliary function
//ffi::Error split_jnp_array_(
//    ffi::BufferR1<ffi::DataType::F32> input_array,
//    ffi::BufferR1<ffi::DataType::S32> size_for_subarrays,
//    ffi::RemainingRets rets,
//    int num_parts
//) {
//
//    auto size_for_subarrays_ = size_for_subarrays.typed_data();
//
//    // we create an array of pointers to the pointers to the results
//    // (rets->untyped_data()) which are already allocated
//    // this saves one memcpy as in the previous version of the function
//    float** subarrays_ = new float*[num_parts];
//    for (int i = 0; i < num_parts; i++) {
//        ffi::Result<ffi::AnyBuffer> ret = rets.get<ffi::AnyBuffer>(i).value();
//        subarrays_[i] = static_cast<float *>(ret->untyped_data());
//
//    }
//
//    split_array(subarrays_, input_array.typed_data(), size_for_subarrays.typed_data(), num_parts);
//
//    delete [] subarrays_;
//
//    return ffi::Error::Success();
//}
