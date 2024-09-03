# Objectives

In this tutorial, we will explore the `jax.extend.ffi` module, with a tutorial available [here](https://jax.readthedocs.io/en/latest/ffi.html). Specifically, we will adapt this module to use a naive Gibbs sampler algorithm written in C++ from JAX.

# C++ code

We have a small C++ library that implements a naive version of the Gibbs sampler. The three functions it offers are available in `lib/gibbs_sampler.h`:

```Cpp
void initialize(int* image, int rows, int cols, int Q);
void printImage(int* image, int rows, int cols);
void RunGibbsSampler(int* image, int rows, int cols, int Q, float beta, int iter);
```

The file `gibbs_sampler_cpp.cpp` times this implementation of the Gibbs sampler for $1000$ iterations on a $200\times200$ image for a Potts model with $3$ classes where $\beta=1.0$:

```Cpp
// Thanks chatgpt for the quick gibbs sampler implementation

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "lib/gibbs_sampler.h"

using namespace std;
using namespace std::chrono;  // Use the chrono namespace

// Main function
int main() {
    int rows = 200;
    int cols = 200;
    int Q = 3; // Number of possible labels (states)
    float beta = 1.0; // Coupling strength
    int iter = 1000;

    // Allocate memory for the image
    int* image = new int[rows * cols];

    auto start = high_resolution_clock::now();
    // Initialize the image with random states
    initialize(image, rows, cols, Q);

    // cout << "Initial image:" << endl;
    // printImage(image, rows, cols);

    RunGibbsSampler(image, rows, cols, Q, beta, iter);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Execution time: " << duration.count() << " milliseconds" << endl;

    // cout << "Final image after Gibbs sampling:" << endl;
    // printImage(image, rows, cols);

    // Free allocated memory
    delete[] image;

    return 0;
}
```

The above file is compiled and executed with the command:

```bash
bazel build main && bazel-bin/main
```

The build main instruction is found in `BUILD.bazel` (this is a simple executable `cc_binary` compilation):

```bash
cc_binary( # no header here, we are not building a library
    name = "main",
    srcs = ["gibbs_sampler_cpp.cpp"],
    visibility = ["//visibility:public"],
    deps = ["//lib:gibbs_sampler_lib"]
)
```

However, this build rule depends on another build rule, which is for the Gibbs sampler library `gibbs_sampler_lib`. This rule, executed automatically by dependency, is located in `lib/BUILD.bazel`:

```bash
cc_library(
    name = "gibbs_sampler_lib",
    srcs = ["gibbs_sampler.cpp"],
    hdrs = [
        "gibbs_sampler.h", # These headers are intended to be included by other libraries or binaries that depend on this library.
    ],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"],
)
```

Finally, running the program gives:

```bash
Execution time: 3718 milliseconds
```

Let's keep this value in mind as a reference for later. Also, note the illustration (see the git repository for the C++ code that includes saving the image):


![Result of `gibbs_sampler_cpp.cpp`](pure_cpp.png)

## C++ from JAX

### XLA wrapping

As indicated in the two available tutorials on the subject (in the [JAX documentation](https://jax.readthedocs.io/en/latest/ffi.html) and the [XLA documentation](https://openxla.org/xla/custom_call)), we will need to wrap the functions of our library in the XLA *custom call* API.

We start by downloading the API in three files, `api.h`, `c_api.h`, and `ffi.h`, which we place in `lib/xla/ffi/api/`. We then create a function `GibbsSamplerImpl` that will wrap `initialize` and `RunGibbsSampler` available in `gibbs_sampler.h`. This wrapping is done using specific objects: `Buffer` (representing `jax.numpy` arrays), `Datatype`, `Error`, `Bind`, and `Attr`. These objects act as intermediaries between the `jax.numpy` data structures and C++. Their usage is described in the tutorials mentioned above.

Interestingly, `GibbsSamplerImpl` does not take an array (`jnp.array`) as input, i.e., no `Bind.Arg<>()`; the only input arguments are *named attributes*, i.e., `Bind().Attr<>()`. However, there will be a return, which will be the resulting sampler array, i.e., `Bind().Ret<>()`. Finally, the actual function of interest that can be used via the `jax.extend` module is constructed by the `XLA_FFI_DEFINE_HANDLER_SYMBOL` macro and will be called `GibbsSampler`.


All these operations are written in
`lib/gibbs_sampler_xla.cpp`:

```cpp
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
```

To simplify the code, wrapping the `gibbs_sampler.h` library using the XLA API constitutes an intermediate library whose header is `gibbs_sampler_xla.h`. We compile it with the `gibbs_sampler_xla_lib` rule defined in `lib/BUILD.bazel`

```bash
cc_library(
    name = "gibbs_sampler_xla_lib",
    srcs = [
        "xla/ffi/api/c_api.h",
        "xla/ffi/api/ffi.h",
        "xla/ffi/api/api.h",
        "gibbs_sampler_xla.cpp",
            ], # When deciding whether to put a header into hdrs or srcs, you should ask whether you want consumers of this library to be able to directly include it
    hdrs = [
        "gibbs_sampler_xla.h",
    ],
    includes = [ # need to use include -I option is not available due to sandbox stuff
            ".", # adding to srcs or hdrs does not include
    ],
    copts = ["-std=c++17 -rdynamic"], # -rdynamic to use extern
    deps = ["//lib:gibbs_sampler_lib"],
    visibility = ["//visibility:public"],
)
```

Note that this rule requires the execution of the `gibbs_sampler_lib` rule.

### Creating the Python Module

As in the *C++ from Python* tutorial, we will use `pybind` and `PyCapsule` for this part of the operations. We will create a Python module called `gibbs_sampler`, which will contain a function also called `gibbs_sampler`. Calling this function will return a `PyCapsule` containing the `GibbsSampler` function defined at the end of the previous section. The code that accomplishes this is in `lib/gibbs_sampler_xla_pybind.cc`.

```cpp
#include <pybind11/pybind11.h>
#include "gibbs_sampler_xla.h"

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(reinterpret_cast<void*>(fn));
}

PYBIND11_MODULE(gibbs_sampler, m) {   // please match the pybind_extension target name
  m.def("gibbs_sampler", []() { return EncapsulateFunction(GibbsSampler); });
}
```

The code is compiled into a Python module with the `gibbs_sampler` rule defined in `lib/BUILD.bazel`:

```bash
pybind_extension( # must be in two steps (first cc_library then pybind_extension)
    name = "gibbs_sampler", # must match the pybind module name
    srcs = ["gibbs_sampler_xla_pybind.cc"],
    deps = ["//lib:gibbs_sampler_xla_lib"],
    copts = ["-std=c++17 -rdynamic"]
)
```

Note the dependency on the `gibbs_sampler_xla_lib` rule.

Finally, the Python module can be compiled with the simple command:

```bash
bazel build //lib:gibbs_sampler
```

### Using the Python Module with `jax.extend.ffi`

We are finally able to use the C++ function from JAX!

In the file `gibbs_sampler_c_from_JAX.py`:

1) We register our FFI call (name, PyCapsule, platform):


```bash
jex.ffi.register_ffi_target("gibbs_sampler", gibbs_sampler_lib.gibbs_sampler(), platform="cpu")
```

2) We define a Python function `gibbs_sampler_cpp` that makes the call to `GibbsSampler`, the function hidden in the PyCapsule, using `jax.ffi.ffi_call`. Note that this call is now quite simple. We need to specify the dimensions and type of the returned result with a `jax.ShapeDtypeStruct` object, and we do not need to specify input arguments since we only have attributes passed as *keyword arguments*.

3) We perform the call with the same parameters as before and time it:


```python
rows = 200
cols = 200
Q = 3
beta = 1
n_iter = 1000

start = time.time()
res = gibbs_sampler_cpp(rows, cols, Q, beta, n_iter)
end = time.time()

print(f"Time: {end - start} seconds")
```

Running this script gives:

```bash
Time: 3.9689433574676514 seconds
```

This is a time slightly higher than calling the same function directly from C++ with the same parameters. Thus, we achieve the expected result in terms of time, as well as in the illustration:

![Result of `gibbs_sampler_c_from_JAX.py`](c_from_JAX.png)

## Comparison with Python / JAX

The JAX library in Python offers the key feature of Just-In-Time (JIT) compilation, which allows for the compilation of Python code at the first execution for much faster subsequent executions, **generally**.

### On CPU

The [mrfx](https://github.com/HGangloff/mrfx) library provides a pure JAX implementation of the Gibbs sampler and the chromatic Gibbs sampler. We can obtain timing for the Gibbs sampler using the code in the file `gibbs_sampler_JAX_cpu.py`:


```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import time

import jax

key = jax.random.PRNGKey(0)

from mrfx.models import Potts
from mrfx.samplers import GibbsSampler
from mrfx.experiments import time_complete_sampling

rows = 200
cols = 200
Q = 3
beta = 1
n_iter = 1000

key, subkey = jax.random.split(key, 2)
times, n_iterations = time_complete_sampling(
    GibbsSampler,
    Potts,
    subkey,
    [Q],
    [(rows, cols)],
    5,
    kwargs_sampler={"eps": 0.01, "max_iter":n_iter, "cv_type":"iter_only"},
    kwargs_model={"beta":1.}
)
print("JAX on CPU", times)
```

We note a displayed compilation time of $7.7714$ seconds for an execution time of $7.3613$.

### On GPU

The file `gibbs_sampler_JAX_gpu.py` allows for running the previous code on a GPU. However, it appears that the algorithm is about 10 times slower in both compilation and execution. This is likely due to the highly iterative nature of the algorithm, which is not well-suited to a GPU architecture...

## Conclusion

We have successfully called C++ code from JAX. We can conclude that for this highly iterative Gibbs sampling algorithm, JAX's Just-In-Time compilation does not surpass a direct call to C++ code. The advantage of using `jax.extend.ffi` here is significant.

