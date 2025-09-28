#include <pybind11/pybind11.h>
#include "split_array_xla.h"

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(reinterpret_cast<void*>(fn));
}

PYBIND11_MODULE(split_array, m) {   // please match the pybind_extension target name
  m.def("split_array", []() { return EncapsulateFunction(split_jnp_array); });
}
