#include <pybind11/pybind11.h>
#include "gibbs_sampler_xla.h"

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(reinterpret_cast<void*>(fn));
}

PYBIND11_MODULE(gibbs_sampler, m) {   // please match the pybind_extension target name
  m.def("gibbs_sampler", []() { return EncapsulateFunction(GibbsSampler); });
}

