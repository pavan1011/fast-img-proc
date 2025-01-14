#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "image/image.h"
#include "processing/processor.h"

namespace nb = nanobind;

NB_MODULE(fast_image_processing, m) {
    // Bind Hardware enum
    nb::enum_<processing::Hardware>(m, "Hardware")
        .value("AUTO", processing::Hardware::AUTO)
        .value("CPU", processing::Hardware::CPU)
        .value("GPU", processing::Hardware::GPU);

    // Bind Image class
    nb::class_<Image>(m, "Image")
        .def(nb::init<const std::string&>())
        .def(nb::init<u_int32_t, uint32_t, uint8_t>())
        .def("save", &Image::save)
        .def_prop_ro("width", &Image::width)
        .def_prop_ro("height", &Image::height)
        .def_prop_ro("channels", &Image::channels);

    // Bind Hardware check functions
    m.def("is_gpu_available", &processing::is_gpu_available,
          "Check if GPU processing is available");
    m.def("get_active_hardware", &processing::get_active_hardware,
          "Get currently active hardware");

    // Bind grayscale function
    m.def("grayscale", &processing::grayscale,
          "Convert an RGB image to grayscale",
          nb::arg("input"), nb::arg("hardware") = processing::Hardware::AUTO);

    // Bind processing functions
    m.def("equalize_histogram", &processing::equalize_histogram,
          "Equalize Histogram (accepts RGB and Grayscale image)",
          nb::arg("input"), nb::arg("hardware") = processing::Hardware::AUTO);
}