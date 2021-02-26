#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  
#include <pybind11/stl.h>    
#include "celllist.hpp"
#include "soapgto.hpp"
#include "kernel.hpp"
#include "gylm.hpp"
#include "ylm.hpp"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(_gylm, m) {
    m.def("evaluate_power", &_py_evaluate_xtunkl, "Power spectra for tnlm-type tensors");
    m.def("evaluate_gylm", &evaluate_gylm, "Gnl-Ylm frequency-damped convolutions");
    m.def("evaluate_soapgto", &soapGTO, "SOAP with gaussian type orbital radial basis set");
    m.def("smooth_match", &smooth_match, "Smooth best-match assignment");
    m.def("ylm", &_py_ylm, "Spherical harmonic series");
    py::class_<CellList>(m, "CellList")
        .def(py::init<py::array_t<double>, double>())
        .def("getNeighboursForIndex", &CellList::getNeighboursForIndex)
        .def("getNeighboursForPosition", &CellList::getNeighboursForPosition);
    py::class_<CellListResult>(m, "CellListResult")
        .def(py::init<>())
        .def_readonly("indices", &CellListResult::indices)
        .def_readonly("distances", &CellListResult::distances)
        .def_readonly("distances_squared", &CellListResult::distancesSquared);
}
