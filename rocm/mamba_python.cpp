#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>
using namespace std;

void test()
{
    cout << "Hello World" << endl;
}

PYBIND11_MODULE(mamba, m) {
    m.doc() = "Mamba for ROCm module";
}
