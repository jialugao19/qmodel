// cpp source code
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>

namespace py = pybind11;

class Permute64 {
private:
    uint64_t mask;
    uint64_t len;
    uint64_t seed;

public:
    Permute64(uint64_t length, uint64_t seed_val)
	: mask(length), len(length), seed(seed_val) {
        // check that length is no larger than 2**63 - 1
        if (length >= 0x8000000000000000) {
            throw std::invalid_argument("Length must be less than 2**63");
        }
        // Initialize mask and other parameters
        mask--;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
    }

    int64_t permute(int64_t index) const {
        int64_t ilen = len;
        uint64_t idx = (index % ilen + ilen) % ilen;
        do {
            idx ^= seed;

            // splittable64
            idx ^= (idx & mask) >> 30;
            idx *= 0xBF58476D1CE4E5B9ul;
            idx ^= (idx & mask) >> 27;
            idx *= 0x94D049BB133111EBul;
            idx ^= (idx & mask) >> 31;
            idx *= 0xBF58476D1CE4E5B9ul;

            idx ^= seed >> 32;
            idx &= mask;
            idx *= 0xED5AD4BBul;

            idx ^= seed >> 48;

            // hash16_xm3
            idx ^= (idx & mask) >> 7;
            idx *= 0x2993u;
            idx ^= (idx & mask) >> 5;
            idx *= 0xE877u;
            idx ^= (idx & mask) >> 9;
            idx *= 0x0235u;
            idx ^= (idx & mask) >> 10;

            // From Andrew Kensler: "Correlated Multi-Jittered Sampling"
            idx ^= seed;
            idx *= 0xe170893d;
            idx ^= seed >> 16;
            idx ^= (idx & mask) >> 4;
            idx ^= seed >> 8;
            idx *= 0x0929eb3f;
            idx ^= seed >> 23;
            idx ^= (idx & mask) >> 1;
            idx *= 1 | seed >> 27;
            idx *= 0x6935fa69;
            idx ^= (idx & mask) >> 11;
            idx *= 0x74dcb303;
            idx ^= (idx & mask) >> 2;
            idx *= 0x9e501cc3;
            idx ^= (idx & mask) >> 2;
            idx *= 0xc860a3df;
            idx &= mask;
            idx ^= idx >> 5;

        } while (idx >= len);

        return int64_t(idx);
    }
};

int main() {
    // Initialize with length and seed
    Permute64 p(10, 0x42);

    // Generate and print permutations
    for (int64_t i = -10; i < 30; ++i) {
        std::cout << p.permute(i) << (i % 10 != 9 ? " " : "\n") ;
    }

    return 0;
}


PYBIND11_MODULE(permute, m) {
    py::class_<Permute64>(m, "Permute64")
        .def(py::init<uint64_t, uint64_t>())
        .def("permute", py::vectorize(&Permute64::permute));
}
