#include "matrix.hpp"

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: benchmark [correctness|latency]\n";
        return 2;
    }
    std::string mode(argv[1]);
    if (mode == "correctness") {
        Matrix a = make_test_matrix(8);
        Matrix b = make_identity(8);
        Matrix result = multiply(a, b);
        if (is_close(a, result)) {
            std::cout << "correctness=true\n";
            return 0;
        }
        std::cout << "correctness=false\n";
        return 1;
    }
    if (mode == "latency") {
        Matrix a = make_test_matrix(64);
        Matrix b = make_test_matrix(64);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; ++i) {
            Matrix out = multiply(a, b);
            (void)out;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
        std::cout << "latency_ms=" << elapsed.count() << "\n";
        return 0;
    }
    std::cerr << "Unknown mode: " << mode << "\n";
    return 2;
}
