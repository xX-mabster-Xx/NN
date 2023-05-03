#include <iostream>

#include "neuro.h"
#include <fstream>

ifstream fin("file.txt");

int main() {
    Neuro_net net({64, 120, 10, 1});
    int n,res;
    while (fin >> n >> res) {
        for (int j = 0; j < 10000000; ++j) {
            for (int i = 0; i < 64; ++i) {
                net.neu[i].value = (n >> i) & 1;
            }
            net.getAns();
            double ress = res;
            net.deltas({ress});
            net.clear_values();
        }
    }
    for (int g = 0; g < 500; ++g) {
        for (int i = 0; i < 64; ++i) {
            net.neu[i].value = (g >> i) & 1;
        }
        net.getAns();
        std::cout << g << ": " << net.neu.back().value << "\n";
        net.clear_values();
    }
}
