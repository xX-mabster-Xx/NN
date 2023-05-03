#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <cmath>
#include <fstream>
#include <chrono>
#include <random>
using namespace std;

class Neuron{
public:
    double value, activated_value, derived_value;
    void activate() {
        activated_value = 1 / (1 + exp(-value));
    }
    void derive() {
        derived_value = activated_value * (1 - activated_value);
    }
    void setValue(double d) {
        value = d;
        activate();
        derive();
    }
};
double random1(int seed, int mx = 10000) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> r(0,0x7fffffff);
    double n = r(rng) % (mx + 1);
    n -= mx / 2;
    return (double)(n) / mx * 4;
}
class Neuro_net {
public:
    int inputLayerSize; // first layer size
    int outputLayerSize;
    int n; // num of neurons
    double E = 0.7, A = 0.3; //
    vector <Neuron> neu;//neurones
    vector < vector <pair <int, double> > > g;//graph
    vector < vector <pair <int, double> > > g2;//last deltas of sinapses
    vector <double> delta;
    Neuro_net(vector <int> layers) {
        auto t1 = std::chrono::high_resolution_clock::now();
        int seed = std::chrono::duration_cast<std::chrono::nanoseconds>(t1.time_since_epoch()).count();
        int t = layers.size();
        vector <int> pref(t + 1, 0);
        for (int i = 0; i < t; ++i) {
            pref[i + 1] = pref[i] + layers[i];
        }
        n = pref.back();
        inputLayerSize = pref[1];
        outputLayerSize = pref.back() - pref[pref.size() - 2];
        neu.resize(n);
        g.resize(n);
        delta.resize(n);
        g2.resize(n);
        for (int i = 0; i < t - 1; ++i) {
            for (int j = pref[i]; j < pref[i + 1]; ++j) {
                for (int q = pref[i + 1]; q < pref[i + 2]; ++q) {
                    g[j].push_back({q, random1(seed * q + i + j)});
                    g2[j].push_back({q, 0});
                }
            }
        }
    }
    Neuro_net(vector <int> layers, int seed) {
        vector <int> pref(layers.size() + 1, 0);
        for (int i = 0; i < layers.size(); ++i) {
            pref[i + 1] = pref[i] + layers[i];
        }
        n = pref.back();
        inputLayerSize = pref[1];
        outputLayerSize = pref.back() - pref[pref.size() - 2];
        neu.resize(n);
        g.resize(n);
        delta.resize(n);
        g2.resize(n);
        for (int i = 0; i < layers.size() - 1; ++i) {
            for (int j = pref[i]; j < pref[i + 1]; ++j) {
                for (int q = pref[i + 1]; q < pref[i + 2]; ++q) {
                    g[j].push_back({q, random1(seed * q + i + j)});
                    g2[j].push_back({q, 0});
                }
            }
        }
    }
    void consts(double e, double a) {
        E = e;
        A = a;
    }
    void getAns() {
        for (int i = 0; i < inputLayerSize; ++i) {
            neu[i].activated_value = neu[i].value;
            for (auto x : g[i]) {
                neu[x.first].value += neu[i].value * x.second;
            }
        }
        for (int i = inputLayerSize; i < g.size(); ++i) {
            neu[i].activate();
            for (auto x : g[i]) {
                neu[x.first].value += neu[i].activated_value * x.second;
            }
        }
    }
    void deltas(vector <double> ideal) {
        if (ideal.size() != outputLayerSize) {
            cout << "Error:" << ideal.size() << "!=" << outputLayerSize << "\n";
            return;
        }
        for (int i = n - 1; i > n - outputLayerSize; --i) {
            neu[i].derive();
            delta[i] = (ideal[i + ideal.size() - n] - neu[i].activated_value) * neu[i].derived_value;
        }
        for (int i = n - 2; i >= 0; --i) {
            neu[i].derive();
            double summ = 0;
            for (auto x: g[i]) {
                summ += x.second * delta[x.first];
            }
            delta[i] = neu[i].derived_value * summ;
            //sinaps update
            for (int j = 0; j < g[i].size(); ++j) {
                g2[i][j].second = E * (neu[i].activated_value * delta[g[i][j].first]) + A * g2[i][j].second;
                g[i][j].second += g2[i][j].second;
            }
        }
    }
    void clear_values() {
        for (int i = inputLayerSize; i < n; ++i) {
            neu[i].value = 0;
        }
    }
};