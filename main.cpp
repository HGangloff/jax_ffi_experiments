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
