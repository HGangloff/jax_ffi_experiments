// Thanks chatgpt for the quick gibbs sampler implementation

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <png++/png.hpp>

#include "lib/gibbs_sampler.h"

using namespace std;
using namespace std::chrono;  // Use the chrono namespace

void saveImageAsPNG(const char* filename, int* image, int rows, int cols) {
    try {
        // Create a grayscale image
        png::image<png::rgb_pixel> img(cols, rows);

        // Copy data from the int* array to the png++ image
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                img[y][x] = png::rgb_pixel(
                    image[y * cols + x] / 3. * 255,
                    image[y * cols + x] / 3. * 255,
                    image[y * cols + x] / 3. * 255
                );
            }
        }

        // Save the image
        img.write(filename);
    } catch (std::exception& e) {
        std::cerr << "Failed to save PNG file: " << e.what() << std::endl;
    }
}

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

    saveImageAsPNG("pure_cpp.png", image, rows, cols);

    // Free allocated memory
    delete[] image;

    return 0;
}
