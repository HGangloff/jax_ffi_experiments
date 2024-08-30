// Thanks chatgpt for the quick gibbs sampler implementation

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "gibbs_sampler.h"

using namespace std;

// Function to initialize the image with random states
void initialize(int* image, int rows, int cols, int Q) {
    srand(0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image[i * cols + j] = rand() % Q;
        }
    }
}

// Function to calculate the energy of a pixel at (i, j)
float localEnergy(int* image, int rows, int cols, int i, int j, int state, float beta) {
    float energy = 0.0;
    // Eight neighbors: including diagonal ones
    int neighbors[8][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}, // Cardinal neighbors
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1} // Diagonal neighbors
    };

    for (int n = 0; n < 8; n++) {
        int ni = i + neighbors[n][0];
        int nj = j + neighbors[n][1];
        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
            if (image[ni * cols + nj] == state) {
                energy -= beta;
            } else {
                energy += beta;
            }
        }
    }

    return energy;
}

// Function to perform one iteration of the Gibbs sampler
void gibbsSampler_(int* image, int rows, int cols, int Q, float beta) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Compute the probability for each possible state
            float probs[Q];
            float sum = 0.0;
            for (int q = 0; q < Q; q++) {
                probs[q] = exp(-localEnergy(image, rows, cols, i, j, q, beta));
                sum += probs[q];
            }

            // Normalize the probabilities
            for (int q = 0; q < Q; q++) {
                probs[q] /= sum;
            }

            // Sample a new state based on the probabilities
            float r = (float)rand() / RAND_MAX;
            float cumulative = 0.0;
            for (int q = 0; q < Q; q++) {
                cumulative += probs[q];
                if (r < cumulative) {
                    image[i * cols + j] = q;
                    break;
                }
            }
        }
    }
}

void RunGibbsSampler(int* image, int rows, int cols, int Q, float beta, int iter) {
    // Perform Gibbs sampling
    for (int i = 0; i < iter; i++) {
        gibbsSampler_(image, rows, cols, Q, beta);
    }
}


// Function to print the image
void printImage(int* image, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << image[i * cols + j] << " ";
        }
        cout << endl;
    }
}
