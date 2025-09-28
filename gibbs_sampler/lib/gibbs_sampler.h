#ifndef GIBBS_SAMPLER_H
#define GIBBS_SAMPLER_H

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

void initialize(int* image, int rows, int cols, int Q);
void printImage(int* image, int rows, int cols);
void RunGibbsSampler(int* image, int rows, int cols, int Q, float beta, int iter);

#endif
