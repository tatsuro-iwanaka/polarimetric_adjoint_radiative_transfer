#pragma once

#include <vector>
#include "enums.hpp"

namespace paad::physics
{

double interpolateVerticalProfile(double x, const std::vector<std::vector<double>>& table, VerticalProfileInterpolation type);

double computePlanckFunction(double spectral, double temperature, SpectralCoordinateDimension dimension);
double computeThermalEmission(double planck_function, double tau, double mu);
double computePlanckFunctionDerivative(double spectral, double temperature, SpectralCoordinateDimension dimension);

}
