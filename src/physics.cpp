#include "physics.hpp"

#include <cmath>
#include <algorithm>
#include "constants.hpp"

namespace paad::physics
{

double interpolateVerticalProfile(double x, const std::vector<std::vector<double>>& table, VerticalProfileInterpolation type)
{
	if (table.empty())
	{
		return 0.0;
	}

	if (x <= table.front()[0])
	{
		return table.front()[1];
	}

	if (x >= table.back()[0])
	{
		return table.back()[1];
	}

	auto it = std::lower_bound(table.begin(), table.end(), x, [](const std::vector<double>& row, double val){return row[0] < val;});
	size_t j = std::distance(table.begin(), it) - 1;

	double x0 = table[j][0];
	double x1 = table[j + 1][0];
	double y0 = table[j][1];
	double y1 = table[j + 1][1];

	if (std::abs(x1 - x0) < 1.0e-9)
	{
		return y0;
	}

	double r = (x - x0) / (x1 - x0);

	if (type == VerticalProfileInterpolation::Linear)
	{
		return ((1.0 - r) * y0 + r * y1);
	}
	else
	{
		if (y0 <= 0.0 || y1 <= 0.0) 
		{
			return ((1.0 - r) * y0 + r * y1);
		}
		double log0 = std::log(y0);
		double log1 = std::log(y1);
		
		return std::exp((1.0 - r) * log0 + r * log1);
	}
}

double computePlanckFunction(double spectral, double temperature, SpectralCoordinateDimension dimension)
{
	double B = 0.0;

	if (dimension == SpectralCoordinateDimension::Wavelength)
	{
		B = 2.0 * paad::constants::SPEED_OF_LIGHT * paad::constants::SPEED_OF_LIGHT * paad::constants::PLANCK_CONSTANT;
		B /= std::pow(spectral, 5);
		B /= (std::exp(paad::constants::PLANCK_CONSTANT * paad::constants::SPEED_OF_LIGHT / (spectral * paad::constants::BOLTZMANN_CONSTANT * temperature)) - 1.0);
	}
	else if (dimension == SpectralCoordinateDimension::Wavenumber)
	{
		B = 2.0 * paad::constants::PLANCK_CONSTANT * paad::constants::SPEED_OF_LIGHT * paad::constants::SPEED_OF_LIGHT * std::pow(spectral, 3);
		B /= (std::exp(paad::constants::PLANCK_CONSTANT * paad::constants::SPEED_OF_LIGHT * spectral / (paad::constants::BOLTZMANN_CONSTANT * temperature)) - 1.0);
	}

	return B;
}

double computeThermalEmission(double planck_function, double tau, double mu)
{
	return planck_function * (-std::expm1(-tau / mu));
}

double computePlanckFunctionDerivative(double spectral, double temperature, SpectralCoordinateDimension dimension)
{
	double X = 0.0;

	if (dimension == SpectralCoordinateDimension::Wavelength)
	{
		X = paad::constants::PLANCK_CONSTANT * paad::constants::SPEED_OF_LIGHT / (spectral * paad::constants::BOLTZMANN_CONSTANT * temperature);		
	}
	else if (dimension == SpectralCoordinateDimension::Wavenumber)
	{
		X = paad::constants::PLANCK_CONSTANT * paad::constants::SPEED_OF_LIGHT * spectral / (paad::constants::BOLTZMANN_CONSTANT * temperature);
	}

	return computePlanckFunction(spectral, temperature, dimension) / temperature * X / (-std::expm1(-X));
}

}
