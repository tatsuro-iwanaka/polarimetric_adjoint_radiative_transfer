#pragma once

#include <vector>
#include <stdexcept>

namespace paad::math
{

std::vector<std::vector<double>> computeGaussRadauQuadratureNodeWeight(int n);

std::vector<std::vector<double>> computeGaussLegendreQuadratureNodeWeight(int degree);

std::vector<std::vector<double>> computeGaussHermiteQuadratureNodeWeight(int degree);

std::vector<std::vector<double>> computeGaussLaguerreQuadratureNodeWeight(int degree);

std::vector<std::vector<double>> computeGeneralizedGaussLaguerreQuadratureNodeWeight(int degree, double alpha_prime);

template <typename T> inline T computeSimpsonIntegration(const std::vector<std::vector<T>>& f)
{
	int n = f.size();

	if (n % 2 == 0)
	{
		throw std::invalid_argument("computeSimpsonIntegration: Number of data points must be odd.");
	}
	
	T result = T(0.0);

	for (int i = 0; i < n; i++)
	{
		if (i == 0 || i == n - 1)
		{
			result += f[i][1];
		}
		else if (i % 2 == 0)
		{
			result += T(2.0) * f[i][1];
		}
		else if (i % 2 == 1)
		{
			result += T(4.0) * f[i][1];
		}
	}

	result *= (f[1][0] - f[0][0]) / T(3.0);
	return result;
}

}
