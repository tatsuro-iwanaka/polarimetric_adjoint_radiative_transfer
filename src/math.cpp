#include "math.hpp"

#include <cmath>
#include <cassert>
#include <numbers>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>

namespace paad::math
{

std::vector<std::vector<double>> computeGaussRadauQuadratureNodeWeight(int n)
{
	assert(n >= 2);
	const int m = n - 1;
	
	const double a = 1.0;
	const double b = 0.0;

	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(m, m);

	for (int k = 0; k < m; ++k)
	{
		J(k, k) = (b * b - a * a) / ((2.0 * k + a + b) * (2.0 * k + a + b + 2.0)); 

		if (k < m - 1)
		{
			double num = 4.0 * (k + 1.0) * (k + a + 1.0) * (k + b + 1.0) * (k + a + b + 1.0);
			double den = (2.0 * k + a + b + 1.0) * (2.0 * k + a + b + 3.0) * std::pow(2.0 * k + a + b + 2.0, 2);
			double e = std::sqrt(num / den);

			J(k, k + 1) = e;
			J(k + 1, k) = e;
		}
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
	auto evals = es.eigenvalues();

	std::vector<double> xi(n);
	for (int i = 0; i < m; ++i)
	{
		xi[i] = evals(i);
	}
	xi[m] = 1.0;
	
	std::vector<double> Wi(n);
	for (int i = 0; i < m; ++i)
	{
		double Pnm1 = std::legendre(n - 1, xi[i]);
		Wi[i] = (1.0 + xi[i]) / (static_cast<double>(n) * static_cast<double>(n) * Pnm1 * Pnm1);
	}
	Wi[m] = 2.0 / (static_cast<double>(n) * static_cast<double>(n));

	std::vector<std::vector<double>> out(n, std::vector<double>(2));
	for (int i = 0; i < n; ++i)
	{
		double x01 = 0.5 * (xi[i] + 1.0);
		double w01 = 0.5 * Wi[i];
		out[i][0] = x01;
		out[i][1] = w01;
	}
	
	return out;
}

std::vector<std::vector<double>> computeGaussLegendreQuadratureNodeWeight(int degree)
{
	if (degree <= 0)
	{
		throw std::runtime_error("Degree must be positive for Gaussian-Legendre quadrature.");
	}

	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(degree, degree);
	
	for (int k = 1; k < degree; ++k)
	{
		double k_double = static_cast<double>(k);
		double beta_k = k_double / std::sqrt(4.0 * k_double * k_double - 1.0);

		J(k - 1, k) = beta_k;
		J(k, k - 1) = beta_k;
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
	
	if (es.info() != Eigen::Success)
	{
		throw std::runtime_error("Eigenvalue solution failed for Legendre Jacobi matrix.");
	}

	std::vector<std::vector<double>> node_weight(degree, std::vector<double>(2));
	const Eigen::VectorXd& eigenvalues = es.eigenvalues();
	const Eigen::MatrixXd& eigenvectors = es.eigenvectors();
	
	for (int i = 0; i < degree; ++i)
	{
		node_weight[i][0] = eigenvalues(i);
		double v_i_1 = eigenvectors(0, i); 
		node_weight[i][1] = 2.0 * v_i_1 * v_i_1;
	}

	std::sort(node_weight.begin(), node_weight.end(), [](const std::vector<double>& a, const std::vector<double>& b){return a[0] < b[0];});

	return node_weight;
}

std::vector<std::vector<double>> computeGaussHermiteQuadratureNodeWeight(int degree)
{
	if (degree <= 0)
	{
		throw std::runtime_error("Degree must be positive for Gaussian-Hermite quadrature.");
	}
	
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(degree, degree);
	
	for (int k = 1; k < degree; ++k)
	{
		double sub_diag_value = std::sqrt(static_cast<double>(k) / 2.0);
		J(k - 1, k) = sub_diag_value;
		J(k, k - 1) = sub_diag_value;
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
	
	if (es.info() != Eigen::Success)
	{
		throw std::runtime_error("Eigenvalue solution failed for Hermite Jacobi matrix.");
	}

	std::vector<std::vector<double>> node_weight(degree, std::vector<double>(2));
	
	const Eigen::VectorXd& eigenvalues = es.eigenvalues();
	const Eigen::MatrixXd& eigenvectors = es.eigenvectors();

	double sqrt_pi = std::sqrt(std::numbers::pi);
	
	for (int i = 0; i < degree; ++i)
	{
		node_weight[i][0] = eigenvalues(i); 
		double v_i_1 = eigenvectors(0, i);
		node_weight[i][1] = sqrt_pi * v_i_1 * v_i_1;
	}

	std::sort(node_weight.begin(), node_weight.end(), [](const std::vector<double>& a, const std::vector<double>& b){return a[0] < b[0];});

	return node_weight;
}

std::vector<std::vector<double>> computeGaussLaguerreQuadratureNodeWeight(int degree)
{
	if (degree <= 0)
	{
		throw std::runtime_error("Degree must be positive for Gaussian-Laguerre quadrature.");
	}
	
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(degree, degree);
	
	for (int i = 0; i < degree; ++i)
	{
		double i_double = static_cast<double>(i);
		J(i, i) = 2.0 * i_double + 1.0;
		
		if (i < degree - 1)
		{
			double off_diag_value = i_double + 1.0;
			J(i, i + 1) = off_diag_value; 
			J(i + 1, i) = off_diag_value;
		}
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
	
	if (es.info() != Eigen::Success)
	{
		throw std::runtime_error("Eigenvalue solution failed for Laguerre Jacobi matrix.");
	}

	std::vector<std::vector<double>> node_weight(degree, std::vector<double>(2));
	
	const Eigen::VectorXd& eigenvalues = es.eigenvalues();
	const Eigen::MatrixXd& eigenvectors = es.eigenvectors();
	
	for (int i = 0; i < degree; ++i)
	{
		node_weight[i][0] = eigenvalues(i); 
		double v_i_1 = eigenvectors(0, i);
		node_weight[i][1] = v_i_1 * v_i_1;
	}

	std::sort(node_weight.begin(), node_weight.end(), [](const std::vector<double>& a, const std::vector<double>& b){return a[0] < b[0];});

	return node_weight;
}

std::vector<std::vector<double>> computeGeneralizedGaussLaguerreQuadratureNodeWeight(int degree, double alpha_prime)
{
	if (degree <= 0)
	{
		throw std::runtime_error("Degree must be positive for Generalized Gaussian-Laguerre quadrature.");
	}
	
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(degree, degree);
	
	for (int i = 0; i < degree; ++i)
	{
		double i_double = static_cast<double>(i);
		J(i, i) = 2.0 * i_double + 1.0 + alpha_prime;
		
		if (i < degree - 1)
		{
			double k_plus_1 = i_double + 1.0;
			double off_diag_value = std::sqrt(k_plus_1 * (k_plus_1 + alpha_prime));
			
			J(i, i + 1) = off_diag_value; 
			J(i + 1, i) = off_diag_value;
		}
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
	
	if (es.info() != Eigen::Success)
	{
		throw std::runtime_error("Eigenvalue solution failed for Generalized Laguerre Jacobi matrix.");
	}

	std::vector<std::vector<double>> node_weight(degree, std::vector<double>(2));
	
	const Eigen::VectorXd& eigenvalues = es.eigenvalues();
	const Eigen::MatrixXd& eigenvectors = es.eigenvectors();
	
	double gamma_factor = std::tgamma(alpha_prime + 1.0);
	
	for (int i = 0; i < degree; ++i)
	{
		node_weight[i][0] = eigenvalues(i);
		double v_i_1 = eigenvectors(0, i); 
		node_weight[i][1] = gamma_factor * v_i_1 * v_i_1;
	}

	std::sort(node_weight.begin(), node_weight.end(), [](const std::vector<double>& a, const std::vector<double>& b){return a[0] < b[0];});

	return node_weight;
}

}
