#include "forward.hpp"
#include <cmath>
#include <omp.h>

namespace paad::core
{

Eigen::MatrixXd expandDiagonal(const Eigen::VectorXd& vec, int n_stokes)
{
	int n = vec.size();
	Eigen::VectorXd expanded(n_stokes * n);
	
	for (int i = 0; i < n; ++i)
	{
		expanded.segment(n_stokes * i, n_stokes).fill(vec(i));
	}

	return expanded.asDiagonal();
}

Eigen::MatrixXd expandWMU(const Eigen::MatrixXd& WMU_small, int n_stokes)
{
	int n = WMU_small.rows();
	Eigen::MatrixXd WMU_large = Eigen::MatrixXd::Zero(n_stokes * n, n_stokes * n);
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_stokes, n_stokes);

	for (int i = 0; i < n; ++i)
	{
		WMU_large.block(n_stokes * i, n_stokes * i, n_stokes, n_stokes) = I * WMU_small(i, i);
	}

	return WMU_large;
}

void applyDelta34(Eigen::MatrixXd& M, int n_stokes)
{
	if (n_stokes < 3)
	{
		return;
	}

	for (int r = 0; r < M.rows(); ++r)
	{
		for (int c = 0; c < M.cols(); ++c)
		{
			bool r_neg = ((r % n_stokes) == 2 || (r % n_stokes) == 3);
			bool c_neg = ((c % n_stokes) == 2 || (c % n_stokes) == 3);

			if (r_neg != c_neg)
			{
				M(r, c) = -M(r, c);
			}
		}
	}
}

RadiativeLayer doubleLayer(const RadiativeLayer& layer, const geometry::Geometry& geometry, int n_parallel_fourier)
{
	int dim = layer.reflectance_m_top[0].rows();
	int n_stokes = dim / geometry.Ntheta;

	Eigen::VectorXd exp_tau_small = Eigen::VectorXd::Zero(geometry.Ntheta);

	for(int i = 0; i < geometry.Ntheta; i++)
	{
		exp_tau_small(i) = std::exp(-layer.optical_thickness / geometry.mu(i));
	}
	
	Eigen::MatrixXd E = expandDiagonal(exp_tau_small, n_stokes);
	Eigen::MatrixXd W = expandWMU(geometry.WMU, n_stokes);

	auto result = layer;
	result.optical_thickness *= 2.0;

	#pragma omp parallel for num_threads(n_parallel_fourier)
	for(int m = 0; m <= geometry.M; m++)
	{
		double factor = 2.0;

		Eigen::MatrixXd Q1 = factor * layer.reflectance_m_bottom[m] * W * layer.reflectance_m_top[m];
		Eigen::MatrixXd Q2 = factor * Q1 * W;
		
		Eigen::MatrixXd I_mat = Eigen::MatrixXd::Identity(dim, dim);
		Eigen::PartialPivLU<Eigen::MatrixXd> lu(I_mat - Q2);
		Eigen::MatrixXd S = lu.solve(Q1);

		Eigen::MatrixXd Sexp = S * E;
		Eigen::MatrixXd D = factor * S * W * layer.transmittance_m_top[m] + layer.transmittance_m_top[m] + Sexp;
		Eigen::MatrixXd U = factor * layer.reflectance_m_top[m] * W * D + layer.reflectance_m_top[m] * E;

		result.reflectance_m_top[m] = factor * layer.transmittance_m_bottom[m] * W * U + layer.reflectance_m_top[m] + E * U;
		result.transmittance_m_top[m] = factor * layer.transmittance_m_top[m] * W * D + layer.transmittance_m_top[m] * E + E * D;

		result.reflectance_m_bottom[m] = result.reflectance_m_top[m];
		result.transmittance_m_bottom[m] = result.transmittance_m_top[m];

		if (m > 0)
		{
			applyDelta34(result.reflectance_m_bottom[m], n_stokes);
			applyDelta34(result.transmittance_m_bottom[m], n_stokes);
		}
	}

	if (geometry.M >= 0)
	{
		int m = 0;
		double factor = 2.0;

		Eigen::MatrixXd Q1 = factor * layer.reflectance_m_bottom[m] * W * layer.reflectance_m_top[m];
		Eigen::MatrixXd Q2 = factor * Q1 * W;
		Eigen::MatrixXd I_mat = Eigen::MatrixXd::Identity(dim, dim);
		Eigen::PartialPivLU<Eigen::MatrixXd> lu(I_mat - Q2);

		Eigen::VectorXd vec_V = layer.source_down + factor * layer.reflectance_m_bottom[m] * W * layer.source_up;
		Eigen::VectorXd vec_D = lu.solve(vec_V);
		Eigen::VectorXd vec_U = layer.source_up + factor * layer.reflectance_m_top[m] * W * vec_D;
		
		result.source_up = layer.source_up + (factor * layer.transmittance_m_bottom[m] * W + E) * vec_U;
		result.source_down = result.source_up; 
	}

	return result;
}

RadiativeLayer addLayer(const RadiativeLayer& layer_bottom, const RadiativeLayer& layer_top, const geometry::Geometry& geometry, int n_parallel_fourier)
{
	int dim = layer_bottom.reflectance_m_top[0].rows();
	int n_stokes = dim / geometry.Ntheta;

	Eigen::VectorXd exp_t_small = Eigen::VectorXd::Zero(geometry.Ntheta);
	Eigen::VectorXd exp_b_small = Eigen::VectorXd::Zero(geometry.Ntheta);

	for(int i = 0; i < geometry.Ntheta; i++)
	{
		exp_t_small(i) = std::exp(-layer_top.optical_thickness / geometry.mu(i));
		exp_b_small(i) = std::exp(-layer_bottom.optical_thickness / geometry.mu(i));
	}
	
	Eigen::MatrixXd E_top = expandDiagonal(exp_t_small, n_stokes);
	Eigen::MatrixXd E_bottom = expandDiagonal(exp_b_small, n_stokes);
	Eigen::MatrixXd W = expandWMU(geometry.WMU, n_stokes);

	auto result = layer_bottom;
	result.optical_thickness = layer_bottom.optical_thickness + layer_top.optical_thickness;

	#pragma omp parallel for num_threads(n_parallel_fourier)
	for(int m = 0; m <= geometry.M; m++)
	{
		double factor = 2.0;
		Eigen::MatrixXd I_mat = Eigen::MatrixXd::Identity(dim, dim);

		Eigen::MatrixXd Q1 = factor * layer_top.reflectance_m_bottom[m] * W * layer_bottom.reflectance_m_top[m];
		Eigen::PartialPivLU<Eigen::MatrixXd> lu(I_mat - factor * Q1 * W);
		Eigen::MatrixXd S = lu.solve(Q1);

		Eigen::MatrixXd D = factor * S * W * layer_top.transmittance_m_top[m] + layer_top.transmittance_m_top[m] + S * E_top;
		Eigen::MatrixXd U = factor * layer_bottom.reflectance_m_top[m] * W * D + layer_bottom.reflectance_m_top[m] * E_top;

		result.reflectance_m_top[m] = factor * layer_top.transmittance_m_bottom[m] * W * U + layer_top.reflectance_m_top[m] + E_top * U;
		result.transmittance_m_top[m] = factor * layer_bottom.transmittance_m_top[m] * W * D + layer_bottom.transmittance_m_top[m] * E_top + E_bottom * D;

		result.reflectance_m_bottom[m].setZero();
		result.transmittance_m_bottom[m].setZero();
	}

	if (geometry.M >= 0)
	{
		int m = 0;
		double factor = 2.0;
		Eigen::MatrixXd I_mat = Eigen::MatrixXd::Identity(dim, dim);

		Eigen::MatrixXd Q1 = factor * layer_top.reflectance_m_bottom[m] * W * layer_bottom.reflectance_m_top[m];
		Eigen::PartialPivLU<Eigen::MatrixXd> lu(I_mat - factor * Q1 * W);

		Eigen::VectorXd vec_V = layer_top.source_down + factor * layer_top.reflectance_m_bottom[m] * W * layer_bottom.source_up;
		Eigen::VectorXd vec_D = lu.solve(vec_V);
		Eigen::VectorXd vec_U = layer_bottom.source_up + factor * layer_bottom.reflectance_m_top[m] * W * vec_D;
		
		result.source_up = layer_top.source_up + (factor * layer_top.transmittance_m_bottom[m] * W + E_top) * vec_U;
		result.source_down = layer_bottom.source_down + (factor * layer_bottom.transmittance_m_top[m] * W + E_bottom) * vec_D;
	}

	return result;
}

void computeInternalRadianceVector(const RadiativeLayer& layer_top, const RadiativeLayer& layer_bottom, const Eigen::VectorXd& I_minus_k, const geometry::Geometry& geo, int n_stokes, Eigen::VectorXd& I_minus_k_1, Eigen::VectorXd& I_plus_k_1)
{
	int dim = layer_top.reflectance_m_top[0].rows();
	Eigen::VectorXd exp_tau = Eigen::VectorXd::Zero(geo.Ntheta);

	for(int i = 0; i < geo.Ntheta; ++i)
	{
		exp_tau(i) = std::exp(-layer_top.optical_thickness / geo.mu(i));
	}
	
	Eigen::MatrixXd E_k = expandDiagonal(exp_tau, n_stokes);
	Eigen::MatrixXd W = expandWMU(geo.WMU, n_stokes);

	double factor = 2.0; 

	Eigen::MatrixXd Q1 = factor * layer_top.reflectance_m_bottom[0] * W * layer_bottom.reflectance_m_top[0];
	Eigen::MatrixXd I_mat = Eigen::MatrixXd::Identity(dim, dim);
	Eigen::PartialPivLU<Eigen::MatrixXd> lu(I_mat - factor * Q1 * W);

	Eigen::VectorXd src = (factor * layer_top.transmittance_m_top[0] * W + E_k) * I_minus_k + layer_top.source_down + factor * layer_top.reflectance_m_bottom[0] * W * layer_bottom.source_up;
	
	I_minus_k_1 = lu.solve(src);
	I_plus_k_1 = layer_bottom.source_up + factor * layer_bottom.reflectance_m_top[0] * W * I_minus_k_1;
}

void computeInternalRadianceMatrix(const RadiativeLayer& layer_top, const RadiativeLayer& layer_bottom, const Eigen::MatrixXd& I_minus_k, const geometry::Geometry& geo, int m, int n_stokes, Eigen::MatrixXd& I_minus_k_1, Eigen::MatrixXd& I_plus_k_1)
{
	int dim = layer_top.reflectance_m_top[0].rows();
	Eigen::VectorXd exp_tau = Eigen::VectorXd::Zero(geo.Ntheta);

	for(int i = 0; i < geo.Ntheta; ++i)
	{
		exp_tau(i) = std::exp(-layer_top.optical_thickness / geo.mu(i));
	}
	
	Eigen::MatrixXd E_k = expandDiagonal(exp_tau, n_stokes);
	Eigen::MatrixXd W = expandWMU(geo.WMU, n_stokes);

	double factor = 2.0; 

	Eigen::MatrixXd Q1 = factor * layer_top.reflectance_m_bottom[m] * W * layer_bottom.reflectance_m_top[m];
	Eigen::MatrixXd I_mat = Eigen::MatrixXd::Identity(dim, dim);
	Eigen::PartialPivLU<Eigen::MatrixXd> lu(I_mat - factor * Q1 * W);

	Eigen::MatrixXd src = (factor * layer_top.transmittance_m_top[m] * W + E_k) * I_minus_k;
	
	I_minus_k_1 = lu.solve(src);
	I_plus_k_1 = factor * layer_bottom.reflectance_m_top[m] * W * I_minus_k_1;
}

}
