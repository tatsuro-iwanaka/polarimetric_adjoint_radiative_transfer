#pragma once

#include <vector>
#include <numbers>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Dense>
#include "autodiff.hpp"
#include "math.hpp"

namespace mie
{

struct DiffFlags
{
	bool delta_r = false;
	bool lnd_r_g = false;
	bool lnd_sigma_g = false;
	bool rect_r_mean = false;
	bool rect_width = false;
	bool gd_a = false;
	bool gd_b = false;
	bool mgd_r_c = false;
	bool mgd_alpha = false;
	bool mgd_gamma = false;
	bool pld_delta = false;
	bool pld_r1 = false;
	bool pld_r2 = false;
	bool n_r = false;
	bool n_i = false;
};

template <typename T> class MieResult
{
	private:
		int _n_scattering_angle = 0;
		std::vector<double> _scattering_angle;
		
		T _scattering_cross_section;
		T _absorption_cross_section;
		T _extinction_cross_section;
		
		std::vector<Eigen::Matrix<T, 4, 4>> _scattering_matrix;

	public:
		int n_scattering_angle() const { return _n_scattering_angle; }
		double scattering_angle(int angle_idx) const { return _scattering_angle[angle_idx]; }

		T& scattering_cross_section() { return _scattering_cross_section; }
		T  scattering_cross_section() const { return _scattering_cross_section; }

		T& absorption_cross_section() { return _absorption_cross_section; }
		T  absorption_cross_section() const { return _absorption_cross_section; }

		T& extinction_cross_section() { return _extinction_cross_section; }
		T  extinction_cross_section() const { return _extinction_cross_section; }

		const Eigen::Matrix<T, 4, 4>& scattering_matrix(int angle_idx) const { return _scattering_matrix[angle_idx]; }
		Eigen::Matrix<T, 4, 4>& scattering_matrix(int angle_idx) { return _scattering_matrix[angle_idx]; }

		void reset(int n_ang)
		{
			_n_scattering_angle = n_ang;
			_scattering_angle.clear();
			_scattering_angle.resize(n_ang);

			_scattering_cross_section = T(0.0);
			_absorption_cross_section = T(0.0);
			_extinction_cross_section = T(0.0);

			double dtheta = std::numbers::pi / static_cast<double>(_n_scattering_angle - 1);

			for(int i = 0; i < _n_scattering_angle; ++i)
			{
				_scattering_angle[i] = dtheta * static_cast<double>(i);
			}

			_scattering_matrix.assign(n_ang, Eigen::Matrix<T, 4, 4>::Zero());
		}

		void normalizeScatteringMatrix(void)
		{
			using std::sin; using autodiff::sin;
			using std::abs; using autodiff::abs;

			const int n = _n_scattering_angle;
			if (n < 2) return;

			std::vector<std::vector<T>> integration_grid(n, std::vector<T>(2));

			for (int i = 0; i < n; ++i)
			{
				T theta = T(_scattering_angle[i]);
				T val = _scattering_matrix[i](0, 0) * T(2.0 * std::numbers::pi) * sin(theta);
				integration_grid[i] = {theta, val};
			}

			T integral = paad::math::computeSimpsonIntegration(integration_grid);
			T norm_factor = integral / (T(4.0) * T(std::numbers::pi));

			if (abs(norm_factor) > T(1.0E-12))
			{
				for (int i = 0; i < n; ++i)
				{
					_scattering_matrix[i] /= norm_factor;
				}
			}
		}
};

class MieJacobian
{
	private:
		int _n_parameter = 0;
		int _n_scattering_angle = 0;

		std::vector<double> _scattering_angle;

		std::vector<double> _scattering_cross_section;
		std::vector<double> _absorption_cross_section;
		std::vector<double> _extinction_cross_section;

		std::vector<Eigen::Matrix<double, 4, 4>> _scattering_matrix;

	public:
		int n_parameter() const { return _n_parameter; }
		int n_scattering_angle() const { return _n_scattering_angle; }
		double scattering_angle(int angle_idx) const { return _scattering_angle[angle_idx]; }

		double& scattering_cross_section(int param_idx) { return _scattering_cross_section[param_idx]; }
		double scattering_cross_section(int param_idx) const { return _scattering_cross_section[param_idx]; }

		double& absorption_cross_section(int param_idx) { return _absorption_cross_section[param_idx]; }
		double absorption_cross_section(int param_idx) const { return _absorption_cross_section[param_idx]; }

		double& extinction_cross_section(int param_idx) { return _extinction_cross_section[param_idx]; }
		double extinction_cross_section(int param_idx) const { return _extinction_cross_section[param_idx]; }

		const Eigen::Matrix<double, 4, 4>& scattering_matrix(int angle_idx, int param_idx) const { return _scattering_matrix[angle_idx * _n_parameter + param_idx]; }
		Eigen::Matrix<double, 4, 4>& scattering_matrix(int angle_idx, int param_idx) { return _scattering_matrix[angle_idx * _n_parameter + param_idx]; }

		void reset(int n_param, int n_ang)
		{
			_n_parameter = n_param;
			_n_scattering_angle = n_ang;
			_scattering_angle.clear();
			_scattering_angle.resize(n_ang);

			double dtheta = std::numbers::pi / static_cast<double>(_n_scattering_angle - 1);

			for(int i = 0; i < _n_scattering_angle; ++i)
			{
				_scattering_angle[i] = dtheta * static_cast<double>(i);
			}

			_scattering_cross_section.assign(n_param, 0.0);
			_absorption_cross_section.assign(n_param, 0.0);
			_extinction_cross_section.assign(n_param, 0.0);
			_scattering_matrix.assign(n_ang * n_param, Eigen::Matrix<double, 4, 4>::Zero());
		}
};

template <typename T>
inline double extract_value(const T& x)
{
	if constexpr (std::is_same_v<T, double>)
	{
		return x;
	}
	else
	{
		return x.val;
	}
}

template <typename T>
inline MieResult<T> computeMieScattering(int n_theta, T radius, T wavelength, autodiff::complex<T> index)
{
	MieResult<T> result;

	if (n_theta < 2)
	{
		n_theta = 2;
	}

	result.reset(n_theta);

	if (radius <= T(0.0) || wavelength <= T(0.0))
	{
		return result;
	}

	T diameter = T(2.0) * radius;
	T x = T(2.0 * std::numbers::pi) * radius / wavelength;

	double x_val = extract_value(x);
	int nstop = int(std::floor(x_val + 4.05 * std::cbrt(x_val) + 2.0));

	std::vector<autodiff::complex<T>> DD(nstop + 1);

	{
		T threshold = (T(13.78) * index.real() - T(10.8)) * index.real() + T(3.9);
		autodiff::complex<T> z = x * index;
		autodiff::complex<T> zinv(T(0.0), T(0.0));
	
		if (abs(index.imag() * x) >= threshold)
		{
			DD[0] = T(1.0) / tan(z);
			zinv  = T(1.0) / z;
			
			for (int i = 1; i <= nstop; ++i)
			{
				T k = T(double(i));
				auto numerator = k * zinv;
				DD[i] = T(1.0) / (numerator - DD[i - 1]) - numerator;
			}
		}
		else
		{
			zinv = T(2.0) / z;
			auto aj = -(T(nstop) + T(1.5)) * zinv;
			auto alpha_j1 = aj + T(1.0) / ((T(nstop) + T(0.5)) * zinv);
			auto alpha_j2 = aj;
			auto ratio = alpha_j1 / alpha_j2;
			auto runratio = ((T(nstop) + T(0.5)) * zinv) * ratio;

			while (abs(abs(ratio) - T(1.0)) > T(1e-12))
			{
				aj = zinv - aj;
				alpha_j1 = T(1.0) / alpha_j1 + aj;
				alpha_j2 = T(1.0) / alpha_j2 + aj;
				ratio = alpha_j1 / alpha_j2;
				runratio = runratio * ratio;
				zinv = -zinv;
			}

			DD[nstop] = -T(double(nstop)) / z + runratio;
			zinv = T(1.0) / z;

			for (int i = nstop - 1; i >= 0; --i)
			{
				T k = T(double(i + 1));
				auto num = k * zinv;
				DD[i] = num - T(1.0) / (DD[i + 1] + num);
			}
		}
	}

	std::vector<autodiff::complex<T>> a(nstop), b(nstop);
	T Qsca = T(0.0);
	T Qext = T(0.0);

	{
		T psi0 = sin(x);
		T psi1 = psi0 / x - cos(x);
	
		autodiff::complex<T> xi0(psi0, -cos(x));
		autodiff::complex<T> xi1(psi1, -(cos(x) / x + sin(x)));

		for (int i = 0; i < nstop; ++i)
		{
			T id = T(double(i + 1));
			a[i] = ((DD[i + 1] / index + id / x) * psi1 - psi0) / ((DD[i + 1] / index + id / x) * xi1 - xi0);
			b[i] = ((DD[i + 1] * index + id / x) * psi1 - psi0) / ((DD[i + 1] * index + id / x) * xi1 - xi0);
			
			T factor0 = T(2.0) * id + T(1.0);
			T norm_a = a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
			T norm_b = b[i].real() * b[i].real() + b[i].imag() * b[i].imag();
			
			Qsca = Qsca + factor0 * (norm_a + norm_b);
			Qext = Qext + factor0 * (a[i].real() + b[i].real());
			
			factor0 = (T(2.0) * id + T(1.0)) / x;
			autodiff::complex<T> xi = factor0 * xi1 - xi0;
			xi0 = xi1;
			xi1 = xi;
			
			T psi = factor0 * psi1 - psi0;
			psi0 = psi1;
			psi1 = xi1.real();
		}
	}

	Qsca = T(2.0) * Qsca / (x * x);
	Qext = T(2.0) * Qext / (x * x);
	T Qabs = Qext - Qsca;

	T area = T(std::numbers::pi) * radius * radius;
	result.scattering_cross_section() = Qsca * area;
	result.absorption_cross_section() = Qabs * area;
	result.extinction_cross_section() = Qext * area;

	T dtheta = T(std::numbers::pi) / T(double(n_theta - 1));
	T norm_factor = T(4.0) / (x * x * Qsca);

	for (int k = 0; k < n_theta; ++k)
	{
		T theta = dtheta * T(double(k));
		T mu = cos(theta);

		autodiff::complex<T> S1(T(0.0), T(0.0));
		autodiff::complex<T> S2(T(0.0), T(0.0));
		T pi0 = T(0.0);
		T pi1 = T(1.0);

		for (int i = 0; i < nstop; ++i)
		{
			T id = T(double(i + 1));
			T weight = (T(2.0) * id + T(1.0)) / (id * (id + T(1.0)));
			T tau = id * mu * pi1 - (id + T(1.0)) * pi0;

			S1 += weight * (a[i] * pi1 + b[i] * tau);
			S2 += weight * (b[i] * pi1 + a[i] * tau);

			T pi2 = ((T(2.0) * id + T(1.0)) * mu * pi1 - (id + T(1.0)) * pi0) / id;
			pi0 = pi1;
			pi1 = pi2;
		}

		T s1_sq = S1.real() * S1.real() + S1.imag() * S1.imag();
		T s2_sq = S2.real() * S2.real() + S2.imag() * S2.imag();
		T s2s1_re = S2.real() * S1.real() + S2.imag() * S1.imag();
		T s2s1_im = S2.imag() * S1.real() - S2.real() * S1.imag();

		Eigen::Matrix<T, 4, 4> F = Eigen::Matrix<T, 4, 4>::Zero();
		
		F(0, 0) = F(1, 1) = T(0.5) * (s2_sq + s1_sq) * norm_factor;
		F(0, 1) = F(1, 0) = T(0.5) * (s2_sq - s1_sq) * norm_factor;
		F(2, 2) = F(3, 3) = s2s1_re * norm_factor;
		F(2, 3) = s2s1_im * norm_factor;
		F(3, 2) = -F(2, 3);
		
		result.scattering_matrix(k) = F;
	}

	return result;
}

template <typename T> 
inline MieResult<T> computeMieScatteringSizeDistribution(int n_theta, T wavelength, const std::vector<std::vector<T>>& node_weight, autodiff::complex<T> index)
{
	MieResult<T> total_result;
	total_result.reset(n_theta);

	int n_r = node_weight.size();

	if (n_r == 0)
	{
		return total_result;
	}

	T max_radius = T(0.0);

	for (int i = 0; i < n_r; ++i)
	{
		if (node_weight[i][0] > max_radius) max_radius = node_weight[i][0];
	}

	double max_x_val = extract_value(T(2.0 * std::numbers::pi) * max_radius / wavelength);
	int max_nstop = int(std::floor(max_x_val + 4.05 * std::cbrt(max_x_val) + 2.0));

	std::vector<std::vector<double>> pi_array(n_theta, std::vector<double>(max_nstop));
	std::vector<std::vector<double>> tau_array(n_theta, std::vector<double>(max_nstop));
	double dtheta_d = std::numbers::pi / static_cast<double>(n_theta - 1);

	for (int k = 0; k < n_theta; ++k)
	{
		double theta = dtheta_d * double(k);
		double mu = std::cos(theta);
		double pi0 = 0.0;
		double pi1 = 1.0;

		for (int i = 0; i < max_nstop; ++i)
		{
			double id = double(i + 1);
			double tau = id * mu * pi1 - (id + 1.0) * pi0;
			pi_array[k][i] = pi1;
			tau_array[k][i] = tau;
			double pi2 = ((2.0 * id + 1.0) * mu * pi1 - (id + 1.0) * pi0) / id;
			pi0 = pi1;
			pi1 = pi2;
		}
	}

	std::vector<autodiff::complex<T>> DD(max_nstop + 1);
	std::vector<autodiff::complex<T>> a(max_nstop);
	std::vector<autodiff::complex<T>> b(max_nstop);

	T total_weight = T(0.0);
	T factor_sq = (wavelength * wavelength) / T(std::numbers::pi);

	for (int ir = 0; ir < n_r; ++ir)
	{
		T radius = node_weight[ir][0];
		T n_dr = node_weight[ir][1];
		total_weight += n_dr;

		T x = T(2.0 * std::numbers::pi) * radius / wavelength;
		double x_val = extract_value(x);
		int nstop = int(std::floor(x_val + 4.05 * std::cbrt(x_val) + 2.0));

		T threshold = (T(13.78) * index.real() - T(10.8)) * index.real() + T(3.9);
		autodiff::complex<T> z = x * index;
		autodiff::complex<T> zinv(T(0.0), T(0.0));
	
		if (abs(index.imag() * x) < threshold)
		{
			DD[0] = T(1.0) / tan(z);
			zinv  = T(1.0) / z;

			for (int i = 1; i <= nstop; ++i)
			{
				T k_val = T(double(i));
				auto numerator = k_val * zinv;
				DD[i] = T(1.0) / (numerator - DD[i - 1]) - numerator;
			}
		}
		else
		{
			zinv = T(2.0) / z;
			auto aj = -(T(nstop) + T(1.5)) * zinv;
			auto alpha_j1 = aj + T(1.0) / ((T(nstop) + T(0.5)) * zinv);
			auto alpha_j2 = aj;
			auto ratio = alpha_j1 / alpha_j2;
			auto runratio = ((T(nstop) + T(0.5)) * zinv) * ratio;

			while (abs(abs(ratio) - T(1.0)) > T(1e-12))
			{
				aj = zinv - aj;
				alpha_j1 = T(1.0) / alpha_j1 + aj;
				alpha_j2 = T(1.0) / alpha_j2 + aj;
				ratio = alpha_j1 / alpha_j2;
				runratio = runratio * ratio;
				zinv = -zinv;
			}

			DD[nstop] = -T(double(nstop)) / z + runratio;
			zinv = T(1.0) / z;

			for (int i = nstop - 1; i >= 0; --i)
			{
				T k_val = T(double(i + 1));
				auto num = k_val * zinv;
				DD[i] = num - T(1.0) / (DD[i + 1] + num);
			}
		}

		T Qsca = T(0.0), Qext = T(0.0);
		T psi0 = sin(x);
		T psi1 = psi0 / x - cos(x);
		autodiff::complex<T> xi0(psi0, -cos(x));
		autodiff::complex<T> xi1(psi1, -(cos(x) / x + sin(x)));

		for (int i = 0; i < nstop; ++i)
		{
			T id = T(double(i + 1));
			a[i] = ((DD[i + 1] / index + id / x) * psi1 - psi0) / ((DD[i + 1] / index + id / x) * xi1 - xi0);
			b[i] = ((DD[i + 1] * index + id / x) * psi1 - psi0) / ((DD[i + 1] * index + id / x) * xi1 - xi0);
			
			T factor0 = T(2.0) * id + T(1.0);
			T norm_a = a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
			T norm_b = b[i].real() * b[i].real() + b[i].imag() * b[i].imag();
			
			Qsca += factor0 * (norm_a + norm_b);
			Qext += factor0 * (a[i].real() + b[i].real());
			
			factor0 = (T(2.0) * id + T(1.0)) / x;
			autodiff::complex<T> xi = factor0 * xi1 - xi0;
			xi0 = xi1;
			xi1 = xi;
			
			T psi = factor0 * psi1 - psi0;
			psi0 = psi1;
			psi1 = xi1.real();
		}

		Qsca = T(2.0) * Qsca / (x * x);
		Qext = T(2.0) * Qext / (x * x);
		T Qabs = Qext - Qsca;
		T area = T(std::numbers::pi) * radius * radius;

		total_result.scattering_cross_section() += Qsca * area * n_dr;
		total_result.absorption_cross_section() += Qabs * area * n_dr;
		total_result.extinction_cross_section() += Qext * area * n_dr;

		T prefactor = factor_sq * n_dr;

		for (int k = 0; k < n_theta; ++k)
		{
			autodiff::complex<T> S1(T(0.0), T(0.0));
			autodiff::complex<T> S2(T(0.0), T(0.0));

			for (int i = 0; i < nstop; ++i)
			{
				T id = T(double(i + 1));
				T weight_term = (T(2.0) * id + T(1.0)) / (id * (id + T(1.0)));
				
				T pi_val = T(pi_array[k][i]);
				T tau_val = T(tau_array[k][i]);

				S1 += weight_term * (a[i] * pi_val + b[i] * tau_val);
				S2 += weight_term * (b[i] * pi_val + a[i] * tau_val);
			}

			T s1_sq = S1.real() * S1.real() + S1.imag() * S1.imag();
			T s2_sq = S2.real() * S2.real() + S2.imag() * S2.imag();
			T s2s1_re = S2.real() * S1.real() + S2.imag() * S1.imag();
			T s2s1_im = S2.imag() * S1.real() - S2.real() * S1.imag();

			total_result.scattering_matrix(k)(0, 0) += T(0.5) * (s2_sq + s1_sq) * prefactor;
			total_result.scattering_matrix(k)(1, 1) += T(0.5) * (s2_sq + s1_sq) * prefactor;
			total_result.scattering_matrix(k)(0, 1) += T(0.5) * (s2_sq - s1_sq) * prefactor;
			total_result.scattering_matrix(k)(1, 0) += T(0.5) * (s2_sq - s1_sq) * prefactor;
			total_result.scattering_matrix(k)(2, 2) += s2s1_re * prefactor;
			total_result.scattering_matrix(k)(3, 3) += s2s1_re * prefactor;
			total_result.scattering_matrix(k)(2, 3) += s2s1_im * prefactor;
			total_result.scattering_matrix(k)(3, 2) -= s2s1_im * prefactor;
		}
	}

	if (total_weight > T(0.0))
	{
		total_result.scattering_cross_section() /= total_weight;
		total_result.absorption_cross_section() /= total_weight;
		total_result.extinction_cross_section() /= total_weight;

		T norm = T(1.0) / (total_result.scattering_cross_section() * total_weight);

		for (int k = 0; k < n_theta; ++k)
		{
			total_result.scattering_matrix(k) *= norm;
		}
	}

	total_result.normalizeScatteringMatrix();
	return total_result;
}

template <typename T>
inline std::vector<std::vector<std::vector<T>>> generateRectangularSizeDistribution(int n_r, T r_mean, T width)
{
	std::vector<std::vector<T>> size_distribution(n_r, std::vector<T>(2));
	std::vector<std::vector<T>> weight(n_r, std::vector<T>(2));

	if (n_r <= 0 || width <= T(0.0))
	{
		throw std::runtime_error("[generateRectangularSizeDistribution] Invalid parameters.");
	}
	
	T r_min = r_mean - T(0.5) * width;
	T r_max = r_mean + T(0.5) * width;

	if (r_min < T(0.0))
	{
		throw std::runtime_error("[generateRectangularSizeDistribution] Rectangular distribution extends to negative radius.");
	}

	std::vector<std::vector<double>> node_weight_std = paad::math::computeGaussLegendreQuadratureNodeWeight(n_r);
	T n_r_val = T(1.0) / (r_max - r_min); 

	for (int i = 0; i < n_r; ++i)
	{
		T x_std = T(node_weight_std[i][0]);
		T w_std = T(node_weight_std[i][1]);
		T r = T(0.5) * (r_max - r_min) * x_std + T(0.5) * (r_max + r_min);

		size_distribution[i][0] = r;
		weight[i][0] = r;
		size_distribution[i][1] = n_r_val;
		weight[i][1] = w_std * T(0.5);
	}

	return {size_distribution, weight};
}

template <typename T> 
inline std::vector<std::vector<std::vector<T>>> generateLogNormalSizeDistribution(int n_r, T r_g, T sigma_g, T r_min, T r_max)
{
	using std::log; using autodiff::log;
	using std::exp; using autodiff::exp;

	if (n_r <= 0 || r_g <= T(0.0) || sigma_g <= T(1.0) || r_min <= T(0.0) || r_max <= r_min)
	{
		throw std::runtime_error("[generateLogNormalSizeDistribution] Invalid parameters.");
	}

	std::vector<std::vector<T>> size_distribution(n_r, std::vector<T>(2));
	std::vector<std::vector<T>> weight(n_r, std::vector<T>(2));

	T ln_r_min = log(r_min);
	T ln_r_max = log(r_max);
	std::vector<std::vector<double>> node_weight_std = paad::math::computeGaussLegendreQuadratureNodeWeight(n_r);
	
	T diff = ln_r_max - ln_r_min;
	T sum_limits = ln_r_max + ln_r_min;
	T jacobian = T(0.5) * diff;

	T ln_rg = log(r_g);
	T sigma_ln = log(sigma_g); 
	T pdf_norm_factor = T(1.0) / (T(std::numbers::sqrt2) * sigma_ln) * T(std::numbers::inv_sqrtpi);

	for (int i = 0; i < n_r; ++i)
	{
		T x_std = T(node_weight_std[i][0]);
		T w_std = T(node_weight_std[i][1]);

		T ln_r = T(0.5) * diff * x_std + T(0.5) * sum_limits;
		T r = exp(ln_r);
		T dr = r * w_std * jacobian; 

		T ln_ratio = (ln_r - ln_rg); 
		T exponent = -(ln_ratio * ln_ratio) / (T(2.0) * sigma_ln * sigma_ln);
		T n_r_val = (pdf_norm_factor / r) * exp(exponent);

		size_distribution[i][0] = r;
		weight[i][0] = r; 
		size_distribution[i][1] = n_r_val;
		weight[i][1] = n_r_val * dr;
	}

	return {size_distribution, weight};
}

template <typename T>
inline std::vector<std::vector<std::vector<T>>> generateGammaSizeDistribution(int n_r, T a, T b)
{
	using std::tgamma; using autodiff::tgamma;
	using std::pow; using autodiff::pow;
	using std::exp; using autodiff::exp;

	std::vector<std::vector<T>> size_distribution(n_r, std::vector<T>(2));
	std::vector<std::vector<T>> weight(n_r, std::vector<T>(2));

	if (n_r <= 0 || a <= T(0.0) || b <= T(0.0))
	{
		throw std::runtime_error("[generateGammaSizeDistribution] Invalid parameters.");
	}

	if ((T(1.0) - b) / b <= T(0.0))
	{
		throw std::runtime_error("[generateGammaSizeDistribution] Invalid b parameter.");
	}

	std::vector<std::vector<double>> node_weight_laguerre = paad::math::computeGaussLaguerreQuadratureNodeWeight(n_r);
	T norm_const = T(1.0) / (a * b * tgamma((T(1.0) - T(2.0) * b) / b));

	for (int i = 0; i < n_r; ++i)
	{
		T x_laguerre = T(node_weight_laguerre[i][0]);
		T w_laguerre = T(node_weight_laguerre[i][1]);
		T r = a * b * x_laguerre;
		if (r < T(1e-12)) r = T(1e-12);

		size_distribution[i][0] = r;
		weight[i][0] = r;
		size_distribution[i][1] = norm_const * pow(r / a / b, (T(1.0) - T(3.0) * b) / b) * exp(-r / a / b);
		weight[i][1] = T(1.0) / tgamma((T(1.0) - T(2.0) * b) / b) * w_laguerre * pow(x_laguerre, (T(1.0) - T(3.0) * b) / b);
	}

	return {size_distribution, weight};
}

template <typename T>
inline std::vector<std::vector<std::vector<T>>> generateModifiedGammaSizeDistribution(int n_r, T r_c, T alpha, T gamma)
{
	using std::tgamma; using autodiff::tgamma;
	using std::pow; using autodiff::pow;
	using std::exp; using autodiff::exp;

	std::vector<std::vector<T>> size_distribution(n_r, std::vector<T>(2));
	std::vector<std::vector<T>> weight(n_r, std::vector<T>(2));

	if (n_r <= 0 || r_c <= T(0.0) || alpha <= T(0.0) || gamma <= T(0.0))
	{
		throw std::runtime_error("[generateModifiedGammaSizeDistribution] Invalid parameters.");
	}
	
	std::vector<std::vector<double>> node_weight_laguerre = paad::math::computeGaussLaguerreQuadratureNodeWeight(n_r);
	T norm_const = gamma / (r_c * tgamma((alpha + T(1.0)) / gamma)) * pow(alpha / gamma, (alpha + T(1.0)) / gamma);
	
	T nu = (alpha - gamma + T(1.0)) / gamma;

	for (int i = 0; i < n_r; ++i)
	{
		T x_laguerre = T(node_weight_laguerre[i][0]);
		T w_laguerre = T(node_weight_laguerre[i][1]);
		T r = (x_laguerre == T(0.0)) ? T(0.0) : r_c * pow((gamma / alpha) * x_laguerre, T(1.0) / gamma);

		if (r < T(1e-12))
		{
			r = T(1e-12);
		}

		size_distribution[i][0] = r;
		weight[i][0] = r;
		size_distribution[i][1] = norm_const * pow(r / r_c, alpha) * exp(-alpha / gamma * pow(r / r_c, gamma));
		
		T adjustment = (x_laguerre == T(0.0)) ? T(0.0) : pow(x_laguerre, nu);
		weight[i][1] = w_laguerre * adjustment / tgamma((alpha + T(1.0)) / gamma);
	}
	return {size_distribution, weight};
}

template <typename T>
inline std::vector<std::vector<std::vector<T>>> generatePowerLawSizeDistribution(int n_r, T pl_delta, T pl_r1, T pl_r2)
{
	using std::log; using autodiff::log;
	using std::exp; using autodiff::exp;
	using std::abs; using autodiff::abs;
	using std::pow; using autodiff::pow;

	std::vector<std::vector<T>> size_distribution(n_r, std::vector<T>(2));
	std::vector<std::vector<T>> weight(n_r, std::vector<T>(2));

	if (n_r <= 0 || pl_r1 <= T(0.0) || pl_r2 <= T(0.0) || pl_r1 >= pl_r2)
	{
		throw std::runtime_error("[generatePowerLawSizeDistribution] Invalid limits.");
	}

	T ln_r_min = log(pl_r1);
	T ln_r_max = log(pl_r2);
	std::vector<std::vector<double>> node_weight_std = paad::math::computeGaussLegendreQuadratureNodeWeight(n_r);
	
	T diff = ln_r_max - ln_r_min;
	T sum_limits = ln_r_max + ln_r_min;
	T jacobian = T(0.5) * diff;
	
	T c;

	if (abs(pl_delta - T(1.0)) < T(1e-9))
	{
		c = T(1.0) / diff;
	}
	else
	{
		T term_nu = T(1.0) - pl_delta;
		c = term_nu / (pow(pl_r2, term_nu) - pow(pl_r1, term_nu));
	}

	for (int i = 0; i < n_r; ++i)
	{
		T x_std = T(node_weight_std[i][0]);
		T w_std = T(node_weight_std[i][1]);

		T ln_r = T(0.5) * diff * x_std + T(0.5) * sum_limits;
		T r = exp(ln_r);
		T n_r_val = c * pow(r, -pl_delta);
		T dr = r * w_std * jacobian;
		
		size_distribution[i][0] = r;
		weight[i][0] = r;
		size_distribution[i][1] = n_r_val;
		weight[i][1] = n_r_val * dr;
	}

	return {size_distribution, weight};
}

template <typename T>
inline MieResult<T> computeDeltaMieScattering(int n_theta, T r, T wavelength, autodiff::complex<T> index)
{
	auto result = computeMieScattering(n_theta, r, wavelength, index);
	result.normalizeScatteringMatrix();
	return result;
}

template <typename T>
inline MieResult<T> computeRectangularMieScattering(int n_theta, int n_r, T r_mean, T width, T wavelength, autodiff::complex<T> index)
{
	auto results = generateRectangularSizeDistribution(n_r, r_mean, width);
	return computeMieScatteringSizeDistribution(n_theta, wavelength, results[1], index);
}

template <typename T>
inline MieResult<T> computeLogNormalMieScattering(int n_theta, int n_r, T r_g, T sigma_g, T r_min, T r_max, T wavelength, autodiff::complex<T> index)
{
	auto results = generateLogNormalSizeDistribution(n_r, r_g, sigma_g, r_min, r_max);
	return computeMieScatteringSizeDistribution(n_theta, wavelength, results[1], index);
}

template <typename T>
inline MieResult<T> computeGammaMieScattering(int n_theta, int n_r, T a, T b, T wavelength, autodiff::complex<T> index)
{
	auto results = generateGammaSizeDistribution(n_r, a, b);
	return computeMieScatteringSizeDistribution(n_theta, wavelength, results[1], index);
}

template <typename T>
inline MieResult<T> computeModifiedGammaMieScattering(int n_theta, int n_r, T r_c, T alpha, T gamma, T wavelength, autodiff::complex<T> index)
{
	auto results = generateModifiedGammaSizeDistribution(n_r, r_c, alpha, gamma);
	return computeMieScatteringSizeDistribution(n_theta, wavelength, results[1], index);
}

template <typename T>
inline MieResult<T> computePowerLawMieScattering(int n_theta, int n_r, T pl_delta, T pl_r1, T pl_r2, T wavelength, autodiff::complex<T> index)
{
	auto results = generatePowerLawSizeDistribution(n_r, pl_delta, pl_r1, pl_r2);
	return computeMieScatteringSizeDistribution(n_theta, wavelength, results[1], index);
}

MieJacobian computeDeltaMieJacobian(int n_theta, double wavelength, double r, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags = {});
MieJacobian computeLogNormalMieJacobian(int n_theta, int n_radius, double wavelength, double r_g, double sigma_g, double r_min, double r_max, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags = {});
MieJacobian computeRectangularMieJacobian(int n_theta, int n_radius, double wavelength, double r_mean, double width, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags = {});
MieJacobian computeGammaMieJacobian(int n_theta, int n_radius, double wavelength, double a, double b, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags = {});
MieJacobian computeModifiedGammaMieJacobian(int n_theta, int n_radius, double wavelength, double r_c, double alpha, double gamma, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags = {});
MieJacobian computePowerLawMieJacobian(int n_theta, int n_radius, double wavelength, double pl_delta, double pl_r1, double pl_r2, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags = {});

}
