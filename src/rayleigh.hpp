#pragma once

#include <vector>
#include <complex>
#include <numbers>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

#include "autodiff.hpp"
#include "math.hpp"

namespace rayleigh
{

struct DiffFlags
{
	bool n_r = false;
	bool depolarization_factor = false;
};

template <typename T> class RayleighResult
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

			if (n < 2)
			{
				return;
			}

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

class RayleighJacobian
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

template <typename T> inline RayleighResult<T> computeRayleighScattering(int n_theta, double wavelength, T refractive_index, double number_density_reference, T depolarization_factor)
{
	RayleighResult<T> res;
	res.reset(n_theta);

	T n2_minus_1 = refractive_index * refractive_index - T(1.0);
	T delta = depolarization_factor;
	T king_factor = (T(6.0) + T(3.0) * delta) / (T(6.0) - T(7.0) * delta);
	
	double wl4 = std::pow(wavelength, 4.0);
	T sigma_s = (T(8.0) * std::pow(std::numbers::pi, 3.0) * n2_minus_1 * n2_minus_1) / (T(3.0) * wl4 * T(number_density_reference) * T(number_density_reference)) * king_factor;

	res.scattering_cross_section() = sigma_s;
	res.extinction_cross_section() = sigma_s;
	res.absorption_cross_section() = T(0.0);

	T gamma = delta / (T(2.0) - delta);

	for (int i = 0; i < n_theta; ++i)
	{
		double theta = res.scattering_angle(i);
		double cos_t = std::cos(theta);
		double cos2_t = cos_t * cos_t;

		Eigen::Matrix<T, 4, 4> F = Eigen::Matrix<T, 4, 4>::Zero();

		F(0, 0) = T(0.75) * ((T(1.0) + cos2_t) + gamma * (T(1.0) - cos2_t));
		F(0, 1) = F(1, 0) = -T(0.75) * (T(1.0) - gamma) * (T(1.0) - cos2_t);
		F(1, 1) = T(0.75) * ((T(1.0) + cos2_t) + gamma * (T(1.0) - cos2_t));
		F(2, 2) = T(1.5) * (T(1.0) - gamma) * cos_t;
		F(3, 3) = T(1.5) * (T(1.0) - T(3.0) * gamma) * cos_t;

		res.scattering_matrix(i) = F;
	}

	res.normalizeScatteringMatrix();

	return res;
}

RayleighJacobian computeRayleighJacobian(int n_theta, double wavelength, double refractive_index, double number_density_reference, double depolarization_factor, RayleighResult<double>& y, const DiffFlags& diff_flags = {});

}
