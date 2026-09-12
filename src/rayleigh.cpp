#include "rayleigh.hpp"

namespace rayleigh
{

RayleighJacobian computeRayleighJacobian(int n_theta, double wavelength, double refractive_index, double number_density_reference, double depolarization_factor, RayleighResult<double>& y, const DiffFlags& diff_flags)
{
	int n_x = (diff_flags.n_r ? 1 : 0) + (diff_flags.depolarization_factor ? 1 : 0);

	y.reset(n_theta);
	RayleighJacobian jacobian;
	jacobian.reset(n_x, n_theta);

	auto run_pass = [&](double nr_s, double delta_s, int col)
	{
		autodiff::dual<double> n_ad(refractive_index, nr_s);
		autodiff::dual<double> delta_ad(depolarization_factor, delta_s);

		auto res = computeRayleighScattering(n_theta, wavelength, n_ad, number_density_reference, delta_ad);

		if (col <= 0)
		{
			y.scattering_cross_section() = res.scattering_cross_section().val;
			y.extinction_cross_section() = res.extinction_cross_section().val;
			y.absorption_cross_section() = res.absorption_cross_section().val;

			for (int i = 0; i < n_theta; ++i)
			{
				y.scattering_matrix(i) = res.scattering_matrix(i).unaryExpr([](const auto& v) { return v.val; });
			}
		}

		if (col >= 0)
		{
			jacobian.scattering_cross_section(col) = res.scattering_cross_section().der;
			jacobian.extinction_cross_section(col) = res.extinction_cross_section().der;
			jacobian.absorption_cross_section(col) = res.absorption_cross_section().der;

			for (int i = 0; i < n_theta; ++i)
			{
				jacobian.scattering_matrix(i, col) = res.scattering_matrix(i).unaryExpr([](const auto& v) { return v.der; });
			}
		}
	};

	if (n_x == 0)
	{
		run_pass(0.0, 0.0, -1);
		return jacobian;
	}

	int c = 0;

	if (diff_flags.n_r)
	{
		run_pass(1.0, 0.0, c++);
	}

	if (diff_flags.depolarization_factor)
	{
		run_pass(0.0, 1.0, c++);
	}

	return jacobian;
}

}
