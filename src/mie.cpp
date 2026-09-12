#include "mie.hpp"

namespace mie
{

namespace
{
	void populateResultAndJacobian(int n_theta, int col, const MieResult<autodiff::dual<double>>& res, MieResult<double>& y, MieJacobian& jacobian)
	{
		if (col < 0)
		{
			y.reset(n_theta);
			y.scattering_cross_section() = res.scattering_cross_section().val;
			y.absorption_cross_section() = res.absorption_cross_section().val;
			y.extinction_cross_section() = res.extinction_cross_section().val;

			for (int i = 0; i < n_theta; ++i)
			{
				for (int r = 0; r < 4; ++r)
				{
					for (int c = 0; c < 4; ++c)
					{
						y.scattering_matrix(i)(r, c) = res.scattering_matrix(i)(r, c).val;
					}
				}
			}
		}
		else
		{
			jacobian.scattering_cross_section(col) = res.scattering_cross_section().der;
			jacobian.absorption_cross_section(col) = res.absorption_cross_section().der;
			jacobian.extinction_cross_section(col) = res.extinction_cross_section().der;

			for (int i = 0; i < n_theta; ++i)
			{
				for (int r = 0; r < 4; ++r)
				{
					for (int c = 0; c < 4; ++c)
					{
						jacobian.scattering_matrix(i, col)(r, c) = res.scattering_matrix(i)(r, c).der;
					}
				}
			}
		}
	}
}

MieJacobian computeDeltaMieJacobian(int n_theta, double wavelength, double r, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags)
{
	int n_x = (diff_flags.delta_r ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	MieJacobian jacobian;
	jacobian.reset(n_x, n_theta);

	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double r_seed, double nr_seed, double ni_seed, int current_col)
	{
		autodiff::dual<double> r_ad(r, r_seed);
		autodiff::dual<double> m_re_ad(index.real(), nr_seed);
		autodiff::dual<double> m_im_ad(index.imag(), ni_seed);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto res = computeMieScattering(n_theta, r_ad, wl_ad, index_ad);
		res.normalizeScatteringMatrix();

		populateResultAndJacobian(n_theta, current_col, res, y, jacobian);
	};

	run_pass(0.0, 0.0, 0.0, -1);

	if (n_x == 0)
	{
		return jacobian;
	}

	int col = 0;

	if (diff_flags.delta_r)
	{
		run_pass(1.0, 0.0, 0.0, col++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 1.0, 0.0, col++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 1.0, col++);
	}

	return jacobian;
}

MieJacobian computeLogNormalMieJacobian(int n_theta, int n_radius, double wavelength, double r_g, double sigma_g, double r_min, double r_max, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags)
{
	int n_x = (diff_flags.lnd_r_g ? 1 : 0) + (diff_flags.lnd_sigma_g ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	MieJacobian jacobian;
	jacobian.reset(n_x, n_theta);

	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double rg_s, double sg_s, double nr_s, double ni_s, int col)
	{
		autodiff::dual<double> rg_ad(r_g, rg_s);
		autodiff::dual<double> sig_ad(sigma_g, sg_s);
		autodiff::dual<double> rm_ad(r_min, 0.0);
		autodiff::dual<double> rx_ad(r_max, 0.0);
		autodiff::dual<double> m_re_ad(index.real(), nr_s);
		autodiff::dual<double> m_im_ad(index.imag(), ni_s);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto sd = generateLogNormalSizeDistribution(n_radius, rg_ad, sig_ad, rm_ad, rx_ad);
		auto res = computeMieScatteringSizeDistribution(n_theta, wl_ad, sd[1], index_ad);

		populateResultAndJacobian(n_theta, col, res, y, jacobian);
	};

	run_pass(0.0, 0.0, 0.0, 0.0, -1);

	if (n_x == 0)
	{
		return jacobian;
	}

	int c = 0;

	if (diff_flags.lnd_r_g)
	{
		run_pass(1.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.lnd_sigma_g)
	{
		run_pass(0.0, 1.0, 0.0, 0.0, c++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 0.0, 1.0, 0.0, c++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 0.0, 1.0, c++);
	}

	return jacobian;
}

MieJacobian computeRectangularMieJacobian(int n_theta, int n_radius, double wavelength, double r_mean, double width, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags)
{
	int n_x = (diff_flags.rect_r_mean ? 1 : 0) + (diff_flags.rect_width ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	MieJacobian jacobian;
	jacobian.reset(n_x, n_theta);
	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double rm_s, double rw_s, double nr_s, double ni_s, int col)
	{
		autodiff::dual<double> rm_ad(r_mean, rm_s);
		autodiff::dual<double> rw_ad(width, rw_s);
		autodiff::dual<double> m_re_ad(index.real(), nr_s);
		autodiff::dual<double> m_im_ad(index.imag(), ni_s);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto sd = generateRectangularSizeDistribution(n_radius, rm_ad, rw_ad);
		auto res = computeMieScatteringSizeDistribution(n_theta, wl_ad, sd[1], index_ad);

		populateResultAndJacobian(n_theta, col, res, y, jacobian);
	};

	run_pass(0.0, 0.0, 0.0, 0.0, -1);

	if (n_x == 0)
	{
		return jacobian;
	}

	int c = 0;

	if (diff_flags.rect_r_mean)
	{
		run_pass(1.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.rect_width)
	{
		run_pass(0.0, 1.0, 0.0, 0.0, c++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 0.0, 1.0, 0.0, c++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 0.0, 1.0, c++);
	}

	return jacobian;
}

MieJacobian computeGammaMieJacobian(int n_theta, int n_radius, double wavelength, double a, double b, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags)
{
	int n_x = (diff_flags.gd_a ? 1 : 0) + (diff_flags.gd_b ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	MieJacobian jacobian;
	jacobian.reset(n_x, n_theta);
	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double a_s, double b_s, double nr_s, double ni_s, int col)
	{
		autodiff::dual<double> a_ad(a, a_s);
		autodiff::dual<double> b_ad(b, b_s);
		autodiff::dual<double> m_re_ad(index.real(), nr_s);
		autodiff::dual<double> m_im_ad(index.imag(), ni_s);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto sd = generateGammaSizeDistribution(n_radius, a_ad, b_ad);
		auto res = computeMieScatteringSizeDistribution(n_theta, wl_ad, sd[1], index_ad);

		populateResultAndJacobian(n_theta, col, res, y, jacobian);
	};

	run_pass(0.0, 0.0, 0.0, 0.0, -1);

	if (n_x == 0)
	{
		return jacobian;
	}

	int c = 0;

	if (diff_flags.gd_a)
	{
		run_pass(1.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.gd_b)
	{
		run_pass(0.0, 1.0, 0.0, 0.0, c++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 0.0, 1.0, 0.0, c++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 0.0, 1.0, c++);
	}

	return jacobian;
}

MieJacobian computeModifiedGammaMieJacobian(int n_theta, int n_radius, double wavelength, double r_c, double alpha, double gamma, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags)
{
	int n_x = (diff_flags.mgd_r_c ? 1 : 0) + (diff_flags.mgd_alpha ? 1 : 0) + (diff_flags.mgd_gamma ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	MieJacobian jacobian;
	jacobian.reset(n_x, n_theta);
	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double rc_s, double al_s, double ga_s, double nr_s, double ni_s, int col)
	{
		autodiff::dual<double> rc_ad(r_c, rc_s);
		autodiff::dual<double> al_ad(alpha, al_s);
		autodiff::dual<double> ga_ad(gamma, ga_s);
		autodiff::dual<double> m_re_ad(index.real(), nr_s);
		autodiff::dual<double> m_im_ad(index.imag(), ni_s);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto sd = generateModifiedGammaSizeDistribution(n_radius, rc_ad, al_ad, ga_ad);
		auto res = computeMieScatteringSizeDistribution(n_theta, wl_ad, sd[1], index_ad);

		populateResultAndJacobian(n_theta, col, res, y, jacobian);
	};

	run_pass(0.0, 0.0, 0.0, 0.0, 0.0, -1);

	if (n_x == 0)
	{
		return jacobian;
	}
	
	int c = 0;

	if (diff_flags.mgd_r_c)
	{
		run_pass(1.0, 0.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.mgd_alpha)
	{
		run_pass(0.0, 1.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.mgd_gamma)
	{
		run_pass(0.0, 0.0, 1.0, 0.0, 0.0, c++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 0.0, 0.0, 1.0, 0.0, c++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 0.0, 0.0, 1.0, c++);
	}

	return jacobian;
}

MieJacobian computePowerLawMieJacobian(int n_theta, int n_radius, double wavelength, double pl_delta, double pl_r1, double pl_r2, autodiff::complex<double> index, MieResult<double>& y, const DiffFlags& diff_flags)
{
	int n_x = (diff_flags.pld_delta ? 1 : 0) + (diff_flags.pld_r1 ? 1 : 0) + (diff_flags.pld_r2 ? 1 : 0) + (diff_flags.n_r ? 1 : 0) + (diff_flags.n_i ? 1 : 0);

	MieJacobian jacobian;
	jacobian.reset(n_x, n_theta);
	autodiff::dual<double> wl_ad(wavelength, 0.0);

	auto run_pass = [&](double dl_s, double r1_s, double r2_s, double nr_s, double ni_s, int col)
	{
		autodiff::dual<double> dl_ad(pl_delta, dl_s);
		autodiff::dual<double> r1_ad(pl_r1, r1_s);
		autodiff::dual<double> r2_ad(pl_r2, r2_s);
		autodiff::dual<double> m_re_ad(index.real(), nr_s);
		autodiff::dual<double> m_im_ad(index.imag(), ni_s);
		autodiff::complex<autodiff::dual<double>> index_ad(m_re_ad, m_im_ad);

		auto sd = generatePowerLawSizeDistribution(n_radius, dl_ad, r1_ad, r2_ad);
		auto res = computeMieScatteringSizeDistribution(n_theta, wl_ad, sd[1], index_ad);

		populateResultAndJacobian(n_theta, col, res, y, jacobian);
	};

	run_pass(0.0, 0.0, 0.0, 0.0, 0.0, -1);

	if (n_x == 0)
	{
		return jacobian;
	}
	
	int c = 0;

	if (diff_flags.pld_delta)
	{
		run_pass(1.0, 0.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.pld_r1)
	{
		run_pass(0.0, 1.0, 0.0, 0.0, 0.0, c++);
	}

	if (diff_flags.pld_r2)
	{
		run_pass(0.0, 0.0, 1.0, 0.0, 0.0, c++);
	}

	if (diff_flags.n_r)
	{
		run_pass(0.0, 0.0, 0.0, 1.0, 0.0, c++);
	}

	if (diff_flags.n_i)
	{
		run_pass(0.0, 0.0, 0.0, 0.0, 1.0, c++);
	}

	return jacobian;
}

}
