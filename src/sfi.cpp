#include "sfi.hpp"
#include <cmath>
#include <numbers>
#include <algorithm>

namespace paad::sfi
{

SFIIntegrator::SFIIntegrator(const geometry::Geometry& geo, int n_stokes, int threads) : geo_(geo), n_stokes_(n_stokes), threads_(threads)
{
	;
}

Eigen::VectorXd SFIIntegrator::computeThermalEmission(const core::InternalField& field, const std::vector<core::OpticalLayer>& opt_layers, double mu_obs)
{
	int num_records = opt_layers.size();

	auto get_Z_thm = [&](double u_scat, double u_inc, const core::OpticalLayer& opt) -> Eigen::MatrixXd
	{
		Eigen::Matrix4d F = core::interpolateScatteringMatrix(opt.scattering_matrix, opt.scattering_angle, std::acos(std::clamp(u_scat * u_inc, -1.0, 1.0)));
		
		return F.block(0, 0, n_stokes_, n_stokes_);
	};

	auto compute_J_thm = [&](int level, const core::OpticalLayer& opt) -> Eigen::VectorXd
	{
		Eigen::VectorXd J = Eigen::VectorXd::Zero(n_stokes_);
		double omega = opt.single_scattering_albedo;
		J(0) += (1.0 - omega) * opt.planck_function;

		if (geo_.M >= 0)
		{
			for (int i = 0; i < geo_.Ntheta; ++i)
			{
				double mu_i = geo_.mu(i);
				double w_i = geo_.WMU(i, i);
				Eigen::VectorXd I_plus  = field.I_plus_thm[level].segment(n_stokes_ * i, n_stokes_);
				Eigen::VectorXd I_minus = field.I_minus_thm[level].segment(n_stokes_ * i, n_stokes_);
				Eigen::MatrixXd Z_up = get_Z_thm(mu_obs,  mu_i, opt);
				Eigen::MatrixXd Z_dn = get_Z_thm(mu_obs, -mu_i, opt);
				J += (omega / 2.0) * w_i * (Z_up * I_plus + Z_dn * I_minus); 
			}
		}

		return J;
	};

	Eigen::VectorXd I_obs = Eigen::VectorXd::Zero(n_stokes_);
	for(int s = 0; s < n_stokes_; ++s) I_obs(s) = field.I_plus_thm[0](s); 

	for (int i = 1; i < num_records; ++i)
	{
		double dtau = opt_layers[i].optical_thickness;
		Eigen::VectorXd J_bot = compute_J_thm(i - 1, opt_layers[i]);
		Eigen::VectorXd J_top = compute_J_thm(i, opt_layers[i]);
		
		double x = dtau / mu_obs;
		double alpha, beta;

		if (x < 1e-4)
		{
			alpha = x/2.0 - x*x/6.0;
			beta = x/2.0 - x*x/3.0;
		} 
		else
		{
			double exp_x = std::exp(-x);
			double frac = (1.0-exp_x)/x;
			alpha = 1.0-frac;
			beta = frac-exp_x;
		}
		
		I_obs = I_obs * std::exp(-x) + alpha * J_top + beta * J_bot;
	}

	return I_obs;
}

Eigen::MatrixXd SFIIntegrator::computeReflectanceMatrix(const core::InternalField& field, const std::vector<core::OpticalLayer>& opt_layers, double mu_obs, double mu_0, double dphi)
{
	int num_records = opt_layers.size();

	auto get_Z = [&](double u_scat, double u_inc, double rot_phi, const core::OpticalLayer& opt) -> Eigen::MatrixXd
	{
		double scattering_angle, rot1, rot2;
		geometry::computeScatteringGeometry(geo_, u_scat, u_inc, rot_phi, scattering_angle, rot1, rot2);
		Eigen::Matrix4d F = core::interpolateScatteringMatrix(opt.scattering_matrix, opt.scattering_angle, scattering_angle);

		return geometry::rotateMuellerMatrix(F, rot1, rot2).block(0, 0, n_stokes_, n_stokes_);
	};

	auto interpolate_field_matrix = [&](const Eigen::MatrixXd& I_mat_all, double mu_target) -> Eigen::MatrixXd
	{
		int idx = 0;

		while (idx < geo_.Ntheta - 1 && geo_.mu(idx + 1) < mu_target)
		{
			idx++;
		}

		double w = std::clamp((mu_target - geo_.mu(idx)) / (geo_.mu(idx + 1) - geo_.mu(idx)), 0.0, 1.0);
		Eigen::MatrixXd col1 = I_mat_all.block(0, n_stokes_ * idx, I_mat_all.rows(), n_stokes_);
		Eigen::MatrixXd col2 = I_mat_all.block(0, n_stokes_ * std::min(idx + 1, geo_.Ntheta - 1), I_mat_all.rows(), n_stokes_);

		return col1 + w * (col2 - col1);
	};

	auto compute_J_mat = [&](int level, const core::OpticalLayer& opt, double tau_cum) -> Eigen::MatrixXd
	{
		Eigen::MatrixXd J_mat = Eigen::MatrixXd::Zero(n_stokes_, n_stokes_);
		double omega = opt.single_scattering_albedo;
		
		Eigen::MatrixXd Z_ss = get_Z(mu_obs, -mu_0, dphi, opt);
		J_mat += (omega / (4.0 * std::numbers::pi)) * Z_ss * std::exp(-tau_cum / mu_0);

		double dphi_weight = 2.0 * std::numbers::pi / static_cast<double>(geo_.Nphi);

		for (int p = 0; p < geo_.Nphi; ++p)
		{
			double phi_p = geo_.phi[p]; 
			Eigen::MatrixXd I_plus_synth  = Eigen::MatrixXd::Zero(n_stokes_ * geo_.Ntheta, n_stokes_);
			Eigen::MatrixXd I_minus_synth = Eigen::MatrixXd::Zero(n_stokes_ * geo_.Ntheta, n_stokes_);
			
			for (int m = 0; m <= geo_.M; ++m)
			{
				double factor = (m == 0) ? 1.0 : 2.0;
				double c_m = std::cos(m * phi_p);
				double s_m = std::sin(m * phi_p);

				Eigen::MatrixXd I_plus_m  = interpolate_field_matrix(field.I_plus_sca[m][level], mu_0);
				Eigen::MatrixXd I_minus_m = interpolate_field_matrix(field.I_minus_sca[m][level], mu_0);

				for (int s = 0; s < I_plus_m.rows(); ++s)
				{
					double trig = ((s % n_stokes_) < 2) ? c_m : s_m;
					I_plus_synth.row(s)  += factor * trig * I_plus_m.row(s);
					I_minus_synth.row(s) += factor * trig * I_minus_m.row(s);
				}
			}
			
			for (int i = 0; i < geo_.Ntheta; ++i)
			{
				double mu_i = geo_.mu(i);
				double w_i = geo_.WMU(i, i);

				Eigen::MatrixXd Z_up = get_Z(mu_obs,  mu_i, dphi - phi_p, opt);
				Eigen::MatrixXd Z_dn = get_Z(mu_obs, -mu_i, dphi - phi_p, opt);

				J_mat += (omega / (4.0 * std::numbers::pi)) * w_i * dphi_weight * (Z_up * I_plus_synth.block(n_stokes_ * i, 0, n_stokes_, n_stokes_) + Z_dn * I_minus_synth.block(n_stokes_ * i, 0, n_stokes_, n_stokes_));
			}
		}

		return J_mat;
	};

	std::vector<double> tau_cum(num_records, 0.0);

	for (int i = num_records - 1; i >= 1; --i)
	{
		tau_cum[i - 1] = tau_cum[i] + opt_layers[i].optical_thickness;
	}

	Eigen::MatrixXd R_obs = Eigen::MatrixXd::Zero(n_stokes_, n_stokes_);

	if (opt_layers.size() > 0 && opt_layers[0].surface_albedo > 0.0)
	{
		R_obs(0, 0) = (opt_layers[0].surface_albedo / std::numbers::pi) * mu_0 * std::exp(-tau_cum[0] / mu_0);
	}

	for (int i = 1; i < num_records; ++i)
	{
		double dtau = opt_layers[i].optical_thickness;

		Eigen::MatrixXd J_bot = compute_J_mat(i - 1, opt_layers[i], tau_cum[i - 1]);
		Eigen::MatrixXd J_top = compute_J_mat(i, opt_layers[i], tau_cum[i]);
		
		double x = dtau / mu_obs;
		double alpha, beta;

		if (x < 1e-4)
		{
			alpha = x/2.0 - x*x/6.0;
			beta = x/2.0 - x*x/3.0;
		} 
		else
		{
			double exp_x = std::exp(-x);
			double frac = (1.0-exp_x)/x;
			alpha = 1.0-frac;
			beta = frac-exp_x;
		}
		
		R_obs = R_obs * std::exp(-x) + alpha * J_top + beta * J_bot;
	}
	return R_obs;
}

SFISensitivity SFIIntegrator::computeAdjoint(const core::InternalField& field, const std::vector<core::OpticalLayer>& opt_layers, const Eigen::VectorXd& grad_I_thm, const Eigen::MatrixXd& grad_R_mat, double mu_obs, double mu_0, double dphi)
{
	int num_records = opt_layers.size();
	int K = num_records - 1;
	int dim = n_stokes_ * geo_.Ntheta;

	SFISensitivity sens;
	sens.direct_gradients.resize(num_records);
	
	for (auto& sg : sens.direct_gradients)
	{
		sg.optical_thickness = 0.0;
		sg.single_scattering_albedo = 0.0;
		sg.planck_function = 0.0;
	}

	std::vector<double> tau_cum(num_records, 0.0);

	for (int i = K; i >= 1; --i)
	{
		tau_cum[i - 1] = tau_cum[i] + opt_layers[i].optical_thickness;
	}

	Eigen::VectorXd adj_I_thm = grad_I_thm;
	Eigen::MatrixXd adj_R_mat = grad_R_mat;

	std::vector<Eigen::VectorXd> adj_J_thm_top(num_records, Eigen::VectorXd::Zero(n_stokes_));
	std::vector<Eigen::VectorXd> adj_J_thm_bot(num_records, Eigen::VectorXd::Zero(n_stokes_));
	std::vector<Eigen::MatrixXd> adj_J_mat_top(num_records, Eigen::MatrixXd::Zero(n_stokes_, n_stokes_));
	std::vector<Eigen::MatrixXd> adj_J_mat_bot(num_records, Eigen::MatrixXd::Zero(n_stokes_, n_stokes_));

	for (int i = K; i >= 1; --i)
	{
		double dtau = opt_layers[i].optical_thickness;
		double x = dtau / mu_obs;
		double alpha, beta;

		if (x < 1e-4)
		{ 
			alpha = x/2.0 - x*x/6.0; 
			beta = x/2.0 - x*x/3.0; 
		}
		else
		{ 
			double exp_x = std::exp(-x); 
			double frac = (1.0-exp_x)/x; 
			alpha = 1.0-frac; 
			beta = frac-exp_x; 
		}

		adj_J_thm_top[i]     += alpha * adj_I_thm;
		adj_J_thm_bot[i - 1] += beta  * adj_I_thm;
		adj_I_thm = adj_I_thm * std::exp(-x);

		adj_J_mat_top[i]     += alpha * adj_R_mat;
		adj_J_mat_bot[i - 1] += beta  * adj_R_mat;
		adj_R_mat = adj_R_mat * std::exp(-x);
	}

	core::InternalField adj_field; 
	adj_field.I_plus_thm.assign(num_records, Eigen::VectorXd::Zero(dim));
	adj_field.I_minus_thm.assign(num_records, Eigen::VectorXd::Zero(dim));
	adj_field.I_plus_sca.assign(geo_.M + 1, std::vector<Eigen::MatrixXd>(num_records, Eigen::MatrixXd::Zero(dim, dim)));
	adj_field.I_minus_sca.assign(geo_.M + 1, std::vector<Eigen::MatrixXd>(num_records, Eigen::MatrixXd::Zero(dim, dim)));

	for(int s = 0; s < n_stokes_; ++s)
	{
		adj_field.I_plus_thm[0](s) += adj_I_thm(s);
	}

	auto get_Z_T = [&](double u_scat, double u_inc, double rot_phi, const core::OpticalLayer& opt) -> Eigen::MatrixXd
	{
		double scattering_angle, rot1, rot2;
		geometry::computeScatteringGeometry(geo_, u_scat, u_inc, rot_phi, scattering_angle, rot1, rot2);
		Eigen::Matrix4d F = core::interpolateScatteringMatrix(opt.scattering_matrix, opt.scattering_angle, scattering_angle);

		return geometry::rotateMuellerMatrix(F, rot1, rot2).block(0, 0, n_stokes_, n_stokes_).transpose();
	};

	double dphi_weight = 2.0 * std::numbers::pi / static_cast<double>(geo_.Nphi);

	for (int level = 0; level <= K; ++level)
	{
		double omega = (level == 0) ? 0.0 : opt_layers[level].single_scattering_albedo; 
		
		Eigen::VectorXd adj_J_thm = adj_J_thm_top[level] + adj_J_thm_bot[level];
		Eigen::MatrixXd adj_J_mat = adj_J_mat_top[level] + adj_J_mat_bot[level];

		if (level > 0)
		{
			sens.direct_gradients[level].planck_function += (1.0 - omega) * adj_J_thm(0);
			sens.direct_gradients[level].single_scattering_albedo -= opt_layers[level].planck_function * adj_J_thm(0);
		}

		int idx = 0;

		while (idx < geo_.Ntheta - 1 && geo_.mu(idx + 1) < mu_0)
		{
			idx++;
		}

		double w = std::clamp((mu_0 - geo_.mu(idx)) / (geo_.mu(idx + 1) - geo_.mu(idx)), 0.0, 1.0);

		for (int i = 0; i < geo_.Ntheta; ++i)
		{
			double mu_i = geo_.mu(i);
			double w_i = geo_.WMU(i, i);
			
			if (adj_J_thm.norm() > 1e-30)
			{
				Eigen::MatrixXd Z_up_T = get_Z_T(mu_obs,  mu_i, 0.0, opt_layers[level]); 
				Eigen::MatrixXd Z_dn_T = get_Z_T(mu_obs, -mu_i, 0.0, opt_layers[level]);

				adj_field.I_plus_thm[level].segment(n_stokes_ * i, n_stokes_)  += (omega / 2.0) * w_i * Z_up_T * adj_J_thm;
				adj_field.I_minus_thm[level].segment(n_stokes_ * i, n_stokes_) += (omega / 2.0) * w_i * Z_dn_T * adj_J_thm;
			}

			if (adj_J_mat.norm() > 1e-30)
			{
				for (int p = 0; p < geo_.Nphi; ++p)
				{
					double phi_p = geo_.phi[p];
					Eigen::MatrixXd Z_up_T = get_Z_T(mu_obs,  mu_i, dphi - phi_p, opt_layers[level]);
					Eigen::MatrixXd Z_dn_T = get_Z_T(mu_obs, -mu_i, dphi - phi_p, opt_layers[level]);
					
					Eigen::MatrixXd adj_I_plus_synth  = (omega / (4.0 * std::numbers::pi)) * w_i * dphi_weight * Z_up_T * adj_J_mat;
					Eigen::MatrixXd adj_I_minus_synth = (omega / (4.0 * std::numbers::pi)) * w_i * dphi_weight * Z_dn_T * adj_J_mat;

					for (int m = 0; m <= geo_.M; ++m)
					{
						double factor = (m == 0) ? 1.0 : 2.0;
						double c_m = std::cos(m * phi_p);
						double s_m = std::sin(m * phi_p);
						
						Eigen::MatrixXd adj_I_plus_m  = Eigen::MatrixXd::Zero(n_stokes_, n_stokes_);
						Eigen::MatrixXd adj_I_minus_m = Eigen::MatrixXd::Zero(n_stokes_, n_stokes_);

						for (int s = 0; s < n_stokes_; ++s)
						{
							double trig = (s < 2) ? c_m : s_m;
							adj_I_plus_m.row(s)  = factor * trig * adj_I_plus_synth.row(s);
							adj_I_minus_m.row(s) = factor * trig * adj_I_minus_synth.row(s);
						}

						adj_field.I_plus_sca[m][level].block(n_stokes_ * i, n_stokes_ * idx, n_stokes_, n_stokes_) += (1.0 - w) * adj_I_plus_m;
						adj_field.I_plus_sca[m][level].block(n_stokes_ * i, n_stokes_ * std::min(idx + 1, geo_.Ntheta - 1), n_stokes_, n_stokes_) += w * adj_I_plus_m;
						
						adj_field.I_minus_sca[m][level].block(n_stokes_ * i, n_stokes_ * idx, n_stokes_, n_stokes_) += (1.0 - w) * adj_I_minus_m;
						adj_field.I_minus_sca[m][level].block(n_stokes_ * i, n_stokes_ * std::min(idx + 1, geo_.Ntheta - 1), n_stokes_, n_stokes_) += w * adj_I_minus_m;
					}
				}
			}
		}
	}

	sens.adj_field = adj_field;
	return sens;
}

}
