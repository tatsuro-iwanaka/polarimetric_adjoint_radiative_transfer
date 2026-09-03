#include "adjoint.hpp"
#include "forward.hpp"

#include <cmath>
#include <algorithm>
#include <numbers>

namespace paad::core
{

namespace
{
	void applyDelta34(Eigen::MatrixXd& M, int n_stokes)
	{
		if (n_stokes < 3) return;

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
}

template <typename T>
void accumulate_gradient_to_grid(double theta_val, const T& grad_val, const std::vector<double>& theta_grid, std::vector<T>& grad_P_array)
{
	if (theta_val <= theta_grid.front())
	{
		grad_P_array[0] += grad_val;
		return;
	}

	if (theta_val >= theta_grid.back())
	{
		grad_P_array.back() += grad_val;
		return;
	}

	auto it = std::lower_bound(theta_grid.begin(), theta_grid.end(), theta_val);
	size_t idx = std::distance(theta_grid.begin(), it);
	size_t idx_prev = idx - 1;

	double w_next = (theta_val - theta_grid[idx_prev]) / (theta_grid[idx] - theta_grid[idx_prev]);
	double w_prev = 1.0 - w_next;

	grad_P_array[idx_prev] += grad_val * w_prev;
	grad_P_array[idx] += grad_val * w_next;
}

RadiativeLayer doubleLayer_adjoint(const RadiativeLayer& layer, const geometry::Geometry& geometry, const RadiativeLayer& adj_result)
{
	RadiativeLayer adj_layer = layer;
	adj_layer.optical_thickness = 0.0;
	adj_layer.source_up.setZero();
	adj_layer.source_down.setZero();

	int dim = layer.reflectance_m_top[0].rows();
	int n_stokes = dim / geometry.Ntheta;

	for(int m = 0; m <= geometry.M; m++)
	{
		adj_layer.reflectance_m_top[m].setZero();
		adj_layer.reflectance_m_bottom[m].setZero();
		adj_layer.transmittance_m_top[m].setZero();
		adj_layer.transmittance_m_bottom[m].setZero();
	}

	Eigen::VectorXd exp_tau_small = Eigen::VectorXd::Zero(geometry.Ntheta);
	Eigen::VectorXd adj_exp_tau_diag = Eigen::VectorXd::Zero(dim);

	for(int i = 0; i < geometry.Ntheta; i++)
	{
		exp_tau_small(i) = std::exp(-layer.optical_thickness / geometry.mu(i));
	}

	Eigen::MatrixXd E = expandDiagonal(exp_tau_small, n_stokes);
	Eigen::MatrixXd W = expandWMU(geometry.WMU, n_stokes);

	adj_layer.optical_thickness += adj_result.optical_thickness * 2.0;

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

		Eigen::VectorXd vec_V, vec_D, vec_U;

		if(m == 0)
		{
			vec_V = layer.source_down + factor * layer.reflectance_m_bottom[m] * W * layer.source_up;
			vec_D = lu.solve(vec_V);
			vec_U = layer.source_up + factor * layer.reflectance_m_top[m] * W * vec_D;
		}

		Eigen::MatrixXd adj_R_bot = adj_result.reflectance_m_bottom[m];
		Eigen::MatrixXd adj_T_bot = adj_result.transmittance_m_bottom[m];
		
		if (m > 0)
		{
			applyDelta34(adj_R_bot, n_stokes);
			applyDelta34(adj_T_bot, n_stokes);
		}
		
		Eigen::MatrixXd adj_ref_top = adj_result.reflectance_m_top[m] + adj_R_bot;
		Eigen::MatrixXd adj_trans_top = adj_result.transmittance_m_top[m] + adj_T_bot;
		Eigen::MatrixXd adj_S = Eigen::MatrixXd::Zero(dim, dim);
		Eigen::MatrixXd adj_Q2_therm = Eigen::MatrixXd::Zero(dim, dim);

		if(m == 0)
		{
			Eigen::VectorXd adj_J_new = adj_result.source_up + adj_result.source_down;
			adj_layer.source_up += adj_J_new;

			Eigen::VectorXd adj_U_vec = (factor * layer.transmittance_m_bottom[m] * W + E).transpose() * adj_J_new;
			adj_layer.transmittance_m_bottom[m] += factor * adj_J_new * (W * vec_U).transpose();
			adj_exp_tau_diag += adj_J_new.cwiseProduct(vec_U);

			adj_layer.source_up += adj_U_vec;
			Eigen::VectorXd adj_D_vec = factor * W.transpose() * layer.reflectance_m_top[m].transpose() * adj_U_vec;
			adj_layer.reflectance_m_top[m] += factor * adj_U_vec * (W * vec_D).transpose();

			Eigen::VectorXd adj_V = lu.transpose().solve(adj_D_vec);
			adj_Q2_therm += adj_V * vec_D.transpose();

			adj_layer.source_down += adj_V;
			adj_layer.source_up += factor * W.transpose() * layer.reflectance_m_bottom[m].transpose() * adj_V;
			adj_layer.reflectance_m_bottom[m] += factor * adj_V * (W * layer.source_up).transpose();
		}

		Eigen::MatrixXd adj_T_res = adj_trans_top;
		adj_layer.transmittance_m_top[m] += adj_T_res * E.transpose();
		adj_exp_tau_diag += (layer.transmittance_m_top[m].transpose() * adj_T_res).diagonal();
		
		Eigen::MatrixXd adj_D = E.transpose() * adj_T_res;
		adj_exp_tau_diag += (adj_T_res * D.transpose()).diagonal();
		adj_layer.transmittance_m_top[m] += factor * adj_T_res * (W * D).transpose();
		adj_D += factor * (layer.transmittance_m_top[m] * W).transpose() * adj_T_res;

		Eigen::MatrixXd adj_R_res = adj_ref_top;
		adj_layer.reflectance_m_top[m] += adj_R_res;
		Eigen::MatrixXd adj_U = E.transpose() * adj_R_res;
		adj_exp_tau_diag += (adj_R_res * U.transpose()).diagonal();
		adj_layer.transmittance_m_bottom[m] += factor * adj_R_res * (W * U).transpose();
		adj_U += factor * (layer.transmittance_m_bottom[m] * W).transpose() * adj_R_res;

		adj_layer.reflectance_m_top[m] += factor * adj_U * (W * D).transpose();
		adj_D += factor * (layer.reflectance_m_top[m] * W).transpose() * adj_U;
		adj_layer.reflectance_m_top[m] += adj_U * E.transpose();
		adj_exp_tau_diag += (layer.reflectance_m_top[m].transpose() * adj_U).diagonal();

		adj_layer.transmittance_m_top[m] += adj_D;
		adj_S += adj_D * E.transpose();
		adj_exp_tau_diag += (S.transpose() * adj_D).diagonal();
		adj_S += factor * adj_D * (W * layer.transmittance_m_top[m]).transpose();
		adj_layer.transmittance_m_top[m] += factor * (S * W).transpose() * adj_D;

		Eigen::MatrixXd adj_Q1_base = lu.transpose().solve(adj_S);
		Eigen::MatrixXd adj_Q2 = adj_Q1_base * S.transpose();
		Eigen::MatrixXd adj_Q1 = adj_Q1_base;
		
		adj_Q2 += adj_Q2_therm;
		adj_Q1 += factor * adj_Q2 * W.transpose();

		adj_layer.reflectance_m_bottom[m] += factor * adj_Q1 * (W * layer.reflectance_m_top[m]).transpose();
		adj_layer.reflectance_m_top[m] += factor * (layer.reflectance_m_bottom[m] * W).transpose() * adj_Q1;
	}

	for(int i = 0; i < geometry.Ntheta; i++)
	{
		double sum_adj = adj_exp_tau_diag.segment(n_stokes * i, n_stokes).sum();
		adj_layer.optical_thickness += sum_adj * (-1.0 / geometry.mu(i)) * exp_tau_small(i);
	}

	return adj_layer;
}

std::vector<RadiativeLayer> addLayer_adjoint(const RadiativeLayer& layer_bottom, const RadiativeLayer& layer_top, const geometry::Geometry& geometry, const RadiativeLayer& adj_result)
{
	RadiativeLayer adj_bot = layer_bottom;
	RadiativeLayer adj_top = layer_top;

	adj_bot.optical_thickness = 0.0;
	adj_top.optical_thickness = 0.0;
	adj_bot.source_up.setZero(); adj_bot.source_down.setZero();
	adj_top.source_up.setZero(); adj_top.source_down.setZero();

	int dim = layer_bottom.reflectance_m_top[0].rows();
	int n_stokes = dim / geometry.Ntheta;

	for(int m = 0; m <= geometry.M; m++)
	{
		adj_bot.reflectance_m_top[m].setZero();
		adj_bot.reflectance_m_bottom[m].setZero();
		adj_bot.transmittance_m_top[m].setZero();
		adj_bot.transmittance_m_bottom[m].setZero();

		adj_top.reflectance_m_top[m].setZero();
		adj_top.reflectance_m_bottom[m].setZero();
		adj_top.transmittance_m_top[m].setZero();
		adj_top.transmittance_m_bottom[m].setZero();
	}

	Eigen::VectorXd exp_tau_top_small = Eigen::VectorXd::Zero(geometry.Ntheta);
	Eigen::VectorXd exp_tau_bot_small = Eigen::VectorXd::Zero(geometry.Ntheta);
	Eigen::VectorXd adj_exp_tau_top_diag = Eigen::VectorXd::Zero(dim);
	Eigen::VectorXd adj_exp_tau_bot_diag = Eigen::VectorXd::Zero(dim);

	for(int i = 0; i < geometry.Ntheta; i++)
	{
		exp_tau_top_small(i) = std::exp(-layer_top.optical_thickness / geometry.mu(i));
		exp_tau_bot_small(i) = std::exp(-layer_bottom.optical_thickness / geometry.mu(i));
	}

	Eigen::MatrixXd E_top = expandDiagonal(exp_tau_top_small, n_stokes);
	Eigen::MatrixXd E_bottom = expandDiagonal(exp_tau_bot_small, n_stokes);
	Eigen::MatrixXd W = expandWMU(geometry.WMU, n_stokes);

	adj_bot.optical_thickness += adj_result.optical_thickness;
	adj_top.optical_thickness += adj_result.optical_thickness;

	for(int m = 0; m <= geometry.M; m++)
	{
		double factor = 2.0;

		Eigen::MatrixXd Q1 = factor * layer_top.reflectance_m_bottom[m] * W * layer_bottom.reflectance_m_top[m];
		Eigen::MatrixXd I_mat = Eigen::MatrixXd::Identity(dim, dim);
		Eigen::PartialPivLU<Eigen::MatrixXd> lu(I_mat - factor * Q1 * W);
		Eigen::MatrixXd S = lu.solve(Q1);
		Eigen::MatrixXd Sexp_t = S * E_top;
		Eigen::MatrixXd D = factor * S * W * layer_top.transmittance_m_top[m] + layer_top.transmittance_m_top[m] + Sexp_t;
		Eigen::MatrixXd U = factor * layer_bottom.reflectance_m_top[m] * W * D + layer_bottom.reflectance_m_top[m] * E_top;

		Eigen::VectorXd vec_V, vec_D, vec_U;

		if(m == 0)
		{
			vec_V = layer_top.source_down + factor * layer_top.reflectance_m_bottom[m] * W * layer_bottom.source_up;
			vec_D = lu.solve(vec_V);
			vec_U = layer_bottom.source_up + factor * layer_bottom.reflectance_m_top[m] * W * vec_D;
		}

		Eigen::MatrixXd adj_S = Eigen::MatrixXd::Zero(dim, dim);
		Eigen::MatrixXd adj_Q1_therm = Eigen::MatrixXd::Zero(dim, dim);

		if(m == 0)
		{
			Eigen::VectorXd adj_J1_new_up = adj_result.source_up;
			Eigen::VectorXd adj_J2_new_dn = adj_result.source_down;

			adj_bot.source_down += adj_J2_new_dn;
			Eigen::VectorXd adj_D_vec = (factor * layer_bottom.transmittance_m_top[m] * W + E_bottom).transpose() * adj_J2_new_dn;
			adj_bot.transmittance_m_top[m] += factor * adj_J2_new_dn * (W * vec_D).transpose();
			adj_exp_tau_bot_diag += adj_J2_new_dn.cwiseProduct(vec_D);

			adj_top.source_up += adj_J1_new_up;
			Eigen::VectorXd adj_U_vec = (factor * layer_top.transmittance_m_bottom[m] * W + E_top).transpose() * adj_J1_new_up;
			adj_top.transmittance_m_bottom[m] += factor * adj_J1_new_up * (W * vec_U).transpose();
			adj_exp_tau_top_diag += adj_J1_new_up.cwiseProduct(vec_U);

			adj_bot.source_up += adj_U_vec;
			adj_D_vec += factor * W.transpose() * layer_bottom.reflectance_m_top[m].transpose() * adj_U_vec;
			adj_bot.reflectance_m_top[m] += factor * adj_U_vec * (W * vec_D).transpose();

			Eigen::VectorXd adj_V = lu.transpose().solve(adj_D_vec);
			adj_Q1_therm += factor * adj_V * (W * vec_D).transpose();

			adj_top.source_down += adj_V;
			adj_bot.source_up += factor * W.transpose() * layer_top.reflectance_m_bottom[m].transpose() * adj_V;
			adj_top.reflectance_m_bottom[m] += factor * adj_V * (W * layer_bottom.source_up).transpose();
		}

		Eigen::MatrixXd adj_T_res = adj_result.transmittance_m_top[m];
		adj_exp_tau_bot_diag += (adj_T_res * D.transpose()).diagonal();
		Eigen::MatrixXd adj_D = E_bottom.transpose() * adj_T_res;
		adj_bot.transmittance_m_top[m] += adj_T_res * E_top.transpose();
		adj_exp_tau_top_diag += (layer_bottom.transmittance_m_top[m].transpose() * adj_T_res).diagonal();
		adj_bot.transmittance_m_top[m] += factor * adj_T_res * (W * D).transpose();
		adj_D += factor * (layer_bottom.transmittance_m_top[m] * W).transpose() * adj_T_res;

		Eigen::MatrixXd adj_R_res = adj_result.reflectance_m_top[m];
		adj_exp_tau_top_diag += (adj_R_res * U.transpose()).diagonal();
		Eigen::MatrixXd adj_U = E_top.transpose() * adj_R_res;
		adj_top.reflectance_m_top[m] += adj_R_res;
		adj_top.transmittance_m_bottom[m] += factor * adj_R_res * (W * U).transpose();
		adj_U += factor * (layer_top.transmittance_m_bottom[m] * W).transpose() * adj_R_res;

		adj_bot.reflectance_m_top[m] += adj_U * E_top.transpose();
		adj_exp_tau_top_diag += (layer_bottom.reflectance_m_top[m].transpose() * adj_U).diagonal();
		adj_bot.reflectance_m_top[m] += factor * adj_U * (W * D).transpose();
		adj_D += factor * (layer_bottom.reflectance_m_top[m] * W).transpose() * adj_U;

		adj_S += adj_D * E_top.transpose();
		adj_exp_tau_top_diag += (S.transpose() * adj_D).diagonal();
		adj_top.transmittance_m_top[m] += adj_D;
		adj_S += factor * adj_D * (W * layer_top.transmittance_m_top[m]).transpose();
		adj_top.transmittance_m_top[m] += factor * (S * W).transpose() * adj_D;

		Eigen::MatrixXd adj_Q1_base = lu.transpose().solve(adj_S);
		Eigen::MatrixXd adj_Q2 = adj_Q1_base * S.transpose();
		Eigen::MatrixXd adj_Q1 = adj_Q1_base + factor * adj_Q2 * W.transpose();
		
		adj_Q1 += adj_Q1_therm;

		adj_top.reflectance_m_bottom[m] += factor * adj_Q1 * (W * layer_bottom.reflectance_m_top[m]).transpose();
		adj_bot.reflectance_m_top[m] += factor * (layer_top.reflectance_m_bottom[m] * W).transpose() * adj_Q1;
	}

	for(int i = 0; i < geometry.Ntheta; i++)
	{
		double sum_adj_top = adj_exp_tau_top_diag.segment(n_stokes * i, n_stokes).sum();
		double sum_adj_bot = adj_exp_tau_bot_diag.segment(n_stokes * i, n_stokes).sum();
		
		adj_top.optical_thickness += sum_adj_top * (-1.0 / geometry.mu(i)) * exp_tau_top_small(i);
		adj_bot.optical_thickness += sum_adj_bot * (-1.0 / geometry.mu(i)) * exp_tau_bot_small(i);
	}

	return {adj_bot, adj_top};
}

OpticalSensitivity computeInitializationSensitivities(const RadiativeLayer& adj_layer, const RadiativeLayer& fwd_matrix_layer, double single_scattering_albedo, double planck_function, const geometry::Geometry& geo, int n_theta)
{
	OpticalSensitivity result;

	result.optical_thickness = 0.0;
	result.single_scattering_albedo = 0.0;
	result.planck_function = 0.0;

	double tau = fwd_matrix_layer.optical_thickness;
	double omega = single_scattering_albedo;

	double inv_tau = (tau > 1e-30) ? 1.0 / tau : 0.0;
	double inv_omega = (omega > 1e-30) ? 1.0 / omega : 0.0;
	double base_coeff = omega * tau / 4.0;

	int dim = adj_layer.reflectance_m_top[0].rows();
	int n_stokes = dim / geo.Ntheta;

	result.optical_thickness += adj_layer.optical_thickness;

	double sum_inner_product = 0.0;

	for (int m = 0; m <= geo.M; ++m)
	{
		sum_inner_product += (adj_layer.reflectance_m_top[m].array() * fwd_matrix_layer.reflectance_m_top[m].array()).sum();
		sum_inner_product += (adj_layer.reflectance_m_bottom[m].array() * fwd_matrix_layer.reflectance_m_bottom[m].array()).sum();
		sum_inner_product += (adj_layer.transmittance_m_top[m].array() * fwd_matrix_layer.transmittance_m_top[m].array()).sum();
		sum_inner_product += (adj_layer.transmittance_m_bottom[m].array() * fwd_matrix_layer.transmittance_m_bottom[m].array()).sum();
	}

	result.optical_thickness += sum_inner_product * inv_tau;

	if (omega >= 1.0E-16)
	{
		result.single_scattering_albedo += sum_inner_product * inv_omega;
	}

	for(int i = 0; i < geo.Ntheta; ++i)
	{
		double mu = geo.mu(i);
		double trans = std::exp(-tau / mu);
		double emit_factor = -std::expm1(-tau / mu); 

		double adj_sum_I = adj_layer.source_up(n_stokes * i) + adj_layer.source_down(n_stokes * i);

		result.optical_thickness += adj_sum_I * (1.0 - omega) * planck_function * trans / mu;
		result.single_scattering_albedo -= adj_sum_I * planck_function * emit_factor;
		result.planck_function += adj_sum_I * (1.0 - omega) * emit_factor;
	}

	std::vector<double> theta_grid(n_theta);
	std::vector<Eigen::Matrix4d> grad_P(n_theta, Eigen::Matrix4d::Zero());
	
	for(int i = 0; i < n_theta; ++i)
	{
		theta_grid[i] = std::numbers::pi / double(n_theta - 1) * double(i);
	}

	auto get_scat_geom = [&](double u_scat, double u_inc, double dphi, double& sca_ang, double& r1, double& r2)
	{
		const double EPS = 1e-12;

		if (std::abs(std::abs(u_scat) - 1.0) < EPS && std::abs(std::abs(u_inc) - 1.0) < EPS)
		{
			sca_ang = (u_scat * u_inc > 0) ? 0.0 : std::numbers::pi;
			r1 = 0.0;
			r2 = (u_scat * u_inc > 0) ? dphi : std::numbers::pi - dphi; 
		}
		else if (std::abs(std::abs(u_scat) - 1.0) < EPS)
		{
			sca_ang = std::acos(std::clamp(u_scat * u_inc, -1.0, 1.0));
			r1 = 0.0;
			r2 = (u_scat > 0) ? -dphi : dphi; 
		}
		else if (std::abs(std::abs(u_inc) - 1.0) < EPS)
		{
			sca_ang = std::acos(std::clamp(u_scat * u_inc, -1.0, 1.0));
			r1 = (u_inc > 0) ? dphi : -dphi;
			r2 = 0.0;
		}
		else
		{
			geometry::computeScatteringGeometry(geo, u_scat, u_inc, dphi, sca_ang, r1, r2);
		}
	};

	for (int i = 0; i < geo.Ntheta; ++i) 
	{
		for (int j = 0; j < geo.Ntheta; ++j) 
		{
			double mu_i = geo.mu(i);
			double mu_j = geo.mu(j);
			double geom_factor = base_coeff / (mu_i * mu_j);

			for (int k = 0; k < geo.Nphi; ++k) 
			{
				double phi = geo.phi(k);

				Eigen::Matrix4d grad_R_val = Eigen::Matrix4d::Zero();
				Eigen::Matrix4d grad_T_val = Eigen::Matrix4d::Zero();

				for (int m = 0; m <= geo.M; ++m) 
				{
					double cos_m_phi = std::cos(m * phi);
					double sin_m_phi = std::sin(m * phi);
					
					double factor = 1.0;

					Eigen::Matrix4d dR = Eigen::Matrix4d::Zero();
					Eigen::Matrix4d dT = Eigen::Matrix4d::Zero();

					for (int r = 0; r < n_stokes; ++r) 
					{
						for (int c = 0; c < n_stokes; ++c) 
						{
							bool r_is_cos = (r < 2);
							bool c_is_cos = (c < 2);

							double val_R_bot = adj_layer.reflectance_m_bottom[m](n_stokes * i + r, n_stokes * j + c);
							double val_T_bot = adj_layer.transmittance_m_bottom[m](n_stokes * i + r, n_stokes * j + c);

							bool r_neg = ((r % n_stokes) == 2 || (r % n_stokes) == 3);
							bool c_neg = ((c % n_stokes) == 2 || (c % n_stokes) == 3);

							if (r_neg != c_neg)
							{
								val_R_bot = -val_R_bot;
								val_T_bot = -val_T_bot;
							}

							double val_R = adj_layer.reflectance_m_top[m](n_stokes * i + r, n_stokes * j + c) + val_R_bot;
							double val_T = adj_layer.transmittance_m_top[m](n_stokes * i + r, n_stokes * j + c) + val_T_bot;

							if (r_is_cos == c_is_cos)
							{
								dR(r, c) = val_R * cos_m_phi;
								dT(r, c) = val_T * cos_m_phi;
							}
							else if (r_is_cos && !c_is_cos)
							{
								dR(r, c) = -val_R * sin_m_phi;
								dT(r, c) = -val_T * sin_m_phi;
							}
							else
							{
								dR(r, c) = val_R * sin_m_phi;
								dT(r, c) = val_T * sin_m_phi;
							}
						}
					}

					grad_R_val += factor * dR;
					grad_T_val += factor * dT;
				}

				double theta_refl, rot1_refl, rot2_refl;
				get_scat_geom(-mu_i, mu_j, phi, theta_refl, rot1_refl, rot2_refl);
				Eigen::Matrix4d L1_refl_T = geometry::getRotationMatrix(std::cos(2.0 * rot1_refl), std::sin(2.0 * rot1_refl));
				Eigen::Matrix4d L2_refl_T = geometry::getRotationMatrix(std::cos(2.0 * rot2_refl), std::sin(2.0 * rot2_refl));
				Eigen::Matrix4d grad_F_refl = L2_refl_T * grad_R_val * L1_refl_T;

				double theta_tran, rot1_tran, rot2_tran;
				get_scat_geom(mu_i, mu_j, phi, theta_tran, rot1_tran, rot2_tran);
				Eigen::Matrix4d L1_tran_T = geometry::getRotationMatrix(std::cos(2.0 * rot1_tran), std::sin(2.0 * rot1_tran));
				Eigen::Matrix4d L2_tran_T = geometry::getRotationMatrix(std::cos(2.0 * rot2_tran), std::sin(2.0 * rot2_tran));
				Eigen::Matrix4d grad_F_tran = L2_tran_T * grad_T_val * L1_tran_T;

				double common_grad_factor = geom_factor * (geo.d_phi / (2.0 * std::numbers::pi));

				#pragma omp critical
				{
					accumulate_gradient_to_grid(theta_refl, (grad_F_refl * common_grad_factor).eval(), theta_grid, grad_P);
					accumulate_gradient_to_grid(theta_tran, (grad_F_tran * common_grad_factor).eval(), theta_grid, grad_P);
				}
			}
		}
	}

	result.scattering_phase_matrix.resize(n_theta);
	
	for(int i = 0; i < n_theta; ++i)
	{
		result.scattering_phase_matrix[i] = {theta_grid[i], grad_P[i]};
	}

	return result;
}

} // namespace paad::core
