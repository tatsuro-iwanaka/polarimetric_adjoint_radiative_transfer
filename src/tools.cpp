#include "tools.hpp"
#include "geometry.hpp"

#include <netcdf>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <numbers>

namespace paad::tools
{

struct InterpWeight
{
	int idx0;
	int idx1;

	double w0;
	double w1;
};

static InterpWeight get_interp_weights(const Eigen::VectorXd& grid, double val)
{
	int n = grid.size();

	if (val <= grid(0))
	{
		return {0, 0, 1.0, 0.0};
	}

	if (val >= grid(n - 1))
	{
		return {n - 1, n - 1, 1.0, 0.0};
	}
	
	int idx1 = 0;

	while(idx1 < n && grid(idx1) < val)
	{
		idx1++;
	}

	int idx0 = idx1 - 1;
	
	double w1 = (val - grid(idx0)) / (grid(idx1) - grid(idx0));

	return {idx0, idx1, 1.0 - w1, w1};
}

static InterpWeight get_phi_weights(const Eigen::VectorXd& grid, double val)
{
	val = std::fmod(val, 2.0 * std::numbers::pi);

	if (val < 0.0)
	{
		val += 2.0 * std::numbers::pi;
	}

	int n = grid.size();

	if (val >= grid(n - 1))
	{
		double w1 = (val - grid(n - 1)) / (2.0 * std::numbers::pi - grid(n - 1));
		return {n - 1, 0, 1.0 - w1, w1};
	}
	
	int idx1 = 0;

	while(idx1 < n && grid(idx1) < val)
	{
		idx1++;
	}

	int idx0 = idx1 - 1;

	if (idx0 < 0)
	{
		idx0 = 0;
	}
	
	double w1 = (val - grid(idx0)) / (grid(idx1) - grid(idx0));

	return {idx0, idx1, 1.0 - w1, w1};
}

void generateAdjointSourceFromObservations(const std::string& forward_filepath, const std::vector<ObservationPoint>& observations, const std::string& output_filepath, ObservableType obs_type)
{
	std::cout << "[paad::tools] Generating Adjoint Source (Analytical Fourier Adjoint) ...\n";
	std::cout << "  Observation points: " << observations.size() << "\n";

	netCDF::NcFile fwdFile(forward_filepath, netCDF::NcFile::read);

	auto var_R_m_fwd = fwdFile.getVar("reflectance_m");

	if (var_R_m_fwd.isNull())
	{
		throw std::runtime_error("Variable 'reflectance_m' not found in forward result.");
	}

	auto dims_R = var_R_m_fwd.getDims();
	int Nspectral = dims_R[0].getSize();
	int Nmode = dims_R[1].getSize() - 1; 
	int Ntheta_e = dims_R[2].getSize();
	int Ntheta_i = dims_R[3].getSize();
	int Nstokes = (dims_R.size() > 4) ? dims_R[4].getSize() : 1;

	if (obs_type == ObservableType::DegreeOfPolarization && Nstokes == 1)
	{
		throw std::runtime_error("ObservableType::DegreeOfPolarization requires Linear or Full Stokes polarization mode (Nstokes >= 3).");
	}

	std::vector<double> R_m_fwd_flat(Nspectral * (Nmode + 1) * Ntheta_e * Ntheta_i * Nstokes * Nstokes);
	var_R_m_fwd.getVar(R_m_fwd_flat.data());

	bool has_thermal = false;
	auto var_E_fwd = fwdFile.getVar("thermal_emission");

	std::vector<double> E_fwd_flat;
	std::vector<double> adj_E_flat;

	if (!var_E_fwd.isNull())
	{
		has_thermal = true;
		E_fwd_flat.resize(Nspectral * Ntheta_e * Nstokes);
		adj_E_flat.assign(Nspectral * Ntheta_e * Nstokes, 0.0);
		var_E_fwd.getVar(E_fwd_flat.data());
	}

	int Nphi = Ntheta_e * 4 + 1;
	auto dim_phi = fwdFile.getDim("delta_phi");

	if (!dim_phi.isNull())
	{
		Nphi = dim_phi.getSize();
	}

	paad::geometry::Geometry geo = paad::geometry::generateGeometryGaussRadau(Ntheta_e, Nphi, Nmode);

	Eigen::VectorXd grid_te = geo.theta_uh;
	Eigen::VectorXd grid_ti = Eigen::VectorXd::Zero(Ntheta_i);

	for (int i = 0; i < Ntheta_i; ++i)
	{
		grid_ti(i) = std::numbers::pi - geo.theta_lh(i);
	}

	std::vector<double> adj_R_m_flat(R_m_fwd_flat.size(), 0.0);
	double cost_function_J = 0.0;
	int dim_mat = Ntheta_e * Nstokes;

	for (int l = 0; l < Nspectral; l++)
	{
		std::vector<Eigen::MatrixXd> F_mode(Nmode + 1, Eigen::MatrixXd::Zero(dim_mat, dim_mat));

		for (int m = 0; m <= Nmode; ++m)
		{
			for (int e = 0; e < Ntheta_e; ++e)
			{
				for (int i = 0; i < Ntheta_i; ++i)
				{
					for (int r = 0; r < Nstokes; ++r)
					{
						for (int c = 0; c < Nstokes; ++c)
						{
							int idx = l * ((Nmode + 1) * Ntheta_e * Ntheta_i * Nstokes * Nstokes) + m * (Ntheta_e * Ntheta_i * Nstokes * Nstokes) + e * (Ntheta_i * Nstokes * Nstokes) + i * (Nstokes * Nstokes) + r * Nstokes + c;
							F_mode[m](e * Nstokes + r, i * Nstokes + c) = R_m_fwd_flat[idx];
						}
					}
				}
			}
		}

		auto R_real_fwd = paad::geometry::reconstruct(F_mode, geo);

		for (const auto& obs : observations)
		{
			auto w_e = get_interp_weights(grid_te, obs.theta_e);
			auto w_i = get_interp_weights(grid_ti, obs.theta_i);
			auto w_p = get_phi_weights(geo.phi, obs.phi);

			int nodes_e[2] = {w_e.idx0, w_e.idx1};
			int nodes_i[2] = {w_i.idx0, w_i.idx1};
			int nodes_p[2] = {w_p.idx0, w_p.idx1};
			
			double I_mod = 0.0, Q_mod = 0.0, U_mod = 0.0, V_mod = 0.0;

			if (obs.solar_flux > 1e-15)
			{
				for (int p_idx = 0; p_idx < 2; ++p_idx)
				{
					for (int ei = 0; ei < 2; ++ei)
					{
						for (int ii = 0; ii < 2; ++ii)
						{
							double w_total = w_e.w0 * (1 - ei) + w_e.w1 * ei;
							w_total *= w_i.w0 * (1 - ii) + w_i.w1 * ii;
							w_total *= w_p.w0 * (1 - p_idx) + w_p.w1 * p_idx;

							I_mod += w_total * R_real_fwd[nodes_p[p_idx]](nodes_e[ei] * Nstokes + 0, nodes_i[ii] * Nstokes + 0);

							if (Nstokes > 1)
							{
								Q_mod += w_total * R_real_fwd[nodes_p[p_idx]](nodes_e[ei] * Nstokes + 1, nodes_i[ii] * Nstokes + 0);
							}

							if (Nstokes > 2)
							{
								U_mod += w_total * R_real_fwd[nodes_p[p_idx]](nodes_e[ei] * Nstokes + 2, nodes_i[ii] * Nstokes + 0);
							}

							if (Nstokes > 3)
							{
								V_mod += w_total * R_real_fwd[nodes_p[p_idx]](nodes_e[ei] * Nstokes + 3, nodes_i[ii] * Nstokes + 0);
							}
						}
					}
				}

				I_mod *= obs.solar_flux;
				Q_mod *= obs.solar_flux;
				U_mod *= obs.solar_flux;
				V_mod *= obs.solar_flux;
			}

			if (has_thermal)
			{
				for (int ei = 0; ei < 2; ++ei)
				{
					double w = (ei == 0) ? w_e.w0 : w_e.w1;
					int idx_E = l * (Ntheta_e * Nstokes) + nodes_e[ei] * Nstokes;
					
					I_mod += w * E_fwd_flat[idx_E + 0];

					if (Nstokes > 1)
					{
						Q_mod += w * E_fwd_flat[idx_E + 1];
					}

					if (Nstokes > 2)
					{
						U_mod += w * E_fwd_flat[idx_E + 2];
					}

					if (Nstokes > 3)
					{
						V_mod += w * E_fwd_flat[idx_E + 3];
					}
				}
			}

			double dJ_dI = 0.0, dJ_dQ = 0.0, dJ_dU = 0.0, dJ_dV = 0.0;

			if (obs_type == ObservableType::Radiance)
			{
				double dI = I_mod - obs.I;
				double dQ = (Nstokes > 1) ? Q_mod - obs.Q : 0.0;
				double dU = (Nstokes > 2) ? U_mod - obs.U : 0.0;
				double dV = (Nstokes > 3) ? V_mod - obs.V : 0.0;
				
				cost_function_J += 0.5 * (dI*dI + dQ*dQ + dU*dU + dV*dV) * obs.weight;
				
				dJ_dI = dI * obs.weight;

				if (Nstokes > 1)
				{
					dJ_dQ = dQ * obs.weight;
				}

				if (Nstokes > 2)
				{
					dJ_dU = dU * obs.weight;
				}

				if (Nstokes > 3)
				{
					dJ_dV = dV * obs.weight;
				}
			}
			else if (obs_type == ObservableType::DegreeOfPolarization)
			{
				if (I_mod > 1e-15 && obs.I > 1e-15)
				{
					double P_mod = std::sqrt(Q_mod * Q_mod + U_mod * U_mod + V_mod * V_mod) / I_mod;
					double P_obs = std::sqrt(obs.Q * obs.Q + obs.U * obs.U + obs.V * obs.V) / obs.I;

					double dP = P_mod - P_obs;
					cost_function_J += 0.5 * dP * dP * obs.weight;
					double dP_scaled = dP * obs.weight;

					if (P_mod > 1e-15)
					{
						dJ_dI = dP_scaled * (-P_mod / I_mod);

						if (Nstokes > 1)
						{
							dJ_dQ = dP_scaled * (Q_mod / (I_mod * I_mod * P_mod));
						}

						if (Nstokes > 2)
						{
							dJ_dU = dP_scaled * (U_mod / (I_mod * I_mod * P_mod));
						}

						if (Nstokes > 3)
						{
							dJ_dV = dP_scaled * (V_mod / (I_mod * I_mod * P_mod));
						}
					}
					else if (P_obs > 1e-15)
					{
						dJ_dI = 0.0;

						if (Nstokes > 1)
						{
							dJ_dQ = dP_scaled * (obs.Q / (obs.I * I_mod));
						}

						if (Nstokes > 2)
						{
							dJ_dU = dP_scaled * (obs.U / (obs.I * I_mod));
						}

						if (Nstokes > 3)
						{
							dJ_dV = dP_scaled * (obs.V / (obs.I * I_mod));
						}
					}
				}
			}

			if (obs.solar_flux > 1e-15)
			{
				for (int p_idx = 0; p_idx < 2; ++p_idx)
				{
					double current_phi = geo.phi(nodes_p[p_idx]);

					for (int ei = 0; ei < 2; ++ei)
					{
						for (int ii = 0; ii < 2; ++ii)
						{
							double w_total = ((ei == 0) ? w_e.w0 : w_e.w1) * ((ii == 0) ? w_i.w0 : w_i.w1) * ((p_idx == 0) ? w_p.w0 : w_p.w1);
							
							double w_scaled = w_total * obs.solar_flux;

							for (int m = 0; m <= Nmode; ++m)
							{
								double fourier_factor = (m == 0) ? 1.0 : 2.0;
								double trig_cos = fourier_factor * std::cos(m * current_phi);
								double trig_sin = fourier_factor * std::sin(m * current_phi);

								for (int r = 0; r < Nstokes; ++r)
								{
									double dJ_dStokes = 0.0;

									if (r == 0)
									{
										dJ_dStokes = dJ_dI;
									}
									else if (r == 1)
									{
										dJ_dStokes = dJ_dQ;
									}
									else if (r == 2)
									{
										dJ_dStokes = dJ_dU;
									}
									else if (r == 3)
									{
										dJ_dStokes = dJ_dV;
									}

									if (std::abs(dJ_dStokes) > 1e-30)
									{
										double trig_val = (r == 0 || r == 1) ? trig_cos : trig_sin;
										
										int idx = l * ((Nmode + 1) * Ntheta_e * Ntheta_i * Nstokes * Nstokes) + m * (Ntheta_e * Ntheta_i * Nstokes * Nstokes) + nodes_e[ei] * (Ntheta_i * Nstokes * Nstokes) + nodes_i[ii] * (Nstokes * Nstokes) + r * Nstokes + 0;
										adj_R_m_flat[idx] += dJ_dStokes * w_scaled * trig_val;
									}
								}
							}
						}
					}
				}
			}

			if (has_thermal)
			{
				for (int ei = 0; ei < 2; ++ei)
				{
					double w = (ei == 0) ? w_e.w0 : w_e.w1;
					int idx_E = l * (Ntheta_e * Nstokes) + nodes_e[ei] * Nstokes;
					
					adj_E_flat[idx_E + 0] += dJ_dI * w;

					if (Nstokes > 1)
					{
						adj_E_flat[idx_E + 1] += dJ_dQ * w;
					}

					if (Nstokes > 2)
					{
						adj_E_flat[idx_E + 2] += dJ_dU * w;
					}

					if (Nstokes > 3)
					{
						adj_E_flat[idx_E + 3] += dJ_dV * w;
					}
				}
			}
		}
	}

	netCDF::NcFile outFile(output_filepath, netCDF::NcFile::replace);
	
	auto get_or_add_dim = [&](const std::string& name)
	{
		auto dim_in = fwdFile.getDim(name);

		if (dim_in.isNull())
		{
			return netCDF::NcDim();
		}

		auto dim_out = outFile.getDim(name);

		if (dim_out.isNull())
		{
			dim_out = outFile.addDim(name, dim_in.getSize());
		}

		return dim_out;
	};

	std::vector<netCDF::NcDim> out_dims_R;

	for (const auto& dim_name : {"wavelength", "wavenumber", "frequency", "M", "theta_e", "theta_i", "stokes"})
	{
		auto dim = get_or_add_dim(dim_name);

		if (!dim.isNull())
		{
			out_dims_R.push_back(dim);

			if (std::string(dim_name) == "stokes")
			{
				out_dims_R.push_back(dim);
			}
		}
	}

	auto var_R_out = outFile.addVar("adjoint_reflectance_m", netCDF::ncDouble, out_dims_R);
	var_R_out.putAtt("long_name", "Fourier coefficients of adjoint source from exact analytical integration");
	var_R_out.putVar(adj_R_m_flat.data());

	if (has_thermal)
	{
		std::vector<netCDF::NcDim> out_dims_E;

		for (const auto& dim_name : {"wavelength", "wavenumber", "frequency", "theta_e", "stokes"})
		{
			auto dim = get_or_add_dim(dim_name);
			
			if (!dim.isNull())
			{
				out_dims_E.push_back(dim);
			}
		}

		auto var_E_out = outFile.addVar("adjoint_thermal_emission", netCDF::ncDouble, out_dims_E);
		var_E_out.putAtt("long_name", "Adjoint source of thermal emission");
		var_E_out.putVar(adj_E_flat.data());
	}

	std::cout << "[paad::tools] Adjoint Source successfully generated.\n";
	std::cout << "  Cost Function J = " << cost_function_J << "\n";
}

void generateSensitivityAdjointSource(const std::string& forward_filepath, const std::string& output_filepath, double target_theta_e, double target_theta_i, double target_phi, int target_stokes, bool is_thermal, double solar_flux)
{
	std::cout << "[paad::tools] Generating Sensitivity Adjoint Source (Analytical Fourier Adjoint)...\n";
	std::cout << "  Target: theta_e=" << target_theta_e << ", theta_i=" << target_theta_i << ", phi=" << target_phi << "\n";
	std::cout << "  Stokes index: " << target_stokes << ", Thermal: " << std::boolalpha << is_thermal << "\n";

	netCDF::NcFile fwdFile(forward_filepath, netCDF::NcFile::read);

	auto var_R_m_fwd = fwdFile.getVar("reflectance_m");

	if (var_R_m_fwd.isNull())
	{
		throw std::runtime_error("Variable 'reflectance_m' not found in forward result.");
	}

	auto dims_R = var_R_m_fwd.getDims();
	int Nspectral = dims_R[0].getSize();
	int Nmode = dims_R[1].getSize() - 1; 
	int Ntheta_e = dims_R[2].getSize();
	int Ntheta_i = dims_R[3].getSize();
	int Nstokes = (dims_R.size() > 4) ? dims_R[4].getSize() : 1;

	if (target_stokes < 0 || target_stokes >= Nstokes)
	{
		throw std::runtime_error("Target stokes index is out of bounds.");
	}

	Eigen::VectorXd grid_te(Ntheta_e), grid_ti(Ntheta_i);
	auto var_te = fwdFile.getVar("theta_e");

	if (!var_te.isNull())
	{
		var_te.getVar(grid_te.data());
	}
	else
	{
		throw std::runtime_error("theta_e not found in netCDF.");
	}

	auto var_ti = fwdFile.getVar("theta_i");

	if (!var_ti.isNull())
	{
		var_ti.getVar(grid_ti.data());
	}
	else
	{
		throw std::runtime_error("theta_i not found in netCDF.");
	}

	std::vector<double> adj_R_m_flat(Nspectral * (Nmode + 1) * Ntheta_e * Ntheta_i * Nstokes * Nstokes, 0.0);
	std::vector<double> adj_E_flat;

	if (is_thermal)
	{
		adj_E_flat.assign(Nspectral * Ntheta_e * Nstokes, 0.0);
	}

	auto w_e = get_interp_weights(grid_te, target_theta_e);
	auto w_i = get_interp_weights(grid_ti, target_theta_i);
	int nodes_e[2] = {w_e.idx0, w_e.idx1};
	int nodes_i[2] = {w_i.idx0, w_i.idx1};

	double source_val = 1.0; 

	for (int l = 0; l < Nspectral; l++)
	{
		if (!is_thermal && solar_flux > 1e-15)
		{
			for (int m = 0; m <= Nmode; ++m)
			{
				double fourier_factor = (m == 0) ? 1.0 : 2.0;
				double trig_val = fourier_factor * ((target_stokes == 0 || target_stokes == 1) ? std::cos(m * target_phi) : std::sin(m * target_phi));

				for (int ei = 0; ei < 2; ++ei)
				{
					for (int ii = 0; ii < 2; ++ii)
					{
						double w_total = ((ei == 0) ? w_e.w0 : w_e.w1) * ((ii == 0) ? w_i.w0 : w_i.w1);
						double w_scaled = w_total * solar_flux * trig_val * source_val;
						int idx = l * ((Nmode + 1) * Ntheta_e * Ntheta_i * Nstokes * Nstokes)+ m * (Ntheta_e * Ntheta_i * Nstokes * Nstokes) + nodes_e[ei] * (Ntheta_i * Nstokes * Nstokes) + nodes_i[ii] * (Nstokes * Nstokes) + target_stokes * Nstokes + 0;
						adj_R_m_flat[idx] += w_scaled;
					}
				}
			}
		}
		else if (is_thermal)
		{
			for (int ei = 0; ei < 2; ++ei)
			{
				double w = (ei == 0) ? w_e.w0 : w_e.w1;
				int idx_E = l * (Ntheta_e * Nstokes) + nodes_e[ei] * Nstokes;
				adj_E_flat[idx_E + target_stokes] += source_val * w;
			}
		}
	}

	netCDF::NcFile outFile(output_filepath, netCDF::NcFile::replace);

	auto get_or_add_dim = [&](const std::string& name)
	{
		auto dim_in = fwdFile.getDim(name);

		if (dim_in.isNull())
		{
			return netCDF::NcDim();
		}

		auto dim_out = outFile.getDim(name);

		if (dim_out.isNull())
		{
			dim_out = outFile.addDim(name, dim_in.getSize());
		}

		return dim_out;
	};

	std::vector<netCDF::NcDim> out_dims_R;

	for (const auto& dim_name : {"wavelength", "wavenumber", "frequency", "M", "theta_e", "theta_i", "stokes"})
	{
		auto dim = get_or_add_dim(dim_name);

		if (!dim.isNull())
		{
			out_dims_R.push_back(dim);

			if (std::string(dim_name) == "stokes")
			{
				out_dims_R.push_back(dim);
			}
		}
	}

	auto var_R_out = outFile.addVar("adjoint_reflectance_m", netCDF::ncDouble, out_dims_R);
	var_R_out.putAtt("long_name", "Sensitivity adjoint source (Analytical)");
	var_R_out.putVar(adj_R_m_flat.data());

	if (is_thermal)
	{
		std::vector<netCDF::NcDim> out_dims_E;

		for (const auto& dim_name : {"wavelength", "wavenumber", "frequency", "theta_e", "stokes"})
		{
			auto dim = get_or_add_dim(dim_name);

			if (!dim.isNull())
			{
				out_dims_E.push_back(dim);
			}
		}

		auto var_E_out = outFile.addVar("adjoint_thermal_emission", netCDF::ncDouble, out_dims_E);
		var_E_out.putAtt("long_name", "Sensitivity adjoint source (thermal)");
		var_E_out.putVar(adj_E_flat.data());
	}
}

}
