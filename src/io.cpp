#include "io.hpp"

#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <numbers>
#include <iostream>

#include <Eigen/Dense>
#include <netcdf>

#include "units.hpp"

namespace paad::io
{

void exportResultNetCDF(const core::RadiativeTransferResult& result, const core::Simulation& sim_info, const atmosphere::AtmosphereModel& atmosphere, const geometry::Geometry& geometry, const core::Spectral& spectral_info)
{
	std::filesystem::path dir(sim_info.directory_name);

	if (!std::filesystem::exists(dir))
	{
		std::filesystem::create_directories(dir);
	}

	std::string filename = (dir / sim_info.result_name).string();

	std::cout << "[paad::io] Exporting polarimetric result to " << filename << " ...\n";

	netCDF::NcFile outputFile(filename, netCDF::NcFile::replace);

	outputFile.putAtt("simulation_name", sim_info.simulation_name);
	std::string pol_mode_str = "Unknown";

	for (const auto& [key, val] : paad::map_polarization_mode)
	{
		if (val == sim_info.polarization_mode && (key == "SCALAR" || key == "LINEAR" || key == "FULL_STOKES"))
		{
			pol_mode_str = key;
			break;
		}
	}

	outputFile.putAtt("polarization_mode", pol_mode_str);

	int Ntheta = geometry.Ntheta;
	int Nphi = geometry.Nphi;
	int Nmode = geometry.M;
	int Nlayer = result.altitude.size();
	
	int Nspectral = spectral_info.output_grid.size();
	int Nscattering_angle = sim_info.n_scattering_angle;
	
	int Nstokes = 4;

	if (sim_info.polarization_mode == PolarizationMode::Scalar)
	{
		Nstokes = 1;
	}
	else if (sim_info.polarization_mode == PolarizationMode::Linear)
	{
		Nstokes = 3;
	}

	int Nstokes2 = Nstokes * Nstokes;

	netCDF::NcDim dim_theta_e = outputFile.addDim("theta_e", Ntheta);
	netCDF::NcDim dim_theta_i = outputFile.addDim("theta_i", Ntheta);
	netCDF::NcDim dim_phi = outputFile.addDim("delta_phi", Nphi);
	netCDF::NcDim dim_M = outputFile.addDim("M", Nmode + 1);
	netCDF::NcDim dim_layer = outputFile.addDim("layer", Nlayer);
	netCDF::NcDim dim_scattering_angle = outputFile.addDim("scattering_angle", Nscattering_angle);
	netCDF::NcDim dim_stokes = outputFile.addDim("stokes", Nstokes); 
	netCDF::NcDim dim_spectral;
	
	if(spectral_info.dimension == SpectralCoordinateDimension::Wavenumber)
	{
		dim_spectral = outputFile.addDim("wavenumber", Nspectral);
	}
	else if (spectral_info.dimension == SpectralCoordinateDimension::Frequency)
	{
		dim_spectral = outputFile.addDim("frequency", Nspectral);
	}
	else
	{
		dim_spectral = outputFile.addDim("wavelength", Nspectral);
	}

	std::vector<double> phi(Nphi);

	for(int i = 0; i < Nphi; i++)
	{
		phi[i] = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(Nphi - 1);
	}

	std::vector<int> _M(Nmode + 1);

	for(int i = 0; i <= Nmode; i++)
	{
		_M[i] = i;
	}

	netCDF::NcVar var_theta_e = outputFile.addVar("theta_e", netCDF::ncDouble, dim_theta_e);
	var_theta_e.putAtt("long_name", "emission zenith angle");
	var_theta_e.putAtt("units", "radian");
	var_theta_e.putVar(result.theta_e.data());

	netCDF::NcVar var_theta_i = outputFile.addVar("theta_i", netCDF::ncDouble, dim_theta_i);
	var_theta_i.putAtt("long_name", "incidence zenith angle");
	var_theta_i.putAtt("units", "radian");
	var_theta_i.putVar(result.theta_i.data());

	netCDF::NcVar var_phi = outputFile.addVar("delta_phi", netCDF::ncDouble, dim_phi);
	var_phi.putAtt("long_name", "azimuthal difference");
	var_phi.putAtt("units", "radian");
	var_phi.putVar(phi.data());

	netCDF::NcVar var_M = outputFile.addVar("M", netCDF::ncInt, dim_M);
	var_M.putAtt("long_name", "Fourier modes");
	var_M.putVar(_M.data());

	netCDF::NcVar var_spectral;

	if(spectral_info.dimension == SpectralCoordinateDimension::Wavenumber)
	{
		var_spectral = outputFile.addVar("wavenumber", netCDF::ncDouble, dim_spectral);
		var_spectral.putAtt("long_name", "wavenumber");
		var_spectral.putAtt("units", "m-1");
	}
	else if (spectral_info.dimension == SpectralCoordinateDimension::Frequency)
	{
		var_spectral = outputFile.addVar("frequency", netCDF::ncDouble, dim_spectral);
		var_spectral.putAtt("long_name", "frequency");
		var_spectral.putAtt("units", "Hz");
	}
	else
	{
		var_spectral = outputFile.addVar("wavelength", netCDF::ncDouble, dim_spectral);
		var_spectral.putAtt("long_name", "wavelength");
		var_spectral.putAtt("units", "m");
	}

	var_spectral.putVar(spectral_info.output_grid.data());

	std::vector<double> reflectance_flat(Nspectral * Ntheta * Ntheta * Nphi * Nstokes2);

	#pragma omp parallel for num_threads(sim_info.n_parallel_spectral)
	for(int l = 0; l < Nspectral; l++)
	{
		auto f_phi = paad::geometry::reconstruct(result.spectral_data[l].reflectance_m_top, phi, Ntheta);

		for(int e = 0; e < Ntheta; e++)
		{
			for(int i = 0; i < Ntheta; i++)
			{
				for(int p = 0; p < Nphi; p++)
				{
					for(int r = 0; r < Nstokes; r++) 
					{
						for(int c = 0; c < Nstokes; c++) 
						{
							int idx = l * (Ntheta * Ntheta * Nphi * Nstokes2) + e * (Ntheta * Nphi * Nstokes2) + i * (Nphi * Nstokes2) + p * Nstokes2 + r * Nstokes + c;
							reflectance_flat[idx] = f_phi[p](Nstokes * e + r, Nstokes * i + c); 
						}
					}
				}
			}
		}
	}

	netCDF::NcVar var_reflectance = outputFile.addVar("reflectance", netCDF::ncDouble, {dim_spectral, dim_theta_e, dim_theta_i, dim_phi, dim_stokes, dim_stokes});
	var_reflectance.setCompression(true, true, 5);
	var_reflectance.putAtt("long_name", "polarimetric reflectance (Mueller matrix)");
	var_reflectance.putVar(reflectance_flat.data());

	std::vector<double> reflectance_m_flat(Nspectral * (Nmode + 1) * Ntheta * Ntheta * Nstokes2);

	#pragma omp parallel for num_threads(sim_info.n_parallel_spectral)
	for(int l = 0; l < Nspectral; l++)
	{
		for(int m = 0; m <= Nmode; m++)
		{
			for(int e = 0; e < Ntheta; e++)
			{
				for(int i = 0; i < Ntheta; i++)
				{
					for(int r = 0; r < Nstokes; r++) 
					{
						for(int c = 0; c < Nstokes; c++) 
						{
							int idx = l * ((Nmode + 1) * Ntheta * Ntheta * Nstokes2) + m * (Ntheta * Ntheta * Nstokes2) + e * (Ntheta * Nstokes2) + i * Nstokes2 + r * Nstokes + c;
							reflectance_m_flat[idx] = result.spectral_data[l].reflectance_m_top[m](Nstokes * e + r, Nstokes * i + c); 
						}
					}
				}
			}
		}
	}

	netCDF::NcVar var_reflectance_m = outputFile.addVar("reflectance_m", netCDF::ncDouble, {dim_spectral, dim_M, dim_theta_e, dim_theta_i, dim_stokes, dim_stokes});
	var_reflectance_m.setCompression(true, true, 5);
	var_reflectance_m.putAtt("long_name", "packed Fourier coefficients for polarimetric reflectance");
	var_reflectance_m.putVar(reflectance_m_flat.data());

	std::vector<double> scattering_angle(Nscattering_angle);

	for(int i = 0; i < Nscattering_angle; ++i)
	{
		scattering_angle[i] = std::numbers::pi / static_cast<double>(Nscattering_angle - 1) * static_cast<double>(i);
	}
	
	netCDF::NcVar var_scattering_angle = outputFile.addVar("scattering_angle", netCDF::ncDouble, dim_scattering_angle);
	var_scattering_angle.putAtt("units", "radian");
	var_scattering_angle.putVar(scattering_angle.data());

	netCDF::NcVar var_z = outputFile.addVar("altitude", netCDF::ncDouble, dim_layer);
	var_z.putAtt("units", "m");
	var_z.putVar(result.altitude.data());

	netCDF::NcVar var_T = outputFile.addVar("temperature", netCDF::ncDouble, dim_layer);
	var_T.putAtt("units", "K");
	var_T.putVar(result.temperature.data());

	netCDF::NcVar var_P = outputFile.addVar("pressure", netCDF::ncDouble, dim_layer);
	var_P.putAtt("units", "Pa");
	var_P.putVar(result.pressure.data());

	netCDF::NcVar var_N = outputFile.addVar("number_density", netCDF::ncDouble, dim_layer);
	var_N.putAtt("units", "m-3");
	var_N.putVar(result.number_density.data());

	std::vector<double> tau(Nlayer * Nspectral), ka(Nlayer * Nspectral), ks(Nlayer * Nspectral), omega(Nlayer * Nspectral), g(Nlayer * Nspectral);
	
	for(int l = 0; l < Nlayer; l++)
	{
		#pragma omp parallel for num_threads(sim_info.n_parallel_spectral)
		for(int i = 0; i < Nspectral; i++)
		{
			int idx = l * Nspectral + i;
			tau[idx] = result.spectral_data[i].optical_thickness[l];
			ka[idx] = result.spectral_data[i].absorption_coefficient[l];
			ks[idx] = result.spectral_data[i].scattering_coefficient[l];
			omega[idx] = result.spectral_data[i].single_scattering_albedo[l];

			if(result.spectral_data[i].asymmetry_parameter.size() > l)
			{
				g[idx] = result.spectral_data[i].asymmetry_parameter[l];
			}
		}
	}

	outputFile.addVar("optical_thickness", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(tau.data());
	outputFile.addVar("absorption_coefficient", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(ka.data());
	outputFile.addVar("scattering_coefficient", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(ks.data());
	outputFile.addVar("single_scattering_albedo", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(omega.data());
	outputFile.addVar("asymmetry_parameter", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(g.data());

	std::vector<double> pf_flat(Nlayer * Nspectral * Nscattering_angle * Nstokes2);

	for(int i = 0; i < Nlayer; ++i)
	{
		#pragma omp parallel for num_threads(sim_info.n_parallel_spectral)
		for(int j = 0; j < Nspectral; ++j)
		{
			for(int k = 0; k < Nscattering_angle; ++k)
			{
				for(int r = 0; r < Nstokes; r++) 
				{
					for(int c = 0; c < Nstokes; c++) 
					{
						int idx = i * (Nspectral * Nscattering_angle * Nstokes2) + j * (Nscattering_angle * Nstokes2) + k * Nstokes2 + r * Nstokes + c;
						pf_flat[idx] = result.spectral_data[j].scattering_matrix[i][k](r, c);
					}
				}
			}
		}
	}

	netCDF::NcVar var_pf = outputFile.addVar("scattering_matrix", netCDF::ncDouble, {dim_layer, dim_spectral, dim_scattering_angle, dim_stokes, dim_stokes});
	var_pf.setCompression(true, true, 5);
	var_pf.putAtt("long_name", "bulk polarimetric scattering matrix");
	var_pf.putVar(pf_flat.data());

	for(size_t i = 0; i < atmosphere.species.size(); i++)
	{
		netCDF::NcGroup group = outputFile.addGroup("species_" + std::to_string(i + 1));
		group.putAtt("name", atmosphere.species[i].name);       

		group.addVar("mixing_ratio", netCDF::ncDouble, dim_layer).putVar(atmosphere.species[i].vertical_mixing_ratio_profile.data());
		group.addVar("number_density", netCDF::ncDouble, dim_layer).putVar(atmosphere.species[i].vertical_number_density_profile.data());

		std::vector<double> scs(Nlayer * Nspectral), acs(Nlayer * Nspectral);

		for(int j = 0; j < Nlayer; ++j)
		{
			for(int k = 0; k < Nspectral; ++k)
			{
				scs[j * Nspectral + k] = result.spectral_data[k].species_scattering_cross_section[j][i];
				acs[j * Nspectral + k] = result.spectral_data[k].species_absorption_cross_section[j][i];
			}
		}

		group.addVar("scattering_cross_section", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(scs.data());
		group.addVar("absorption_cross_section", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(acs.data());
	}

	netCDF::NcVar var_emission = outputFile.addVar("thermal_emission", netCDF::ncDouble, {dim_spectral, dim_theta_e, dim_stokes});
	var_emission.setCompression(true, true, 5);
	
	if(spectral_info.dimension == SpectralCoordinateDimension::Wavenumber)
	{
		var_emission.putAtt("long_name", "upward atmospheric thermal emission (wavenumber, emission direction)");
		var_emission.putAtt("units", "W/m2/sr/m-1");
	}
	else if (spectral_info.dimension == SpectralCoordinateDimension::Frequency)
	{
		var_emission.putAtt("long_name", "upward atmospheric thermal emission (frequency, emission direction)");
		var_emission.putAtt("units", "W/m2/sr/Hz");
	}
	else
	{
		var_emission.putAtt("long_name", "upward atmospheric thermal emission (wavelength, emission direction)");
		var_emission.putAtt("units", "W/m2/sr/m");
	}

	std::vector<double> thermal_emission_flat(Nspectral * Ntheta * Nstokes);
	
	#pragma omp parallel for num_threads(sim_info.n_parallel_spectral)
	for(int l = 0; l < Nspectral; l++)
	{
		for(int e = 0; e < Ntheta; e++)
		{
			for(int r = 0; r < Nstokes; r++)
			{
				int idx = l * (Ntheta * Nstokes) + e * Nstokes + r;

				if(result.spectral_data[l].source_up.size() > 0)
				{
					thermal_emission_flat[idx] = result.spectral_data[l].source_up(Nstokes * e + r);
				}
				else
				{
					thermal_emission_flat[idx] = 0.0;
				}
			}
		}
	}

	var_emission.putVar(thermal_emission_flat.data());

	if (sim_info.use_sfi)
	{
		int N_sfi_mu0  = sim_info.sfi_geometry.mu_0.size();
		int N_sfi_mobs = sim_info.sfi_geometry.mu_obs.size();
		int N_sfi_dphi = sim_info.sfi_geometry.dphi.size();

		netCDF::NcDim dim_sfi_mu0  = outputFile.addDim("sfi_mu_0", N_sfi_mu0);
		netCDF::NcDim dim_sfi_mobs = outputFile.addDim("sfi_mu_obs", N_sfi_mobs);
		netCDF::NcDim dim_sfi_dphi = outputFile.addDim("sfi_delta_phi", N_sfi_dphi);

		netCDF::NcVar var_sfi_mu0 = outputFile.addVar("sfi_mu_0", netCDF::ncDouble, dim_sfi_mu0);
		var_sfi_mu0.putAtt("long_name", "SFI incident zenith angle (cos)");
		var_sfi_mu0.putVar(sim_info.sfi_geometry.mu_0.data());

		netCDF::NcVar var_sfi_mobs = outputFile.addVar("sfi_mu_obs", netCDF::ncDouble, dim_sfi_mobs);
		var_sfi_mobs.putAtt("long_name", "SFI observation zenith angle (cos)");
		var_sfi_mobs.putVar(sim_info.sfi_geometry.mu_obs.data());

		netCDF::NcVar var_sfi_dphi = outputFile.addVar("sfi_delta_phi", netCDF::ncDouble, dim_sfi_dphi);
		var_sfi_dphi.putAtt("long_name", "SFI azimuthal difference");
		var_sfi_dphi.putAtt("units", "radian");
		var_sfi_dphi.putVar(sim_info.sfi_geometry.dphi.data());

		netCDF::NcVar var_sfi_ref = outputFile.addVar("sfi_reflectance", netCDF::ncDouble, {dim_spectral, dim_sfi_mu0, dim_sfi_mobs, dim_sfi_dphi, dim_stokes, dim_stokes});
		var_sfi_ref.setCompression(true, true, 5);
		var_sfi_ref.putAtt("long_name", "SFI exact polarimetric reflectance matrix");
		var_sfi_ref.putVar(result.sfi_reflectance.data());

		netCDF::NcVar var_sfi_thm = outputFile.addVar("sfi_thermal_emission", netCDF::ncDouble, {dim_spectral, dim_sfi_mobs, dim_stokes});
		var_sfi_thm.setCompression(true, true, 5);
		var_sfi_thm.putAtt("long_name", "SFI exact upward thermal emission vector");
		var_sfi_thm.putVar(result.sfi_thermal_emission.data());
	}

}

void exportJacobianNetCDF(const std::string& filename, const std::vector<core::AtmosphereSensitivity>& jacobians, const atmosphere::AtmosphereModel& atmosphere, const core::Spectral& spectral_info)
{
	std::cout << "[paad::io] Exporting Jacobian (Sensitivities) to " << filename << " ...\n";

	netCDF::NcFile outputFile(filename, netCDF::NcFile::replace);

	int Nlayer = atmosphere.layers.size();
	
	int Nspectral = spectral_info.output_grid.size();
	if (jacobians.size() != Nspectral) throw std::runtime_error("Jacobian size mismatch with spectral grid.");

	netCDF::NcDim dim_layer = outputFile.addDim("layer", Nlayer);
	netCDF::NcDim dim_spectral;
	
	if(spectral_info.dimension == SpectralCoordinateDimension::Wavenumber)
	{
		dim_spectral = outputFile.addDim("wavenumber", Nspectral);
		outputFile.addVar("wavenumber", netCDF::ncDouble, dim_spectral).putVar(spectral_info.output_grid.data());
	}
	else if (spectral_info.dimension == SpectralCoordinateDimension::Frequency)
	{
		dim_spectral = outputFile.addDim("frequency", Nspectral);
		outputFile.addVar("frequency", netCDF::ncDouble, dim_spectral).putVar(spectral_info.output_grid.data());
	}
	else
	{
		dim_spectral = outputFile.addDim("wavelength", Nspectral);
		outputFile.addVar("wavelength", netCDF::ncDouble, dim_spectral).putVar(spectral_info.output_grid.data());
	}

	std::vector<double> dT(Nlayer * Nspectral), dP(Nlayer * Nspectral), dN(Nlayer * Nspectral);

	for(int l = 0; l < Nlayer; ++l)
	{
		for(int s = 0; s < Nspectral; ++s)
		{
			dT[l * Nspectral + s] = jacobians[s].temperature[l];
			dP[l * Nspectral + s] = jacobians[s].pressure[l];
			dN[l * Nspectral + s] = jacobians[s].number_density[l];
		}
	}

	outputFile.addVar("grad_temperature", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(dT.data());
	outputFile.addVar("grad_pressure", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(dP.data());
	outputFile.addVar("grad_number_density", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(dN.data());

	std::vector<double> dSurfAlb(Nspectral), dSurfEmis(Nspectral), dSurfTemp(Nspectral);

	for(int s = 0; s < Nspectral; ++s)
	{
		dSurfAlb[s] = jacobians[s].surface.albedo;
		dSurfEmis[s] = jacobians[s].surface.emissivity;
		dSurfTemp[s] = jacobians[s].surface.temperature;
	}

	outputFile.addVar("grad_surface_albedo", netCDF::ncDouble, dim_spectral).putVar(dSurfAlb.data());
	outputFile.addVar("grad_surface_emissivity", netCDF::ncDouble, dim_spectral).putVar(dSurfEmis.data());
	outputFile.addVar("grad_surface_temperature", netCDF::ncDouble, dim_spectral).putVar(dSurfTemp.data());

	for(size_t i = 0; i < atmosphere.species.size(); ++i)
	{
		netCDF::NcGroup group = outputFile.addGroup("species_" + std::to_string(i + 1));
		group.putAtt("name", atmosphere.species[i].name);

		std::vector<double> dNd(Nlayer * Nspectral), dMr(Nlayer * Nspectral);

		for(int l = 0; l < Nlayer; ++l)
		{
			for(int s = 0; s < Nspectral; ++s)
			{
				dNd[l * Nspectral + s] = jacobians[s].species[i].number_density[l];
				dMr[l * Nspectral + s] = jacobians[s].species[i].mixing_ratio[l];
			}
		}

		group.addVar("grad_number_density", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(dNd.data());
		group.addVar("grad_mixing_ratio", netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(dMr.data());

		auto extract_2d = [&](const std::string& var_name, auto extract_func)
		{
			std::vector<double> vec(Nlayer * Nspectral);

			for(int l = 0; l < Nlayer; ++l)
			{
				for(int s = 0; s < Nspectral; ++s)
				{
					vec[l * Nspectral + s] = extract_func(jacobians[s].species[i].scattering)[l];
				}
			}

			group.addVar(var_name, netCDF::ncDouble, {dim_layer, dim_spectral}).putVar(vec.data());
		};

		if (std::holds_alternative<atmosphere::MieScattering>(atmosphere.species[i].scattering_model))
		{
			extract_2d("grad_refractive_index_real", [](const auto& sca) { return std::get<core::MieSensitivity>(sca).refractive_index_real; });
			extract_2d("grad_refractive_index_imag", [](const auto& sca) { return std::get<core::MieSensitivity>(sca).refractive_index_imag; });
			
			auto& sd = std::get<atmosphere::MieScattering>(atmosphere.species[i].scattering_model).size_distribution;

			if (std::holds_alternative<atmosphere::LogNormalDistribution>(sd))
			{
				extract_2d("grad_r_g", [](const auto& sca) { return std::get<core::LogNormalSensitivity>(std::get<core::MieSensitivity>(sca).size_distribution).r_g; });
				extract_2d("grad_sigma_g", [](const auto& sca) { return std::get<core::LogNormalSensitivity>(std::get<core::MieSensitivity>(sca).size_distribution).sigma_g; });
			}
			else if (std::holds_alternative<atmosphere::GammaDistribution>(sd))
			{
				extract_2d("grad_a", [](const auto& sca) { return std::get<core::GammaSensitivity>(std::get<core::MieSensitivity>(sca).size_distribution).a; });
				extract_2d("grad_b", [](const auto& sca) { return std::get<core::GammaSensitivity>(std::get<core::MieSensitivity>(sca).size_distribution).b; });
			}
		}
		else if (std::holds_alternative<atmosphere::RayleighScattering>(atmosphere.species[i].scattering_model))
		{
			extract_2d("grad_refractive_index", [](const auto& sca) { return std::get<core::RayleighSensitivity>(sca).refractive_index; });
			extract_2d("grad_depolarization_factor", [](const auto& sca) { return std::get<core::RayleighSensitivity>(sca).depolarization_factor; });
		}
	}
}

std::pair<std::vector<std::vector<Eigen::MatrixXd>>, std::vector<Eigen::VectorXd>> importAdjointSourceNetCDF(const std::string& filepath, const geometry::Geometry& geom, const core::Spectral& spectral_info, PolarizationMode pol_mode)
{
	std::cout << "[paad::io] Importing adjoint source from " << filepath << " ...\n";

	if (!std::filesystem::exists(filepath))
	{
		throw std::runtime_error("Adjoint source file does not exist: " + filepath);
	}

	netCDF::NcFile inputFile(filepath, netCDF::NcFile::read);

	int Ntheta = geom.Ntheta;
	int Nmode = geom.M; 
	int Nspectral = spectral_info.output_grid.size();
	int Nstokes = (pol_mode == PolarizationMode::Scalar) ? 1 : ((pol_mode == PolarizationMode::Linear) ? 3 : 4);
	int Nstokes2 = Nstokes * Nstokes;

	netCDF::NcVar var_R = inputFile.getVar("adjoint_reflectance_m");

	if (var_R.isNull())
	{
		throw std::runtime_error("Variable 'adjoint_reflectance_m' not found in adjoint source file.");
	}

	std::vector<double> R_flat(Nspectral * (Nmode + 1) * Ntheta * Ntheta * Nstokes2);
	var_R.getVar(R_flat.data());

	std::vector<std::vector<Eigen::MatrixXd>> F_mode_list(Nspectral, std::vector<Eigen::MatrixXd>(Nmode + 1, Eigen::MatrixXd::Zero(Ntheta * Nstokes, Ntheta * Nstokes)));

	for(int l = 0; l < Nspectral; l++)
	{
		for(int m = 0; m <= Nmode; m++)
		{
			for(int e = 0; e < Ntheta; e++)
			{
				for(int i = 0; i < Ntheta; i++)
				{
					for(int r = 0; r < Nstokes; r++)
					{
						for(int c = 0; c < Nstokes; c++)
						{
							int idx = l * ((Nmode + 1) * Ntheta * Ntheta * Nstokes2) + m * (Ntheta * Ntheta * Nstokes2) + e * (Ntheta * Nstokes2) + i * Nstokes2 + r * Nstokes + c;
							F_mode_list[l][m](Nstokes * e + r, Nstokes * i + c) = R_flat[idx];
						}
					}
				}
			}
		}
	}

	std::vector<Eigen::VectorXd> emission_up_list(Nspectral, Eigen::VectorXd::Zero(Ntheta * Nstokes));
	netCDF::NcVar var_E = inputFile.getVar("adjoint_thermal_emission");
	
	if (!var_E.isNull())
	{
		std::vector<double> E_flat(Nspectral * Ntheta * Nstokes);
		var_E.getVar(E_flat.data());
		
		for(int l = 0; l < Nspectral; l++)
		{
			for(int e = 0; e < Ntheta; e++)
			{
				for(int r = 0; r < Nstokes; r++)
				{
					int idx = l * (Ntheta * Nstokes) + e * Nstokes + r;
					emission_up_list[l](Nstokes * e + r) = E_flat[idx];
				}
			}
		}
	}

	return {F_mode_list, emission_up_list};
}

}
