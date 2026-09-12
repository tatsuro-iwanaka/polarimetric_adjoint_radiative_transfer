#include "radiative_transfer.hpp"

#include <iostream>
#include <numbers>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <cmath>

#include <netcdf>

#include "configuration.hpp"
#include "io.hpp"
#include "units.hpp"
#include "constants.hpp"
#include "logger.hpp"

#include "sfi.hpp"

namespace paad
{

RadiativeTransfer::RadiativeTransfer() {}

RadiativeTransfer::RadiativeTransfer(const std::string& config_filename) : config_filename_(config_filename)
{
	loadConfiguration();
	setup();
}

void RadiativeTransfer::setup(void)
{
	result_.Ntheta = geometry_.Ntheta;
	result_.Nphi = geometry_.Nphi;
	result_.Nmode = geometry_.M;

	result_.theta_e.resize(geometry_.Ntheta);
	result_.theta_i.resize(geometry_.Ntheta);
	result_.phi.resize(geometry_.Nphi);

	for(int i = 0; i < geometry_.Ntheta; ++i)
	{
		result_.theta_e[i] = geometry_.theta_uh(i);
		result_.theta_i[i] = std::numbers::pi - geometry_.theta_lh(i);
	}

	for(int i = 0; i < geometry_.Nphi; ++i)
	{
		result_.phi[i] = geometry_.phi(i);
	}

	int n_layers = atmosphere_model_.layers.size();
	result_.altitude.resize(n_layers);
	result_.altitude_top.resize(n_layers);
	result_.altitude_bottom.resize(n_layers);
	result_.physical_thickness.resize(n_layers);
	result_.temperature.resize(n_layers);
	result_.pressure.resize(n_layers);
	result_.number_density.resize(n_layers);

	for(int i = 0; i < n_layers; ++i)
	{
		const auto& layer = atmosphere_model_.layers[i];
		result_.altitude[i] = layer.altitude;
		result_.altitude_top[i] = layer.altitude_top;
		result_.altitude_bottom[i] = layer.altitude_bottom;
		result_.physical_thickness[i] = layer.altitude_top - layer.altitude_bottom;
		result_.temperature[i] = layer.temperature;
		result_.pressure[i] = layer.pressure;
		result_.number_density[i] = layer.number_density;
	}

	if (simulation_.use_sfi)
	{
		result_.sfi_mu_0 = simulation_.sfi_geometry.mu_0;
		result_.sfi_mu_obs = simulation_.sfi_geometry.mu_obs;
		result_.sfi_dphi = simulation_.sfi_geometry.dphi;
		
		int n_spec = spectral_.output_grid.empty() ? spectral_.calculation_grid.size() : spectral_.output_grid.size();
		int n_mu0 = result_.sfi_mu_0.size();
		int n_mobs = result_.sfi_mu_obs.size();
		int n_dphi = result_.sfi_dphi.size();
		int n_stokes = (simulation_.polarization_mode == PolarizationMode::Scalar) ? 1 : ((simulation_.polarization_mode == PolarizationMode::Linear) ? 3 : 4);

		result_.sfi_reflectance.assign(n_spec * n_mu0 * n_mobs * n_dphi * n_stokes * n_stokes, 0.0);
		result_.sfi_thermal_emission.assign(n_spec * n_mobs * n_stokes, 0.0);
	}

	PAAD_INFO("Atmosphere setup completed. Layers: " << n_layers);
}

void RadiativeTransfer::setupSpectralGrid()
{
	if (std::holds_alternative<core::MonochromeConfig>(spectral_.config))
	{
		auto mono = std::get<core::MonochromeConfig>(spectral_.config);
		spectral_.calculation_grid = { mono.value };
		spectral_.output_grid = { mono.value };
	}
	else if (std::holds_alternative<core::SpectrumConfig>(spectral_.config))
	{
		auto spec = std::get<core::SpectrumConfig>(spectral_.config);
		
		if (spec.spacing_type == SpectralGridSpacingType::Step)
		{
			int n_out = static_cast<int>(std::round((spec.max - spec.min) / spec.step)) + 1;
			spectral_.output_grid.resize(n_out);

			for (int i = 0; i < n_out; ++i)
			{
				spectral_.output_grid[i] = spec.min + spec.step * i;
			}
		}
		else if (spec.spacing_type == SpectralGridSpacingType::Count)
		{
			spectral_.output_grid.resize(spec.count);

			if (spec.count == 1)
			{
				spectral_.output_grid[0] = spec.min;
			}
			else
			{
				double d = (spec.max - spec.min) / static_cast<double>(spec.count - 1);

				for (int i = 0; i < spec.count; ++i)
				{
					spectral_.output_grid[i] = spec.min + d * i;
				}
			}
		}
		else if (spec.spacing_type == SpectralGridSpacingType::ResolvingPower)
		{
			std::vector<double> grid;
			double current = spec.min;

			while (current <= spec.max)
			{
				grid.push_back(current);
				current += current / spec.resolving_power;
			}

			spectral_.output_grid = grid;
		}

		if (spec.ils.type == ILSType::None)
		{
			spectral_.calculation_grid = spectral_.output_grid;
		}
		else
		{
			double margin = 0.0;

			if (spec.ils.type == ILSType::Gaussian || spec.ils.type == ILSType::Lorentzian)
			{
				double sigma = spec.ils.fwhm / 2.354820045;
				margin = sigma * spec.ils.cutoff_sigma;
			}
			else if (spec.ils.type == ILSType::Boxcar)
			{
				margin = spec.ils.fwhm / 2.0;
			}
			else
			{
				margin = spec.ils.fwhm;
			}
			
			double calc_min = spec.min - margin;
			double calc_max = spec.max + margin;

			if (spec.spacing_type == SpectralGridSpacingType::Step ||  spec.spacing_type == SpectralGridSpacingType::Count)
			{
				double base_step = (spec.spacing_type == SpectralGridSpacingType::Step) ? spec.step : (spec.max - spec.min) / (spec.count > 1 ? static_cast<double>(spec.count - 1) : 1.0);
				
				double calc_step = base_step / static_cast<double>(spec.ils.oversample_factor);
				int n_calc = static_cast<int>(std::round((calc_max - calc_min) / calc_step)) + 1;
				
				spectral_.calculation_grid.resize(n_calc);

				for (int i = 0; i < n_calc; ++i)
				{
					spectral_.calculation_grid[i] = calc_min + calc_step * i;
				}
			}
			else if (spec.spacing_type == SpectralGridSpacingType::ResolvingPower)
			{
				if (calc_min <= 0.0)
				{
					throw std::runtime_error("calc_min <= 0.0 is invalid for ResolvingPower grid generation.");
				}

				double calc_rp = spec.resolving_power * static_cast<double>(spec.ils.oversample_factor);
				std::vector<double> grid;
				double current = calc_min;
				
				while (current <= calc_max)
				{
					grid.push_back(current);
					current += current / calc_rp;
				}

				spectral_.calculation_grid = grid;
			}
		}
	}
	else if (std::holds_alternative<core::BandpassConfig>(spectral_.config))
	{
		auto bp = std::get<core::BandpassConfig>(spectral_.config);
		PAAD_INFO("Loading bandpass filter from: " << bp.filename);
		
		try
		{
			netCDF::NcFile nc_file(bp.filename, netCDF::NcFile::read);
			
			netCDF::NcVar var_spec = nc_file.getVar(bp.varname_spectral);
			if(var_spec.isNull()) throw std::runtime_error("NetCDF var not found: " + bp.varname_spectral);
			
			netCDF::NcVar var_trans = nc_file.getVar(bp.varname_transmission);
			if(var_trans.isNull()) throw std::runtime_error("NetCDF var not found: " + bp.varname_transmission);

			size_t n_data = var_spec.getDim(0).getSize();
			std::vector<double> raw_grid(n_data);
			std::vector<double> raw_trans(n_data);

			var_spec.getVar(raw_grid.data());
			var_trans.getVar(raw_trans.data());

			netCDF::NcVarAtt att_unit = var_spec.getAtt("units");

			if (att_unit.isNull()) 
			{
				throw std::runtime_error("NetCDF var '" + bp.varname_spectral + "' requires 'units' attribute.");
			}

			std::string nc_unit_str;
			att_unit.getValues(nc_unit_str);
			double to_si = units::getUnitInfo(nc_unit_str).to_si;
			
			for (size_t i = 0; i < n_data; ++i)
			{
				raw_grid[i] *= to_si;
			}

			double max_trans = 0.0;

			for (size_t i = 0; i < n_data; ++i) 
			{
				if (raw_trans[i] > max_trans)
				{
					max_trans = raw_trans[i];
				}
			}

			if (max_trans <= 0.0)
			{
				throw std::runtime_error("Filter transmission is all zero or negative.");
			}

			double threshold = max_trans * bp.cutoff_threshold;
			
			size_t start_idx = 0;
			size_t end_idx = n_data - 1;

			while (start_idx < n_data && raw_trans[start_idx] < threshold)
			{
				start_idx++;
			}

			while (end_idx > start_idx && raw_trans[end_idx] < threshold)
			{
				end_idx--;
			}

			if (start_idx > end_idx)
			{
				throw std::runtime_error("Filter transmission is entirely below cutoff threshold.");
			}

			int valid_size = end_idx - start_idx + 1;
			bp.grid.resize(valid_size);
			bp.values.resize(valid_size);

			for (int i = 0; i < valid_size; ++i)
			{
				bp.grid[i] = raw_grid[start_idx + i];
				bp.values[i] = raw_trans[start_idx + i];
			}
			
			double calc_min = bp.grid.front();
			double calc_max = bp.grid.back();

			if (bp.spacing_type == SpectralGridSpacingType::Step)
			{
				int n_calc = static_cast<int>(std::round((calc_max - calc_min) / bp.step)) + 1;
				spectral_.calculation_grid.resize(n_calc);

				for (int i = 0; i < n_calc; ++i)
				{
					spectral_.calculation_grid[i] = calc_min + bp.step * i;
				}
			}
			else if (bp.spacing_type == SpectralGridSpacingType::Count)
			{
				spectral_.calculation_grid.resize(bp.count);

				if (bp.count == 1)
				{
					spectral_.calculation_grid[0] = calc_min;
				}
				else
				{
					double d = (calc_max - calc_min) / static_cast<double>(bp.count - 1);

					for (int i = 0; i < bp.count; ++i)
					{
						spectral_.calculation_grid[i] = calc_min + d * i;
					}
				}
			}
			else if (bp.spacing_type == SpectralGridSpacingType::ResolvingPower)
			{
				std::vector<double> grid;
				double current = calc_min;

				while (current <= calc_max)
				{
					grid.push_back(current);
					current += current / bp.resolving_power;
				}

				spectral_.calculation_grid = grid;
			}

			double num = 0.0;
			double den = 0.0;

			for (size_t i = 0; i < bp.grid.size() - 1; ++i)
			{
				double d_lambda = bp.grid[i + 1] - bp.grid[i];
				double t_avg = (bp.values[i] + bp.values[i + 1]) * 0.5;
				double lambda_t_avg = (bp.grid[i] * bp.values[i] + bp.grid[i + 1] * bp.values[i + 1]) * 0.5;
				num += lambda_t_avg * d_lambda;
				den += t_avg * d_lambda;
			}

			double effective_wavelength = (den > 0.0) ? (num / den) : ((calc_min + calc_max) / 2.0);
			spectral_.output_grid = { effective_wavelength };
			spectral_.config = bp;
			
			PAAD_INFO("Bandpass grid generated. Calculation points: " << spectral_.calculation_grid.size() << ", Effective WL: " << effective_wavelength);

		}
		catch (const netCDF::exceptions::NcException& e)
		{
			PAAD_ERROR("NetCDF Error in Bandpass loading: " << e.what());
			throw std::runtime_error(std::string("NetCDF Error in Bandpass loading: ") + e.what());
		}
	}
}

void RadiativeTransfer::loadConfiguration(void)
{
	configuration::ConfigParser parser;
	auto config = parser.import_json(config_filename_);

	simulation_ = config.simulation;
	spectral_ = config.spectral;
	geometry_ = config.geometry;
	atmosphere_model_ = config.atmosphere;

	PAAD_INFO("Setting up spectral grids...");
	setupSpectralGrid();

	double wavenumber_min = 1.0e100;
	double wavenumber_max = 0.0;
	
	if (!spectral_.calculation_grid.empty())
	{
		double v_front = spectral_.calculation_grid.front();
		double v_back = spectral_.calculation_grid.back();

		if (spectral_.dimension == SpectralCoordinateDimension::Wavelength)
		{
			wavenumber_min = 1.0 / std::max(v_front, v_back);
			wavenumber_max = 1.0 / std::min(v_front, v_back);
		}
		else if (spectral_.dimension == SpectralCoordinateDimension::Wavenumber)
		{
			wavenumber_min = std::min(v_front, v_back);
			wavenumber_max = std::max(v_front, v_back);
		}
		else if (spectral_.dimension == SpectralCoordinateDimension::Frequency)
		{
			wavenumber_min = std::min(v_front, v_back) / constants::SPEED_OF_LIGHT;
			wavenumber_max = std::max(v_front, v_back) / constants::SPEED_OF_LIGHT;
		}
	}

	double line_wing_buffer = 100000.0; 
	wavenumber_min = std::max(0.0, wavenumber_min - line_wing_buffer);
	wavenumber_max = wavenumber_max + line_wing_buffer;

	PAAD_INFO("Compiling HITRAN data in range [" << wavenumber_min << ", " << wavenumber_max << "] m^-1...");
	atmosphere::compileHitranData(atmosphere_model_, simulation_.hitran_filepath, wavenumber_min, wavenumber_max);
	PAAD_INFO("HITRAN compilation finished.");

	setup();
}

void RadiativeTransfer::setAdjointSourceFourier(const std::vector<std::vector<Eigen::MatrixXd>>& F_mode_list, const std::vector<Eigen::VectorXd>& emission_up_list)
{
	int n_spectral = F_mode_list.size();
	adjoint_sources_.resize(n_spectral);
	
	int n_stokes = (simulation_.polarization_mode == PolarizationMode::Scalar) ? 1 : ((simulation_.polarization_mode == PolarizationMode::Linear) ? 3 : 4);
	
	for (int i = 0; i < n_spectral; ++i)
	{
		core::RadiativeLayer adj_source;
		adj_source.resize(geometry_.Ntheta, geometry_.M, n_stokes);
		
		adj_source.reflectance_m_top = F_mode_list[i];
		
		if (!emission_up_list.empty() && emission_up_list.size() > i)
		{
			adj_source.source_up = emission_up_list[i];
		}
		
		adjoint_sources_[i] = adj_source;
	}

	has_custom_adjoint_source_ = true;
	PAAD_INFO("Custom adjoint source loaded. Spectral dimensions: " << n_spectral);
}

void RadiativeTransfer::setAdjointSourceSFI(const std::vector<Eigen::VectorXd>& grad_thm, const std::vector<Eigen::MatrixXd>& grad_ref)
{
	sfi_grad_I_thm_ = grad_thm;
	sfi_grad_R_mat_ = grad_ref;
	has_custom_sfi_gradient_ = true;
	PAAD_INFO("Custom SFI gradients loaded.");
}

void RadiativeTransfer::run(void)
{
	radiative_transfer_solver_.geometry(geometry_);
	radiative_transfer_solver_.setScatteringAngleResolution(simulation_.n_scattering_angle);
	radiative_transfer_solver_.setPolarizationMode(simulation_.polarization_mode);

	radiative_transfer_solver_.setFourierParallelThreads(simulation_.n_parallel_fourier);

	Eigen::setNbThreads(1);
	omp_set_max_active_levels(1); // 暫定
	
	double initial_layer = simulation_.initial_optical_thickness;
	
	int n_spectral_calc = spectral_.calculation_grid.size();
	
	PAAD_INFO("Starting Radiative Transfer Simulation.");
	PAAD_INFO("Mode: " << (simulation_.run_mode == RunMode::Forward ? "FORWARD" : "ADJOINT"));
	PAAD_INFO("Calculation grid points: " << n_spectral_calc << ", Threads (Spectral): " << simulation_.n_parallel_spectral  << ", Threads (Fourier): "  << simulation_.n_parallel_fourier);

	int n_out = spectral_.output_grid.size();
	result_.output_grid = spectral_.output_grid;

	std::vector<std::vector<double>> norm_weights(n_out, std::vector<double>(n_spectral_calc, 0.0));

	bool requires_convolution = false;

	if (std::holds_alternative<core::SpectrumConfig>(spectral_.config) || std::holds_alternative<core::BandpassConfig>(spectral_.config)) 
	{
		for (int i = 0; i < n_out; ++i) 
		{
			std::vector<double> raw_w(n_spectral_calc, 0.0);
			
			if (std::holds_alternative<core::SpectrumConfig>(spectral_.config))
			{
				auto spec = std::get<core::SpectrumConfig>(spectral_.config);

				if (spec.ils.type != ILSType::None)
				{
					requires_convolution = true;
					double out_spec = spectral_.output_grid[i];

					for (int j = 0; j < n_spectral_calc; ++j)
					{
						double delta_x = spectral_.calculation_grid[j] - out_spec;

						if (spec.ils.type == ILSType::Gaussian)
						{
							double sigma = spec.ils.fwhm / 2.354820045;
							raw_w[j] = std::exp(-0.5 * (delta_x * delta_x) / (sigma * sigma));
						}
						else if (spec.ils.type == ILSType::Lorentzian)
						{
							double gamma = spec.ils.fwhm / 2.0;
							raw_w[j] = (gamma * gamma) / ((delta_x * delta_x) + (gamma * gamma));
						}
						else if (spec.ils.type == ILSType::Boxcar)
						{
							if (std::abs(delta_x) <= spec.ils.fwhm / 2.0)
							{
								raw_w[j] = 1.0;
							}
						}
					}
				}
			}
			else if (std::holds_alternative<core::BandpassConfig>(spectral_.config))
			{
				requires_convolution = true;
				auto bp = std::get<core::BandpassConfig>(spectral_.config);

				for (int j = 0; j < n_spectral_calc; ++j)
				{
					raw_w[j] = bp.values[j]; 
				}
			}

			if (requires_convolution)
			{
				double weight_sum = 0.0;

				for (int j = 0; j < n_spectral_calc - 1; ++j)
				{
					double dx = spectral_.calculation_grid[j+1] - spectral_.calculation_grid[j];
					weight_sum += 0.5 * (raw_w[j] + raw_w[j+1]) * dx;
				}

				if (weight_sum > 0.0)
				{
					for (int j = 0; j < n_spectral_calc; ++j)
					{
						norm_weights[i][j] = raw_w[j] / weight_sum;
					}
				}
			}
		}
	}

	auto clear_data = [](core::MonochromeData& d)
	{
		for(auto& m : d.reflectance_m_top)
		{
			m.setZero();
		}

		if(d.source_up.size() > 0)
		{
			d.source_up.setZero();
		}

		for(auto& layer : d.scattering_matrix)
		{
			for(auto& mat : layer)
			{
				mat.setZero();
			}
		}
		
		for(auto& sp : d.species_scattering_matrix)
		{
			for(auto& layer : sp)
			{
				for(auto& mat : layer)
				{
					mat.setZero();
				}
			}
		}
		
		for(auto& sp : d.species_absorption_cross_section)
		{
			for(auto& val : sp)
			{
				val = 0.0;
			}
		}
		
		for(auto& sp : d.species_scattering_cross_section)
		{
			for(auto& val : sp)
			{
				val = 0.0;
			}
		}
		
		for(auto& val : d.absorption_coefficient)
		{
			val = 0.0;
		}

		for(auto& val : d.scattering_coefficient)
		{
			val = 0.0;
		}

		for(auto& val : d.single_scattering_albedo)
		{
			val = 0.0;
		}

		for(auto& val : d.optical_thickness)
		{
			val = 0.0;
		}

		for(auto& val : d.asymmetry_parameter)
		{
			val = 0.0;
		}
	};

	auto add_weighted = [](core::MonochromeData& dest, const core::MonochromeData& src, double weight)
	{
		for(size_t i = 0; i < dest.reflectance_m_top.size(); ++i)
		{
			dest.reflectance_m_top[i] += src.reflectance_m_top[i] * weight;
		}

		if(dest.source_up.size() > 0)
		{
			dest.source_up += src.source_up * weight;
		}

		for(size_t i = 0; i < dest.scattering_matrix.size(); ++i)
		{
			for(size_t j = 0; j < dest.scattering_matrix[i].size(); ++j)
			{
				dest.scattering_matrix[i][j] += src.scattering_matrix[i][j] * weight;
			}
		}

		for(size_t s = 0; s < dest.species_scattering_matrix.size(); ++s)
		{
			for(size_t i = 0; i < dest.species_scattering_matrix[s].size(); ++i)
			{
				for(size_t j = 0; j < dest.species_scattering_matrix[s][i].size(); ++j)
				{
					dest.species_scattering_matrix[s][i][j] += src.species_scattering_matrix[s][i][j] * weight;
				}
			}
		}
		
		for(size_t s = 0; s < dest.species_absorption_cross_section.size(); ++s)
		{
			for(size_t i = 0; i < dest.species_absorption_cross_section[s].size(); ++i)
			{
				dest.species_absorption_cross_section[s][i] += src.species_absorption_cross_section[s][i] * weight;
			}
		}

		for(size_t s = 0; s < dest.species_scattering_cross_section.size(); ++s)
		{
			for(size_t i = 0; i < dest.species_scattering_cross_section[s].size(); ++i)
			{
				dest.species_scattering_cross_section[s][i] += src.species_scattering_cross_section[s][i] * weight;
			}
		}

		for(size_t i = 0; i < dest.absorption_coefficient.size(); ++i)
		{
			dest.absorption_coefficient[i] += src.absorption_coefficient[i] * weight;
		}

		for(size_t i = 0; i < dest.scattering_coefficient.size(); ++i)
		{
			dest.scattering_coefficient[i] += src.scattering_coefficient[i] * weight;
		}

		for(size_t i = 0; i < dest.single_scattering_albedo.size(); ++i)
		{
			dest.single_scattering_albedo[i] += src.single_scattering_albedo[i] * weight;
		}

		for(size_t i = 0; i < dest.optical_thickness.size(); ++i)
		{
			dest.optical_thickness[i] += src.optical_thickness[i] * weight;
		}

		for(size_t i = 0; i < dest.asymmetry_parameter.size(); ++i)
		{
			dest.asymmetry_parameter[i] += src.asymmetry_parameter[i] * weight;
		}
	};

	if (simulation_.run_mode == RunMode::Forward)
	{
		std::vector<core::MonochromeData> raw_spectral_data(n_spectral_calc);

		int n_stokes = (simulation_.polarization_mode == PolarizationMode::Scalar) ? 1 : ((simulation_.polarization_mode == PolarizationMode::Linear) ? 3 : 4);
		int n_mu0  = simulation_.use_sfi ? simulation_.sfi_geometry.mu_0.size() : 0;
		int n_mobs = simulation_.use_sfi ? simulation_.sfi_geometry.mu_obs.size() : 0;
		int n_dphi = simulation_.use_sfi ? simulation_.sfi_geometry.dphi.size() : 0;
		size_t sfi_ref_size = n_mu0 * n_mobs * n_dphi * n_stokes * n_stokes;
		size_t sfi_thm_size = n_mobs * n_stokes;

		std::vector<double> raw_sfi_ref(simulation_.use_sfi ? n_spectral_calc * sfi_ref_size : 0, 0.0);
		std::vector<double> raw_sfi_thm(simulation_.use_sfi ? n_spectral_calc * sfi_thm_size : 0, 0.0);

		#pragma omp parallel for num_threads(simulation_.n_parallel_spectral)
		for(int j = 0; j < n_spectral_calc; ++j)
		{
			double spectral = spectral_.calculation_grid[j];
			raw_spectral_data[j] = radiative_transfer_solver_.computeMonochrome(atmosphere_model_, spectral, spectral_.dimension, initial_layer);

			if (simulation_.use_sfi)
			{
				auto state = radiative_transfer_solver_.computeForwardState(atmosphere_model_, spectral, spectral_.dimension, initial_layer);
				auto fwd_field = radiative_transfer_solver_.computeInternalField(state);
				sfi::SFIIntegrator integrator(geometry_, n_stokes, simulation_.n_parallel_fourier);

				for (int obs = 0; obs < n_mobs; ++obs)
				{
					double mu_obs = simulation_.sfi_geometry.mu_obs[obs];
					
					Eigen::VectorXd I_thm = integrator.computeThermalEmission(fwd_field, state.initial_optical_layers, mu_obs);

					for (int s = 0; s < n_stokes; ++s)
					{
						raw_sfi_thm[j * sfi_thm_size + obs * n_stokes + s] = I_thm(s);
					}

					for (int m0 = 0; m0 < n_mu0; ++m0)
					{
						double mu_0 = simulation_.sfi_geometry.mu_0[m0];

						for (int p = 0; p < n_dphi; ++p)
						{
							double dphi = simulation_.sfi_geometry.dphi[p];
							Eigen::MatrixXd R_mat = integrator.computeReflectanceMatrix(fwd_field, state.initial_optical_layers, mu_obs, mu_0, dphi);
							
							for (int r = 0; r < n_stokes; ++r)
							{
								for (int c = 0; c < n_stokes; ++c)
								{
									int idx = j * sfi_ref_size + m0 * (n_mobs * n_dphi * n_stokes * n_stokes) + obs * (n_dphi * n_stokes * n_stokes) + p * (n_stokes * n_stokes) + r * n_stokes + c;
									raw_sfi_ref[idx] = R_mat(r, c);
								}
							}
						}
					}
				}
			}
		}

		result_.spectral_data.resize(n_out);
		
		if (simulation_.use_sfi)
		{
			result_.sfi_reflectance.assign(n_out * sfi_ref_size, 0.0);
			result_.sfi_thermal_emission.assign(n_out * sfi_thm_size, 0.0);
		}

		if (!requires_convolution)
		{
			if (n_spectral_calc != n_out)
			{
				throw std::runtime_error("Grid size mismatch in direct copy.");
			}

			result_.spectral_data = raw_spectral_data;
			
			if (simulation_.use_sfi)
			{
				result_.sfi_reflectance = raw_sfi_ref;
				result_.sfi_thermal_emission = raw_sfi_thm;
			}
		}
		else
		{
			PAAD_INFO("Applying convolution (Filter/ILS)...");

			#pragma omp parallel for num_threads(simulation_.n_parallel_spectral)
			for (int i = 0; i < n_out; ++i)
			{
				core::MonochromeData conv_data = raw_spectral_data[0]; 
				clear_data(conv_data);

				double W = 0.0;

				for (int j = 0; j < n_spectral_calc - 1; ++j)
				{
					double dx = spectral_.calculation_grid[j+1] - spectral_.calculation_grid[j];
					double w0 = norm_weights[i][j];
					double w1 = norm_weights[i][j+1];
					W += 0.5 * (w0 + w1) * dx;
				}

				for (int j = 0; j < n_spectral_calc - 1; ++j)
				{
					double dx = spectral_.calculation_grid[j+1] - spectral_.calculation_grid[j];
					double w0 = norm_weights[i][j];
					double w1 = norm_weights[i][j+1];
					
					if (W > 0)
					{
						add_weighted(conv_data, raw_spectral_data[j], 0.5 * (w0 / W) * dx);
						add_weighted(conv_data, raw_spectral_data[j+1], 0.5 * (w1 / W) * dx);

						if (simulation_.use_sfi)
						{
							double weight0 = 0.5 * (w0 / W) * dx;
							double weight1 = 0.5 * (w1 / W) * dx;

							for(size_t k = 0; k < sfi_ref_size; ++k)
							{
								result_.sfi_reflectance[i * sfi_ref_size + k] += raw_sfi_ref[j * sfi_ref_size + k] * weight0 + raw_sfi_ref[(j+1) * sfi_ref_size + k] * weight1;
							}

							for(size_t k = 0; k < sfi_thm_size; ++k)
							{
								result_.sfi_thermal_emission[i * sfi_thm_size + k] += raw_sfi_thm[j * sfi_thm_size + k] * weight0 + raw_sfi_thm[(j+1) * sfi_thm_size + k] * weight1;
							}
						}
					}
				}

				result_.spectral_data[i] = conv_data;
			}
		}
	}
	else if (simulation_.run_mode == RunMode::Adjoint)
	{
		std::vector<core::AtmosphereSensitivity> raw_jacobians(n_spectral_calc);
		
		if (!simulation_.adjoint_source_filepath.empty() && !has_custom_adjoint_source_)
		{
			PAAD_INFO("Importing adjoint source from: " << simulation_.adjoint_source_filepath);
			auto [F_mode_list, emission_list] = io::importAdjointSourceNetCDF(simulation_.adjoint_source_filepath, geometry_, spectral_, simulation_.polarization_mode);
			setAdjointSourceFourier(F_mode_list, emission_list);
		}

		#pragma omp parallel for num_threads(simulation_.n_parallel_spectral)
		for(int j = 0; j < n_spectral_calc; ++j)
		{
			double spectral = spectral_.calculation_grid[j];
			auto state = radiative_transfer_solver_.computeForwardState(atmosphere_model_, spectral, spectral_.dimension, initial_layer);

			core::RadiativeLayer adj_source;
			int n_stokes = (simulation_.polarization_mode == PolarizationMode::Scalar) ? 1 : ((simulation_.polarization_mode == PolarizationMode::Linear) ? 3 : 4);
			adj_source.resize(geometry_.Ntheta, geometry_.M, n_stokes);
			adj_source.optical_thickness = 0.0;
			adj_source.n_doubling = 0;
			adj_source.is_surface = false;

			std::vector<core::OpticalSensitivity> opt_sens;

			if (simulation_.use_sfi)
			{
				auto fwd_field = radiative_transfer_solver_.computeInternalField(state);
				sfi::SFIIntegrator integrator(geometry_, n_stokes, simulation_.n_parallel_fourier);

				double mu_obs = simulation_.sfi_geometry.mu_obs[0];
				double mu_0   = simulation_.sfi_geometry.mu_0[0];
				double dphi   = simulation_.sfi_geometry.dphi[0];

				Eigen::VectorXd grad_thm = has_custom_sfi_gradient_ ? sfi_grad_I_thm_[j] : Eigen::VectorXd::Ones(n_stokes);
				Eigen::MatrixXd grad_ref = has_custom_sfi_gradient_ ? sfi_grad_R_mat_[j] : Eigen::MatrixXd::Ones(n_stokes, n_stokes);

				auto sfi_sens = integrator.computeAdjoint(fwd_field, state.initial_optical_layers, grad_thm, grad_ref, mu_obs, mu_0, dphi);
				
				opt_sens = radiative_transfer_solver_.computeAdjointMonochrome(state, adj_source, &fwd_field, &sfi_sens.adj_field);

				for (size_t l = 0; l < opt_sens.size(); ++l)
				{
					opt_sens[l].optical_thickness += sfi_sens.direct_gradients[l].optical_thickness;
					opt_sens[l].single_scattering_albedo += sfi_sens.direct_gradients[l].single_scattering_albedo;
					opt_sens[l].planck_function += sfi_sens.direct_gradients[l].planck_function;
				}
			}
			else
			{
				if (has_custom_adjoint_source_)
				{
					if (!requires_convolution)
					{
						if (j < adjoint_sources_.size())
						{
							adj_source = adjoint_sources_[j];
						}
					}
					else 
					{
						for(auto& m : adj_source.reflectance_m_top)
						{
							m.setZero();
						}

						if(adj_source.source_up.size() > 0)
						{
							adj_source.source_up.setZero();
						}
						
						for(int i = 0; i < n_out; ++i)
						{
							if (i < adjoint_sources_.size() && norm_weights[i][j] > 0.0)
							{
								double w = norm_weights[i][j]; 

								for(size_t m = 0; m < adj_source.reflectance_m_top.size(); ++m)
								{
									adj_source.reflectance_m_top[m] += adjoint_sources_[i].reflectance_m_top[m] * w;
								}

								if(adj_source.source_up.size() > 0)
								{
									adj_source.source_up += adjoint_sources_[i].source_up * w;
								}
							}
						}
					}
				}
				else
				{
					adj_source.reflectance_m_top[0].setOnes(); 
					if(adj_source.source_up.size() > 0) adj_source.source_up.setOnes();
				}

				opt_sens = radiative_transfer_solver_.computeAdjointMonochrome(state, adj_source);
			}

			core::OpticalLayer surface_opt;
			core::OpticalSensitivity surface_sens;
			std::vector<core::OpticalSensitivity> atmos_opt_sens;

			if (state.offset == 1)
			{ 
				surface_opt = state.initial_optical_layers[0];
				surface_sens = opt_sens[0];
				atmos_opt_sens.assign(opt_sens.begin() + 1, opt_sens.end());
			}
			else
			{
				atmos_opt_sens = opt_sens;
			}

			raw_jacobians[j] = core::computeAtmosphericJacobian(atmosphere_model_, atmos_opt_sens, surface_opt, surface_sens, spectral, spectral_.dimension, simulation_.n_parallel_fourier);
		}

		auto clear_jac = [](core::AtmosphereSensitivity& jac)
		{
			for(auto& v : jac.temperature)
			{
				v = 0.0;
			}

			for(auto& v : jac.pressure)
			{
				v = 0.0;
			}

			for(auto& v : jac.number_density)
			{
				v = 0.0;
			}

			jac.surface.albedo = 0.0; jac.surface.emissivity = 0.0; jac.surface.temperature = 0.0;

			for(auto& sp : jac.species)
			{
				for(auto& v : sp.number_density) v = 0.0;
				for(auto& v : sp.mixing_ratio) v = 0.0;
				
				if(std::holds_alternative<core::ConstantAbsorptionSensitivity>(sp.absorption))
				{
					for (auto& v : std::get<core::ConstantAbsorptionSensitivity>(sp.absorption).cross_section) v = 0.0;
				}

				if(std::holds_alternative<core::HitranAbsorptionSensitivity>(sp.absorption))
				{
					for(auto& v : std::get<core::HitranAbsorptionSensitivity>(sp.absorption).abundance) v = 0.0;
					for(auto& v : std::get<core::HitranAbsorptionSensitivity>(sp.absorption).scalar) v = 0.0;
				}

				if(std::holds_alternative<core::RayleighSensitivity>(sp.scattering))
				{
					auto& ray = std::get<core::RayleighSensitivity>(sp.scattering);
					for (auto& v : ray.refractive_index) v = 0.0;
					for (auto& v : ray.depolarization_factor) v = 0.0;
					for (auto& v : ray.number_density_reference) v = 0.0;
				}

				if(std::holds_alternative<core::MieSensitivity>(sp.scattering))
				{
					auto& mie = std::get<core::MieSensitivity>(sp.scattering);
					for (auto& v : mie.refractive_index_real) v = 0.0;
					for (auto& v : mie.refractive_index_imag) v = 0.0;

					if(std::holds_alternative<core::DeltaSensitivity>(mie.size_distribution))
					{
						for (auto& v : std::get<core::DeltaSensitivity>(mie.size_distribution).r) v = 0.0;
					}

					if(std::holds_alternative<core::LogNormalSensitivity>(mie.size_distribution))
					{
						auto& d = std::get<core::LogNormalSensitivity>(mie.size_distribution);
						for (auto& v : d.r_g) v = 0.0;
						for (auto& v : d.sigma_g) v = 0.0;
					}

					if(std::holds_alternative<core::RectangularSensitivity>(mie.size_distribution))
					{
						auto& d = std::get<core::RectangularSensitivity>(mie.size_distribution);
						for (auto& v : d.r_mean) v = 0.0;
						for (auto& v : d.width) v = 0.0;
					}

					if(std::holds_alternative<core::GammaSensitivity>(mie.size_distribution))
					{
						auto& d = std::get<core::GammaSensitivity>(mie.size_distribution);
						for (auto& v : d.a) v = 0.0;
						for (auto& v : d.b) v = 0.0;
					}

					if(std::holds_alternative<core::ModifiedGammaSensitivity>(mie.size_distribution))
					{
						auto& d = std::get<core::ModifiedGammaSensitivity>(mie.size_distribution);
						for (auto& v : d.r_c) v = 0.0;
						for (auto& v : d.alpha) v = 0.0;
						for (auto& v : d.gamma) v = 0.0;
					}

					if(std::holds_alternative<core::PowerLawSensitivity>(mie.size_distribution))
					{
						auto& d = std::get<core::PowerLawSensitivity>(mie.size_distribution);
						for (auto& v : d.delta) v = 0.0;
						for (auto& v : d.r1) v = 0.0;
						for (auto& v : d.r2) v = 0.0;
					}
				}

				if(std::holds_alternative<core::HenyeyGreensteinSensitivity>(sp.scattering))
				{
					auto& hg = std::get<core::HenyeyGreensteinSensitivity>(sp.scattering);
					for (auto& v : hg.cross_section) v = 0.0;
					for (auto& v : hg.asymmetry_factor) v = 0.0;
				}

				if(std::holds_alternative<core::ConstantScatteringSensitivity>(sp.scattering))
				{
					for (auto& v : std::get<core::ConstantScatteringSensitivity>(sp.scattering).cross_section) v = 0.0;
				}

				if(std::holds_alternative<core::ExternalScatteringSensitivity>(sp.scattering))
				{
					for (auto& v : std::get<core::ExternalScatteringSensitivity>(sp.scattering).cross_section) v = 0.0;
				}
			}
		};

		auto add_weighted_jac = [](core::AtmosphereSensitivity& dest, const core::AtmosphereSensitivity& src, double weight)
		{
			for(size_t i = 0; i < dest.temperature.size(); ++i)
			{
				dest.temperature[i] += src.temperature[i] * weight;
				dest.pressure[i] += src.pressure[i] * weight;
				dest.number_density[i] += src.number_density[i] * weight;
			}

			dest.surface.albedo += src.surface.albedo * weight;
			dest.surface.emissivity += src.surface.emissivity * weight;
			dest.surface.temperature += src.surface.temperature * weight;

			for(size_t s = 0; s < dest.species.size(); ++s)
			{
				for(size_t i = 0; i < dest.species[s].number_density.size(); ++i)
				{
					dest.species[s].number_density[i] += src.species[s].number_density[i] * weight;
					dest.species[s].mixing_ratio[i] += src.species[s].mixing_ratio[i] * weight;
				}

				if(std::holds_alternative<core::ConstantAbsorptionSensitivity>(dest.species[s].absorption))
				{
					auto& d = std::get<core::ConstantAbsorptionSensitivity>(dest.species[s].absorption).cross_section;
					const auto& s_src = std::get<core::ConstantAbsorptionSensitivity>(src.species[s].absorption).cross_section;
					for(size_t i = 0; i < d.size(); ++i) d[i] += s_src[i] * weight;
				}

				if(std::holds_alternative<core::HitranAbsorptionSensitivity>(dest.species[s].absorption))
				{
					auto& d = std::get<core::HitranAbsorptionSensitivity>(dest.species[s].absorption);
					auto& s_src = std::get<core::HitranAbsorptionSensitivity>(src.species[s].absorption);
					for(size_t i = 0; i < d.abundance.size(); ++i) d.abundance[i] += s_src.abundance[i] * weight;
					for(size_t i = 0; i < d.scalar.size(); ++i) d.scalar[i] += s_src.scalar[i] * weight;
				}

				if(std::holds_alternative<core::RayleighSensitivity>(dest.species[s].scattering))
				{
					auto& d = std::get<core::RayleighSensitivity>(dest.species[s].scattering);
					auto& s_src = std::get<core::RayleighSensitivity>(src.species[s].scattering);
					for(size_t i = 0; i < d.refractive_index.size(); ++i) d.refractive_index[i] += s_src.refractive_index[i] * weight;
					for(size_t i = 0; i < d.depolarization_factor.size(); ++i) d.depolarization_factor[i] += s_src.depolarization_factor[i] * weight;
					for(size_t i = 0; i < d.number_density_reference.size(); ++i) d.number_density_reference[i] += s_src.number_density_reference[i] * weight;
				}

				if(std::holds_alternative<core::MieSensitivity>(dest.species[s].scattering))
				{
					auto& d = std::get<core::MieSensitivity>(dest.species[s].scattering);
					auto& s_src = std::get<core::MieSensitivity>(src.species[s].scattering);

					for(size_t i = 0; i < d.refractive_index_real.size(); ++i)
					{
						d.refractive_index_real[i] += s_src.refractive_index_real[i] * weight;
						d.refractive_index_imag[i] += s_src.refractive_index_imag[i] * weight;
					}
					
					if(std::holds_alternative<core::DeltaSensitivity>(d.size_distribution))
					{
						auto& dd = std::get<core::DeltaSensitivity>(d.size_distribution).r;
						const auto& ss = std::get<core::DeltaSensitivity>(s_src.size_distribution).r;
						for(size_t i = 0; i < dd.size(); ++i) dd[i] += ss[i] * weight;
					}
					
					if(std::holds_alternative<core::LogNormalSensitivity>(d.size_distribution))
					{
						auto& dd = std::get<core::LogNormalSensitivity>(d.size_distribution);
						auto& ss = std::get<core::LogNormalSensitivity>(s_src.size_distribution);
						for(size_t i = 0; i < dd.r_g.size(); ++i) {
							dd.r_g[i] += ss.r_g[i] * weight;
							dd.sigma_g[i] += ss.sigma_g[i] * weight;
						}
					}

					if(std::holds_alternative<core::RectangularSensitivity>(d.size_distribution))
					{
						auto& dd = std::get<core::RectangularSensitivity>(d.size_distribution);
						auto& ss = std::get<core::RectangularSensitivity>(s_src.size_distribution);
						for(size_t i = 0; i < dd.r_mean.size(); ++i) {
							dd.r_mean[i] += ss.r_mean[i] * weight;
							dd.width[i] += ss.width[i] * weight;
						}
					}

					if(std::holds_alternative<core::GammaSensitivity>(d.size_distribution))
					{
						auto& dd = std::get<core::GammaSensitivity>(d.size_distribution);
						auto& ss = std::get<core::GammaSensitivity>(s_src.size_distribution);
						for(size_t i = 0; i < dd.a.size(); ++i) {
							dd.a[i] += ss.a[i] * weight;
							dd.b[i] += ss.b[i] * weight;
						}
					}

					if(std::holds_alternative<core::ModifiedGammaSensitivity>(d.size_distribution))
					{
						auto& dd = std::get<core::ModifiedGammaSensitivity>(d.size_distribution);
						auto& ss = std::get<core::ModifiedGammaSensitivity>(s_src.size_distribution);
						for(size_t i = 0; i < dd.r_c.size(); ++i) {
							dd.r_c[i] += ss.r_c[i] * weight;
							dd.alpha[i] += ss.alpha[i] * weight;
							dd.gamma[i] += ss.gamma[i] * weight;
						}
					}

					if(std::holds_alternative<core::PowerLawSensitivity>(d.size_distribution))
					{
						auto& dd = std::get<core::PowerLawSensitivity>(d.size_distribution);
						auto& ss = std::get<core::PowerLawSensitivity>(s_src.size_distribution);
						for(size_t i = 0; i < dd.delta.size(); ++i) {
							dd.delta[i] += ss.delta[i] * weight;
							dd.r1[i] += ss.r1[i] * weight;
							dd.r2[i] += ss.r2[i] * weight;
						}
					}
				}

				if(std::holds_alternative<core::HenyeyGreensteinSensitivity>(dest.species[s].scattering))
				{
					auto& d = std::get<core::HenyeyGreensteinSensitivity>(dest.species[s].scattering);
					auto& s_src = std::get<core::HenyeyGreensteinSensitivity>(src.species[s].scattering);
					for(size_t i = 0; i < d.cross_section.size(); ++i) {
						d.cross_section[i] += s_src.cross_section[i] * weight;
						d.asymmetry_factor[i] += s_src.asymmetry_factor[i] * weight;
					}
				}

				if(std::holds_alternative<core::ConstantScatteringSensitivity>(dest.species[s].scattering))
				{
					auto& d = std::get<core::ConstantScatteringSensitivity>(dest.species[s].scattering).cross_section;
					const auto& s_src = std::get<core::ConstantScatteringSensitivity>(src.species[s].scattering).cross_section;
					for(size_t i = 0; i < d.size(); ++i) d[i] += s_src[i] * weight;
				}

				if(std::holds_alternative<core::ExternalScatteringSensitivity>(dest.species[s].scattering))
				{
					auto& d = std::get<core::ExternalScatteringSensitivity>(dest.species[s].scattering).cross_section;
					const auto& s_src = std::get<core::ExternalScatteringSensitivity>(src.species[s].scattering).cross_section;
					for(size_t i = 0; i < d.size(); ++i) d[i] += s_src[i] * weight;
				}
			}
		};

		jacobians_.resize(n_out);

		if (!requires_convolution)
		{
			jacobians_ = raw_jacobians;
		}
		else
		{
			PAAD_INFO("Applying convolution to Jacobians...");

			#pragma omp parallel for num_threads(simulation_.n_parallel_spectral)
			for (int i = 0; i < n_out; ++i)
			{
				core::AtmosphereSensitivity conv_jac = raw_jacobians[0]; 
				clear_jac(conv_jac);
				
				double W = 0.0;

				for (int j = 0; j < n_spectral_calc - 1; ++j)
				{
					double dx = spectral_.calculation_grid[j+1] - spectral_.calculation_grid[j];
					double w0 = norm_weights[i][j];
					double w1 = norm_weights[i][j+1];
					W += 0.5 * (w0 + w1) * dx;
				}

				for (int j = 0; j < n_spectral_calc - 1; ++j)
				{
					double dx = spectral_.calculation_grid[j+1] - spectral_.calculation_grid[j];
					
					if (has_custom_adjoint_source_ || simulation_.use_sfi)
					{
						add_weighted_jac(conv_jac, raw_jacobians[j], 0.5 * dx);
						add_weighted_jac(conv_jac, raw_jacobians[j+1], 0.5 * dx);
					}
					else
					{
						double w0 = norm_weights[i][j];
						double w1 = norm_weights[i][j+1];

						if (W > 0)
						{
							add_weighted_jac(conv_jac, raw_jacobians[j], 0.5 * (w0 / W) * dx);
							add_weighted_jac(conv_jac, raw_jacobians[j+1], 0.5 * (w1 / W) * dx);
						}
					}
				}

				jacobians_[i] = conv_jac;
			}
		}
	}
	
	PAAD_INFO("RT simulation finished successfully.");
}

void RadiativeTransfer::exportResult(void) const
{
	if (simulation_.run_mode == RunMode::Forward)
	{
		std::string out_path = (std::filesystem::path(simulation_.directory_name) / simulation_.result_name).string();
		PAAD_INFO("Exporting forward result to: " << out_path);
		io::exportResultNetCDF(result_, simulation_, atmosphere_model_, geometry_, spectral_);
	}
	else if (simulation_.run_mode == RunMode::Adjoint)
	{
		std::filesystem::path dir(simulation_.directory_name);
		if (!std::filesystem::exists(dir)) std::filesystem::create_directories(dir);
		
		std::string filename = (dir / ("jacobian_" + simulation_.result_name)).string();
		PAAD_INFO("Exporting adjoint Jacobians to: " << filename);
		io::exportJacobianNetCDF(filename, jacobians_, atmosphere_model_, spectral_);
	}
}

}
