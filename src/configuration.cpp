#include "configuration.hpp"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <iomanip>
#include <thread>

#include "units.hpp"
#include "enums.hpp"
#include "constants.hpp"
#include "string.hpp"
#include "rayleigh.hpp"
#include "mie.hpp"
#include "vira.hpp"

#include "logger.hpp"

namespace paad::configuration
{

ConfigurationData ConfigParser::import_json(const std::string& filepath)
{
	std::ifstream ifs(filepath);

	if (!ifs.is_open())
	{
		PAAD_ERROR("Cannot open JSON file: " << filepath);
		throw std::runtime_error("Cannot open JSON file: " + filepath);
	}
	
	std::stringstream buffer;
	buffer << ifs.rdbuf();
	root_ = utils::json::parse(buffer.str());

	ConfigurationData config;
	
	config.simulation = parseSimulation_();
	
	PAAD_INFO("Parsing JSON configuration file: " << filepath);

	config.spectral = parseSpectral_();
	config.geometry = parseGeometry_();
	config.atmosphere = parseAtmosphere_();

	PAAD_INFO("JSON configuration successfully loaded.");

	return config;
}

core::Simulation ConfigParser::parseSimulation_()
{
	core::Simulation sim;
	const auto& j_sim = root_["simulation"];

	sim.simulation_name = j_sim["name"].get_string();
	sim.directory_name = j_sim["directory"].get_string();
	
	std::filesystem::path p(sim.directory_name);

	if (!std::filesystem::exists(p))
	{
		std::filesystem::create_directories(p);
	}

	if (j_sim.has("logfile"))
	{
		sim.logfile_name = j_sim["logfile"].get_string();
		std::string log_path = (p / sim.logfile_name).string();
		paad::utils::Logger::getInstance().setLogFile(log_path);
		PAAD_INFO("Logger initialized. Simulation: " << sim.simulation_name);
	}

	if (j_sim.has("n_parallel_spectral") && j_sim.has("n_parallel_fourier"))
	{
		sim.n_parallel_spectral = static_cast<int>(j_sim["n_parallel_spectral"].get_number());
		sim.n_parallel_fourier  = static_cast<int>(j_sim["n_parallel_fourier"].get_number());
	}
	else
	{
		PAAD_WARN("Parallel configuration keys 'n_parallel_spectral' and/or 'n_parallel_fourier' not found. Defaulting to 1.");
		sim.n_parallel_spectral = 1;
		sim.n_parallel_fourier  = 1;
	}

	unsigned int hardware_threads = std::thread::hardware_concurrency();
	int total_requested = sim.n_parallel_spectral * sim.n_parallel_fourier;
	
	if (hardware_threads > 0 && total_requested > static_cast<int>(hardware_threads))
	{
		PAAD_WARN("Requested total threads (Spectral: " << sim.n_parallel_spectral << " * Fourier: " << sim.n_parallel_fourier << " = " << total_requested << ") exceed available hardware threads (" << hardware_threads << "). " << "This may cause thread thrashing and degrade performance.");
	}
	
	if (sim.n_parallel_spectral > 1 && sim.n_parallel_fourier > 1)
	{
		PAAD_WARN("Both spectral and fourier parallelisms are > 1. Nested parallelism requires OMP_NESTED=true to be fully active.");
	}

	sim.result_name = j_sim["result"].get_string();
	sim.n_scattering_angle = static_cast<int>(j_sim["n_scattering_angle"].get_number());
	sim.initial_optical_thickness = j_sim["initial_optical_thickness"].get_number();

	if (j_sim.has("polarization_mode"))
	{
		sim.polarization_mode = parseEnum_(j_sim["polarization_mode"].get_string(), map_polarization_mode);
	}
	else
	{
		sim.polarization_mode = PolarizationMode::Scalar;
	}

	if (j_sim.has("run_mode"))
	{
		sim.run_mode = parseEnum_(j_sim["run_mode"].get_string(), map_run_mode);
	}
	else
	{
		sim.run_mode = RunMode::Forward;
	}

	if (j_sim.has("hitran_filepath"))
	{
		sim.hitran_filepath = j_sim["hitran_filepath"].get_string();
	}

	if (j_sim.has("adjoint_source_filepath"))
	{
		sim.adjoint_source_filepath = j_sim["adjoint_source_filepath"].get_string();
	}

	if (j_sim.has("adjoint_source_filepath"))
	{
		sim.adjoint_source_filepath = j_sim["adjoint_source_filepath"].get_string();
	}

	if (j_sim.has("use_sfi"))
	{
		sim.use_sfi = j_sim["use_sfi"].get_bool();
	}

	if (j_sim.has("use_sfi"))
	{
		sim.use_sfi = j_sim["use_sfi"].get_bool();
	}

	if (sim.use_sfi && j_sim.has("sfi_geometry"))
	{
		const auto& j_sfi = j_sim["sfi_geometry"];

		auto parse_array_or_scalar = [](const auto& node) -> std::vector<double>
		{
			std::vector<double> vec;
			
			try
			{
				vec.push_back(node.get_number());
			}
			catch (const std::runtime_error&)
			{
				for (size_t i = 0; i < node.size(); ++i)
				{
					vec.push_back(node[i].get_number());
				}
			}

			return vec;
		};

		if (j_sfi.has("solar_flux"))
		{
			sim.sfi_geometry.solar_flux = parse_array_or_scalar(j_sfi["solar_flux"]);
		}

		if (j_sfi.has("mu_0"))
		{
			sim.sfi_geometry.mu_0 = parse_array_or_scalar(j_sfi["mu_0"]);
		}

		if (j_sfi.has("mu_obs"))
		{
			sim.sfi_geometry.mu_obs = parse_array_or_scalar(j_sfi["mu_obs"]);
		}
		
		if (j_sfi.has("dphi"))
		{
			sim.sfi_geometry.dphi = parse_array_or_scalar(j_sfi["dphi"]);
			
			for (auto& val : sim.sfi_geometry.dphi)
			{
				val *= std::numbers::pi / 180.0;
			}
		}
	}

	return sim;
}

core::Spectral ConfigParser::parseSpectral_()
{
	core::Spectral spec;
	const auto& j_spec = root_["simulation"]["spectral"];

	spec.type = parseEnum_(j_spec["type"].get_string(), map_spectral_coordinate_type);
	spec.dimension = parseEnum_(j_spec["dimension"].get_string(), map_spectral_coordinate_dimension);
	
	std::string unit = j_spec["unit"].get_string();
	double to_si = units::getUnitInfo(unit).to_si;

	if (spec.type == SpectralCoordinateType::Monochrome)
	{
		const auto& j_mono = j_spec["monochrome"];

		if (!j_mono.has("value"))
		{
			PAAD_ERROR("Monochrome type requires 'value'.");
			throw std::runtime_error("Monochrome type requires 'value'.");
		}
		
		core::MonochromeConfig mono;
		mono.value = j_mono["value"].get_number() * to_si;
		spec.config = mono;
	}
	else if (spec.type == SpectralCoordinateType::Spectrum)
	{
		const auto& j_s_conf = j_spec["spectrum"];

		if (!j_s_conf.has("min") || !j_s_conf.has("max"))
		{
			PAAD_ERROR("Spectrum type requires 'min' and 'max'.");
			throw std::runtime_error("Spectrum type requires 'min' and 'max'.");
		}

		core::SpectrumConfig s_conf;
		s_conf.min = j_s_conf["min"].get_number() * to_si;
		s_conf.max = j_s_conf["max"].get_number() * to_si;

		if (s_conf.min >= s_conf.max && (!j_s_conf.has("count") || j_s_conf["count"].get_number() != 1.0))
		{
			PAAD_ERROR("Spectrum min must be strictly less than max, unless count is 1.");
			throw std::runtime_error("Spectrum min must be strictly less than max, unless count is 1.");
		}

		if (j_s_conf.has("step"))
		{
			s_conf.spacing_type = SpectralGridSpacingType::Step;
			s_conf.step = j_s_conf["step"].get_number() * to_si;

			if (s_conf.step <= 0.0)
			{
				throw std::runtime_error("Spectrum step must be positive.");
			}
		}
		else if (j_s_conf.has("count"))
		{
			s_conf.spacing_type = SpectralGridSpacingType::Count;
			s_conf.count = static_cast<int>(j_s_conf["count"].get_number());

			if (s_conf.count < 1)
			{
				throw std::runtime_error("Spectrum count must be >= 1.");
			}
		}
		else if (j_s_conf.has("resolving_power")) 
		{
			s_conf.spacing_type = SpectralGridSpacingType::ResolvingPower;
			s_conf.resolving_power = j_s_conf["resolving_power"].get_number();

			if (s_conf.resolving_power <= 0.0)
			{
				throw std::runtime_error("Resolving power must be positive.");
			}
		}
		else
		{
			PAAD_ERROR("Spectrum requires one of 'step', 'count', or 'resolving_power'.");
			throw std::runtime_error("Spectrum requires one of 'step', 'count', or 'resolving_power'.");
		}

		if (j_spec.has("ils"))
		{
			const auto& j_ils = j_spec["ils"];
			s_conf.ils.type = parseEnum_(j_ils["type"].get_string(), map_ils_type);
			
			if (s_conf.ils.type != ILSType::None)
			{
				std::string ils_unit = j_ils.has("unit") ? j_ils["unit"].get_string() : unit;
				s_conf.ils.fwhm = j_ils["fwhm"].get_number() * units::getUnitInfo(ils_unit).to_si;
				
				if (s_conf.ils.fwhm <= 0.0)
				{
					throw std::runtime_error("ILS fwhm must be positive.");
				}

				if (j_ils.has("oversample_factor"))
				{
					s_conf.ils.oversample_factor = static_cast<int>(j_ils["oversample_factor"].get_number());
				}

				if (j_ils.has("cutoff_sigma"))
				{
					s_conf.ils.cutoff_sigma = j_ils["cutoff_sigma"].get_number();
				}
				
				if (s_conf.ils.oversample_factor < 1)
				{
					throw std::runtime_error("ILS oversample_factor must be >= 1.");
				}

				if (s_conf.ils.cutoff_sigma <= 0.0)
				{
					throw std::runtime_error("ILS cutoff_sigma must be positive.");
				}
			}
		}

		spec.config = s_conf;
	}
	else if (spec.type == SpectralCoordinateType::Bandpass)
	{
		if (!j_spec.has("filter"))
		{
			PAAD_ERROR("Bandpass type requires 'filter' block.");
			throw std::runtime_error("Bandpass type requires 'filter' block.");
		}
		
		const auto& j_filter = j_spec["filter"];
		core::BandpassConfig b_conf;

		b_conf.filename = j_filter["filename"].get_string();
		b_conf.varname_spectral = j_filter["varname_spectral"].get_string();
		b_conf.varname_transmission = j_filter["varname_transmission"].get_string();

		if (j_filter.has("cutoff_threshold"))
		{
			b_conf.cutoff_threshold = j_filter["cutoff_threshold"].get_number();

			if (b_conf.cutoff_threshold < 0.0 || b_conf.cutoff_threshold >= 1.0) 
			{
				throw std::runtime_error("Filter cutoff_threshold must be in [0.0, 1.0).");
			}
		}

		if (j_filter.has("step"))
		{
			b_conf.spacing_type = SpectralGridSpacingType::Step;
			b_conf.step = j_filter["step"].get_number() * to_si;

			if (b_conf.step <= 0.0)
			{
				throw std::runtime_error("Filter step must be positive.");
			}
		}
		else if (j_filter.has("count"))
		{
			b_conf.spacing_type = SpectralGridSpacingType::Count;
			b_conf.count = static_cast<int>(j_filter["count"].get_number());

			if (b_conf.count < 1)
			{
				throw std::runtime_error("Filter count must be >= 1.");
			}
		}
		else if (j_filter.has("resolving_power"))
		{
			b_conf.spacing_type = SpectralGridSpacingType::ResolvingPower;
			b_conf.resolving_power = j_filter["resolving_power"].get_number();

			if (b_conf.resolving_power <= 0.0)
			{
				throw std::runtime_error("Filter resolving power must be positive.");
			}
		}
		else
		{
			PAAD_ERROR("Bandpass filter requires one of 'step', 'count', or 'resolving_power'.");
			throw std::runtime_error("Bandpass filter requires one of 'step', 'count', or 'resolving_power'.");
		}

		spec.config = b_conf;
	}

	PAAD_INFO("Spectral configuration parsed. Type: " << enumToString_(spec.type, map_spectral_coordinate_type));
	return spec;
}

geometry::Geometry ConfigParser::parseGeometry_()
{
	const auto& j_grid = root_["simulation"]["grid"];
	auto grid_type = parseEnum_(j_grid["type"].get_string(), map_grid_type);
	int n_zenith = static_cast<int>(j_grid["n_zenith_angle"].get_number());

	PAAD_INFO("Geometry configuration parsed. Grid type: " << enumToString_(grid_type, map_grid_type) << ", n_zenith: " << n_zenith);

	return (grid_type == GridType::GaussRadau ? geometry::generateGeometryGaussRadau(n_zenith) : geometry::generateGeometryRegular(n_zenith));
}

atmosphere::AtmosphereModel ConfigParser::parseAtmosphere_()
{
	atmosphere::AtmosphereModel atom;
	const auto& j_surf = root_["surface"];
	
	atom.surface.type = parseEnum_(j_surf["type"].get_string(), map_surface_type);

	if (atom.surface.type != SurfaceType::NoSurface)
	{
		atom.surface.albedo = j_surf["albedo"].get_number();
		atom.surface.emissivity = j_surf["emissivity"].get_number();
		atom.surface.temperature = j_surf["temperature"].get_number();
	}

	const auto& j_atmos = root_["atmosphere"];

	if (j_atmos.has("background_gas"))
	{
		const auto& j_bg = j_atmos["background_gas"];

		if (j_bg.has("diluent_species") && j_bg.has("diluent_ratio"))
		{
			const auto& ds = j_bg["diluent_species"];
			const auto& dr = j_bg["diluent_ratio"];

			for (int k = 0; k < ds.size(); ++k)
			{
				atom.background_gas.diluent_species.push_back(ds[k].get_string());
				atom.background_gas.diluent_ratio.push_back(dr[k].get_number());
			}
		}
	}
	
	if (atom.background_gas.diluent_species.empty())
	{
		atom.background_gas.diluent_species.push_back("Air");
		atom.background_gas.diluent_ratio.push_back(1.0);
	}
	
	{
		const auto& j_layer = j_atmos["layering"];
		const auto& z_edge_json = j_layer["z_edge"];
		std::string z_unit = j_layer["unit"].get_string();
		
		int n_edge = z_edge_json.size();
		std::vector<double> z_edge(n_edge);

		for (int i = 0; i < n_edge; i++)
		{
			z_edge[i] = z_edge_json[i].get_number() * units::getUnitInfo(z_unit).to_si;
		}

		atom.layers.resize(n_edge - 1);

		for (int i = 0; i < n_edge - 1; ++i)
		{
			atom.layers[i].altitude = (z_edge[i] + z_edge[i + 1]) * 0.5;
			atom.layers[i].altitude_bottom = z_edge[i];
			atom.layers[i].altitude_top = z_edge[i + 1];
		}
	}

	{
		const auto& j_temp = j_atmos["temperature"];
		auto profile = parseEnum_(j_temp["profile"].get_string(), map_vertical_temperature_profile);

		if (profile == VerticalTemperatureProfile::Table)
		{
			const auto& table = j_temp["table"];

			if (table.size() != atom.layers.size())
			{
				PAAD_ERROR("Temperature table size mismatch.");
				throw std::runtime_error("altitude grid size mismatch.");
			}

			for (size_t i = 0; i < atom.layers.size(); i++) atom.layers[i].temperature = table[i].get_number();
		}
		else
		{
			const std::vector<std::vector<double>>* vira_ptr = nullptr;

			if (profile == VerticalTemperatureProfile::VIRA_EQUATOR)
			{
				vira_ptr = &atmosphere::vira::equator;
			}
			else if (profile == VerticalTemperatureProfile::VIRA_45)
			{
				vira_ptr = &atmosphere::vira::latitude_45;
			}
			else if (profile == VerticalTemperatureProfile::VIRA_60)
			{
				vira_ptr = &atmosphere::vira::latitude_60;
			}

			if (vira_ptr)
			{
				for(size_t i = 0; i < atom.layers.size(); ++i)
				{
					double z_target = atom.layers[i].altitude / 1000.0;
					const auto& vira = *vira_ptr;

					if (z_target <= vira.front()[0] || z_target >= vira.back()[0])
					{
						PAAD_ERROR("VIRA temperature range error at altitude: " << z_target << " km");
						throw std::runtime_error("VIRA range error.");
					}

					auto it = std::lower_bound(vira.begin(), vira.end(), z_target, [](const std::vector<double>& row, double val){return row[0] < val;});
					int idx = std::distance(vira.begin(), it) - 1;
					double w = (z_target - vira[idx][0]) / (vira[idx + 1][0] - vira[idx][0]);
					atom.layers[i].temperature = vira[idx][1] + w * (vira[idx + 1][1] - vira[idx][1]);
				}
			}
		}
	}

	// pressure
	{
		const auto& j_pres = j_atmos["pressure"];
		auto profile = parseEnum_(j_pres["profile"].get_string(), map_vertical_pressure_profile);

		if (profile == VerticalPressureProfile::Table)
		{
			std::string p_unit = j_pres["unit"].get_string();
			const auto& table = j_pres["table"];

			if (table.size() != atom.layers.size())
			{
				PAAD_ERROR("Pressure table size mismatch.");
				throw std::runtime_error("altitude grid size mismatch.");
			}

			for (size_t i = 0; i < atom.layers.size(); i++)
			{
				atom.layers[i].pressure = table[i].get_number() * units::getUnitInfo(p_unit).to_si;
			}
		}
		else
		{
			const std::vector<std::vector<double>>* vira_ptr = nullptr;

			if (profile == VerticalPressureProfile::VIRA_EQUATOR)
			{
				vira_ptr = &atmosphere::vira::equator;
			}
			else if (profile == VerticalPressureProfile::VIRA_45)
			{
				vira_ptr = &atmosphere::vira::latitude_45;
			}
			else if (profile == VerticalPressureProfile::VIRA_60)
			{
				vira_ptr = &atmosphere::vira::latitude_60;
			}

			if (vira_ptr)
			{
				for(size_t i = 0; i < atom.layers.size(); ++i)
				{
					double z_target = atom.layers[i].altitude / 1000.0;
					const auto& vira = *vira_ptr;

					if (z_target <= vira.front()[0] || z_target >= vira.back()[0])
					{
						PAAD_ERROR("VIRA pressure range error at altitude: " << z_target << " km");
						throw std::runtime_error("VIRA range error.");
					}

					auto it = std::lower_bound(vira.begin(), vira.end(), z_target, [](const std::vector<double>& row, double val){return row[0] < val;});
					int idx = std::distance(vira.begin(), it) - 1;
					double w = (z_target - vira[idx][0]) / (vira[idx + 1][0] - vira[idx][0]);
					double ln_p = std::log(vira[idx][2]) + w * (std::log(vira[idx + 1][2]) - std::log(vira[idx][2]));
					atom.layers[i].pressure = std::exp(ln_p) * units::scaleUnit("bar", "Pa");
				}
			}
		}
	}

	{
		const auto& j_nd = j_atmos["number_density"];
		auto profile = parseEnum_(j_nd["profile"].get_string(), map_vertical_number_density_profile);

		if (profile == VerticalNumberDensityProfile::Table)
		{
			std::string nd_unit = j_nd["unit"].get_string();
			const auto& table = j_nd["table"];

			if (table.size() != atom.layers.size())
			{
				PAAD_ERROR("Number density table size mismatch.");
				throw std::runtime_error("altitude grid size mismatch.");
			}

			if (units::getUnitInfo(nd_unit).dim == units::UnitDim::NumberDensity)
			{
				for (size_t i = 0; i < atom.layers.size(); i++)
				{
					atom.layers[i].number_density = table[i].get_number() * units::getUnitInfo(nd_unit).to_si;
				}
			}
			else if (units::getUnitInfo(nd_unit).dim == units::UnitDim::ColumnNumberDensity)
			{
				for (size_t i = 0; i < atom.layers.size(); i++)
				{
					double dz = atom.layers[i].altitude_top - atom.layers[i].altitude_bottom;
					atom.layers[i].number_density = table[i].get_number() * units::getUnitInfo(nd_unit).to_si / dz;
				}
			}
		}
		else if (profile == VerticalNumberDensityProfile::IdealGas)
		{
			for (size_t i = 0; i < atom.layers.size(); i++)
			{
				atom.layers[i].number_density = atom.layers[i].pressure * constants::AVOGADRO_CONSTANT / (constants::MOLAR_GAS_CONSTANT * atom.layers[i].temperature);
			}
		}
		else
		{
			const std::vector<std::vector<double>>* vira_ptr = nullptr;

			if (profile == VerticalNumberDensityProfile::VIRA_EQUATOR)
			{
				vira_ptr = &atmosphere::vira::equator;
			}
			else if (profile == VerticalNumberDensityProfile::VIRA_45)
			{
				vira_ptr = &atmosphere::vira::latitude_45;
			}
			else if (profile == VerticalNumberDensityProfile::VIRA_60)
			{
				vira_ptr = &atmosphere::vira::latitude_60;
			}

			if (vira_ptr)
			{
				for(size_t i = 0; i < atom.layers.size(); ++i)
				{
					double z_target = atom.layers[i].altitude / 1000.0;
					const auto& vira = *vira_ptr;

					if (z_target <= vira.front()[0] || z_target >= vira.back()[0])
					{
						PAAD_ERROR("VIRA number density range error at altitude: " << z_target << " km");
						throw std::runtime_error("VIRA range error.");
					}

					auto it = std::lower_bound(vira.begin(), vira.end(), z_target, [](const std::vector<double>& row, double val){return row[0] < val;});
					int idx = std::distance(vira.begin(), it) - 1;
					double w = (z_target - vira[idx][0]) / (vira[idx + 1][0] - vira[idx][0]);
					double ln_n = std::log(vira[idx][3]) + w * (std::log(vira[idx + 1][3]) - std::log(vira[idx][3]));
					atom.layers[i].number_density = std::exp(ln_n);
				}
			}
		}
	}

	{
		const auto& j_species = j_atmos["species"];
		atom.species.resize(j_species.size());

		for (size_t i = 0; i < j_species.size(); i++)
		{
			const auto& j_s = j_species[i];
			atmosphere::Species s;
			s.name = j_s["name"].get_string();
			s.species_state = parseEnum_(j_s["state"].get_string(), map_species_state);
			s.species_type = parseEnum_(j_s["type"].get_string(), map_species_type);

			const auto& j_vp = j_s["vertical_profile"];
			std::string unit_vp = j_vp["unit"].get_string();
			const auto& table_vp = j_vp["table"];
			
			if (table_vp.size() != atom.layers.size())
			{
				PAAD_ERROR("Species vertical profile size mismatch for species: " << s.name);
				throw std::runtime_error("altitude grid size mismatch (species " + std::to_string(i) + ").");
			}

			s.vertical_number_density_profile.resize(atom.layers.size());
			s.vertical_mixing_ratio_profile.resize(atom.layers.size());

			if (units::getUnitInfo(unit_vp).dim == units::UnitDim::NumberDensity)
			{
				for (size_t j = 0; j < atom.layers.size(); j++)
				{
					s.vertical_number_density_profile[j] = table_vp[j].get_number() * units::getUnitInfo(unit_vp).to_si;
					s.vertical_mixing_ratio_profile[j] = s.vertical_number_density_profile[j] / atom.layers[j].number_density;
				}
			}
			else if (units::getUnitInfo(unit_vp).dim == units::UnitDim::ColumnNumberDensity)
			{
				for (size_t j = 0; j < atom.layers.size(); j++)
				{
					double dz = atom.layers[j].altitude_top - atom.layers[j].altitude_bottom;
					s.vertical_number_density_profile[j] = table_vp[j].get_number() * units::getUnitInfo(unit_vp).to_si / dz;
					s.vertical_mixing_ratio_profile[j] = s.vertical_number_density_profile[j] / atom.layers[j].number_density;
				}
			}
			else if (units::getUnitInfo(unit_vp).dim == units::UnitDim::Dimensionless)
			{
				for (size_t j = 0; j < atom.layers.size(); j++)
				{
					s.vertical_mixing_ratio_profile[j] = table_vp[j].get_number() * units::getUnitInfo(unit_vp).to_si;
					s.vertical_number_density_profile[j] = atom.layers[j].number_density * s.vertical_mixing_ratio_profile[j];
				}
			}

			if (s.species_type == SpeciesType::Absorber || s.species_type == SpeciesType::Extinction)
			{
				const auto& j_abs = j_s["absorption_cross_section"];
				CrossSectionType abs_type = CrossSectionType::Constant;

				if (j_abs.has("type"))
				{
					abs_type = parseEnum_(j_abs["type"].get_string(), map_cross_section_type);
				}
				else if (j_abs.has("molecule_id"))
				{
					abs_type = CrossSectionType::HITRAN;
				}
				else if (j_abs.has("filename"))
				{
					abs_type = CrossSectionType::External;
				}

				if (abs_type == CrossSectionType::Constant)
				{
					atmosphere::ConstantAbsorption abs_model;
					std::string unit = j_abs.has("unit") ? j_abs["unit"].get_string() : "m2";
					abs_model.cross_section = j_abs["value"].get_number() * units::getUnitInfo(unit).to_si;
					s.absorption_model = abs_model;
				}
				else if (abs_type == CrossSectionType::External)
				{
					atmosphere::ArrheniusAbsorption abs_model;
					abs_model.config.filename = j_abs["filename"].get_string();
					abs_model.config.var_name_spectral = j_abs["varname_spectral"].get_string();
					abs_model.config.var_name_temperature = j_abs["varname_temperature"].get_string();
					abs_model.config.var_name_cross_section = j_abs["varname_cross_section"].get_string();
					abs_model.model.loadNetCDF(abs_model.config);
					s.absorption_model = abs_model;
				}
				else if (abs_type == CrossSectionType::HITRAN)
				{
					atmosphere::HitranAbsorption abs_model;
					abs_model.molecule_id = static_cast<int>(j_abs["molecule_id"].get_number());
					
					if (j_abs.has("isotopologue_type"))
					{
						abs_model.isotopologue_type = parseEnum_(j_abs["isotopologue_type"].get_string(), map_isotopologue_type);
					}
					else
					{
						abs_model.isotopologue_type = IsotopologueType::All;
					}

					if (abs_model.isotopologue_type == IsotopologueType::Defined && j_abs.has("local_isotopologue_id"))
					{
						const auto& loc_ids = j_abs["local_isotopologue_id"];
						for (int k = 0; k < loc_ids.size(); ++k)
						{
							abs_model.local_isotopologue_id.push_back(static_cast<int>(loc_ids[k].get_number()));
						}
					}

					if (j_abs.has("abundance_type"))
					{
						abs_model.abundance_type = parseEnum_(j_abs["abundance_type"].get_string(), map_isotopologue_abundance_type);
					}
					else
					{
						abs_model.abundance_type = IsotopologueAbundanceType::HITRAN;
					}

					if (abs_model.abundance_type == IsotopologueAbundanceType::Defined && j_abs.has("abundance"))
					{
						const auto& abds = j_abs["abundance"];

						for (int k = 0; k < abds.size(); ++k)
						{
							abs_model.abundance.push_back(abds[k].get_number());
						}
					}

					if (j_abs.has("scalar"))
					{
						const auto& scs = j_abs["scalar"];

						for (int k = 0; k < scs.size(); ++k)
						{
							abs_model.scalar.push_back(scs[k].get_number());
						}
					}

					if (j_abs.has("is_normalize"))
					{
						abs_model.is_normalize = j_abs["is_normalize"].get_bool();
					}
					else
					{
						abs_model.is_normalize = false;
					}

					s.absorption_model = abs_model;
				}
			}
			else
			{
				s.absorption_model = std::monostate{};
			}

			if (s.species_type == SpeciesType::Scatterer || s.species_type == SpeciesType::Extinction)
			{
				s.scatter_type = parseEnum_(j_s["scatter_type"].get_string(), map_scatter_type);

				if (s.scatter_type == ScatterType::Rayleigh)
				{
					atmosphere::RayleighScattering rayleigh;
					rayleigh.refractive_index.resize(1);
					rayleigh.refractive_index[0] = j_s["refractive_index"].get_number();
					rayleigh.depolarization_factor = j_s["depolarization_factor"].get_number();
					
					const auto& j_ndr = j_s["number_density_reference"];
					std::string unit_ndr = j_ndr["unit"].get_string();
					rayleigh.number_density_reference = j_ndr["value"].get_number() * units::getUnitInfo(unit_ndr).to_si;

					s.scattering_model = rayleigh;
				}
				else if (s.scatter_type == ScatterType::Mie)
				{
					atmosphere::MieScattering mie;
					const auto& j_sd = j_s["size_distribution"];
					auto dist_func = parseEnum_(j_sd["function"].get_string(), map_particle_size_distribution);
					std::string unit_input_radius = j_sd["unit"].get_string();
					int count_radius = static_cast<int>(j_sd["n_sampling"].get_number());

					if (dist_func == ParticleSizeDistribution::Delta)
					{
						atmosphere::DeltaDistribution dist;
						dist.r = j_sd["r"].get_number() * units::getUnitInfo(unit_input_radius).to_si;
						mie.size_distribution = dist;
						mie.particle_size_distribution = {{dist.r, 1.0}};
						mie.weight_particle_size_distribution = {{dist.r, 1.0}};
					}
					else if (dist_func == ParticleSizeDistribution::Rectangular)
					{
						atmosphere::RectangularDistribution dist;
						dist.r_mean = j_sd["r_mean"].get_number() * units::getUnitInfo(unit_input_radius).to_si;
						dist.width = j_sd["width"].get_number() * units::getUnitInfo(unit_input_radius).to_si;
						mie.size_distribution = dist;
						auto ps = mie::generateRectangularSizeDistribution(count_radius, dist.r_mean, dist.width);
						mie.particle_size_distribution = ps[0];
						mie.weight_particle_size_distribution = ps[1];
					}
					else if (dist_func == ParticleSizeDistribution::LogNormal)
					{
						atmosphere::LogNormalDistribution dist;
						dist.r_g = j_sd["r_g"].get_number() * units::getUnitInfo(unit_input_radius).to_si;
						dist.sigma_g = j_sd["sigma_g"].get_number();
						
						double r_min = j_sd.has("r_min") ? j_sd["r_min"].get_number() * units::getUnitInfo(unit_input_radius).to_si : 0.001e-6;
						double r_max = j_sd.has("r_max") ? j_sd["r_max"].get_number() * units::getUnitInfo(unit_input_radius).to_si : 100.0e-6;
						
						dist.r_min = r_min;
						dist.r_max = r_max;

						mie.size_distribution = dist;
						auto ps = mie::generateLogNormalSizeDistribution(count_radius, dist.r_g, dist.sigma_g, r_min, r_max);
						mie.particle_size_distribution = ps[0];
						mie.weight_particle_size_distribution = ps[1];
					}
					else if (dist_func == ParticleSizeDistribution::Gamma)
					{
						atmosphere::GammaDistribution dist;
						dist.a = j_sd["a"].get_number();
						dist.b = j_sd["b"].get_number() * units::getUnitInfo(unit_input_radius).to_si;
						dist.r_min = 0.0;
						dist.r_max = 0.0;
						mie.size_distribution = dist;
						auto ps = mie::generateGammaSizeDistribution(count_radius, dist.a, dist.b);
						mie.particle_size_distribution = ps[0];
						mie.weight_particle_size_distribution = ps[1];
					}
					else if (dist_func == ParticleSizeDistribution::ModifiedGamma)
					{
						atmosphere::ModifiedGammaDistribution dist;
						dist.r_c = j_sd["r_c"].get_number() * units::getUnitInfo(unit_input_radius).to_si;
						dist.alpha = j_sd["alpha"].get_number();
						dist.gamma = j_sd["gamma"].get_number();
						dist.r_min = 0.0;
						dist.r_max = 0.0;
						mie.size_distribution = dist;
						auto ps = mie::generateModifiedGammaSizeDistribution(count_radius, dist.r_c, dist.alpha, dist.gamma);
						mie.particle_size_distribution = ps[0];
						mie.weight_particle_size_distribution = ps[1];
					}
					else if (dist_func == ParticleSizeDistribution::PowerLaw)
					{
						atmosphere::PowerLawDistribution dist;
						dist.delta = j_sd["delta"].get_number();
						dist.r1 = j_sd["r_1"].get_number() * units::getUnitInfo(unit_input_radius).to_si;
						dist.r2 = j_sd["r_2"].get_number() * units::getUnitInfo(unit_input_radius).to_si;
						mie.size_distribution = dist;
						auto ps = mie::generatePowerLawSizeDistribution(count_radius, dist.delta, dist.r1, dist.r2);
						mie.particle_size_distribution = ps[0];
						mie.weight_particle_size_distribution = ps[1];
					}

					const auto& j_ri = j_s["refractive_index"];
					mie.refractive_index.resize(2);
					mie.refractive_index[0] = j_ri[0].get_number();
					mie.refractive_index[1] = j_ri[1].get_number();

					s.scattering_model = mie;
				}
				else if (s.scatter_type == ScatterType::HenyeyGreenstein)
				{
					atmosphere::HenyeyGreensteinScattering hg;
					const auto& j_sca = j_s["scattering_cross_section"];
					std::string unit = j_sca["unit"].get_string();
					hg.cross_section = j_sca["value"].get_number() * units::getUnitInfo(unit).to_si;
					
					if (j_s.has("asymmetry_factor"))
					{
						hg.asymmetry_factor = j_s["asymmetry_factor"].get_number();
					}
					else
					{
						hg.asymmetry_factor = 0.0;
					}

					s.scattering_model = hg;
				}
				else if (s.scatter_type == ScatterType::Isotropic)
				{
					atmosphere::ConstantScattering iso;
					const auto& j_sca = j_s["scattering_cross_section"];
					std::string unit = j_sca["unit"].get_string();
					iso.cross_section = j_sca["value"].get_number() * units::getUnitInfo(unit).to_si;

					s.scattering_model = iso;
				}
				else if (s.scatter_type == ScatterType::External)
				{
					atmosphere::ExternalScattering ext;
					
					const auto& j_sca = j_s["scattering_cross_section"];
					std::string unit = j_sca.has("unit") ? j_sca["unit"].get_string() : "m2";
					ext.cross_section = j_sca["value"].get_number() * units::getUnitInfo(unit).to_si;

					ext.config.filename = j_s["filename"].get_string();
					
					if (j_s.has("varname_scattering_angle"))
					{
						ext.config.var_name_scattering_angle = j_s["varname_scattering_angle"].get_string();
					}

					if (j_s.has("varname_scattering_matrix"))
					{
						ext.config.var_name_scattering_matrix = j_s["varname_scattering_matrix"].get_string();
					}

					PolarizationMode pol_mode = PolarizationMode::Scalar;

					if (root_["simulation"].has("polarization_mode"))
					{
						pol_mode = parseEnum_(root_["simulation"]["polarization_mode"].get_string(), map_polarization_mode);
					}

					ext.model.loadNetCDF(ext.config, pol_mode);
					
					s.scattering_model = ext;
				}
			}
			else
			{
				s.scattering_model = std::monostate{};
			}

			atom.species[i] = s;
		}
	}

	PAAD_INFO("Atmosphere parsed. Layers: " << atom.layers.size() << ", Species: " << atom.species.size());
	return atom;
}

void ConfigParser::export_json(const ConfigurationData& config, const std::string& filepath)
{
	std::ofstream ofs(filepath);

	if (!ofs.is_open())
	{
		PAAD_ERROR("Cannot open file for export: " << filepath);
		throw std::runtime_error("Cannot open file for export: " + filepath);
	}

	PAAD_INFO("Exporting JSON configuration to: " << filepath);

	ofs << std::scientific;
	ofs << std::setprecision(12);

	ofs << "{\n";
	ofs << "  \"simulation\": {\n";
	ofs << "    \"name\": \"" << config.simulation.simulation_name << "\",\n";
	ofs << "    \"directory\": \"" << config.simulation.directory_name << "\",\n";

	if (!config.simulation.logfile_name.empty())
	{
		ofs << "    \"logfile\": \"" << config.simulation.logfile_name << "\",\n";
	}

	ofs << "    \"run_mode\": \"" << enumToString_(config.simulation.run_mode, map_run_mode) << "\",\n";
	ofs << "    \"n_parallel_fourier\": " << config.simulation.n_parallel_fourier << ",\n";
	ofs << "    \"n_parallel_spectral\": " << config.simulation.n_parallel_spectral << ",\n";
	ofs << "    \"result\": \"" << config.simulation.result_name << "\",\n";
	ofs << "    \"n_scattering_angle\": " << config.simulation.n_scattering_angle << ",\n";
	ofs << "    \"initial_optical_thickness\": " << config.simulation.initial_optical_thickness << ",\n";
	ofs << "    \"polarization_mode\": \"" << enumToString_(config.simulation.polarization_mode, map_polarization_mode) << "\",\n";
	
	if (!config.simulation.hitran_filepath.empty())
	{
		ofs << "    \"hitran_filepath\": \"" << config.simulation.hitran_filepath << "\",\n";
	}

	if (!config.simulation.adjoint_source_filepath.empty())
	{
		ofs << "    \"adjoint_source_filepath\": \"" << config.simulation.adjoint_source_filepath << "\",\n";
	}
	
	std::string spec_unit = "m";
	if (config.spectral.dimension == SpectralCoordinateDimension::Wavenumber) spec_unit = "m-1";
	else if (config.spectral.dimension == SpectralCoordinateDimension::Frequency) spec_unit = "hz";

	ofs << "    \"spectral\": {\n";
	ofs << "      \"type\": \"" << enumToString_(config.spectral.type, map_spectral_coordinate_type) << "\",\n";
	ofs << "      \"dimension\": \"" << enumToString_(config.spectral.dimension, map_spectral_coordinate_dimension) << "\",\n";
	ofs << "      \"unit\": \"" << spec_unit << "\"";

	if (std::holds_alternative<core::MonochromeConfig>(config.spectral.config))
	{
		auto mono = std::get<core::MonochromeConfig>(config.spectral.config);
		ofs << ",\n      \"value\": " << mono.value << "\n";
	}
	else if (std::holds_alternative<core::SpectrumConfig>(config.spectral.config))
	{
		auto spec = std::get<core::SpectrumConfig>(config.spectral.config);
		ofs << ",\n      \"min\": " << spec.min << ",\n";
		ofs << "      \"max\": " << spec.max;

		if (spec.spacing_type == SpectralGridSpacingType::Step)
		{
			ofs << ",\n      \"step\": " << spec.step;
		}
		else if (spec.spacing_type == SpectralGridSpacingType::Count)
		{
			ofs << ",\n      \"count\": " << spec.count;
		}
		else if (spec.spacing_type == SpectralGridSpacingType::ResolvingPower)
		{
			ofs << ",\n      \"resolving_power\": " << spec.resolving_power;
		}

		if (spec.ils.type != ILSType::None)
		{
			ofs << ",\n      \"ils\": {\n";
			ofs << "        \"type\": \"" << enumToString_(spec.ils.type, map_ils_type) << "\",\n";
			ofs << "        \"fwhm\": " << spec.ils.fwhm << ",\n";
			ofs << "        \"oversample_factor\": " << spec.ils.oversample_factor << ",\n";
			ofs << "        \"cutoff_sigma\": " << spec.ils.cutoff_sigma << "\n";
			ofs << "      }\n";
		}
		else
		{
			ofs << "\n";
		}
	}
	else if (std::holds_alternative<core::BandpassConfig>(config.spectral.config))
	{
		auto bp = std::get<core::BandpassConfig>(config.spectral.config);
		ofs << ",\n      \"filter\": {\n";
		ofs << "        \"filename\": \"" << bp.filename << "\",\n";
		ofs << "        \"varname_spectral\": \"" << bp.varname_spectral << "\",\n";
		ofs << "        \"varname_transmission\": \"" << bp.varname_transmission << "\",\n";
		ofs << "        \"cutoff_threshold\": " << bp.cutoff_threshold;

		if (bp.spacing_type == SpectralGridSpacingType::Step)
		{
			ofs << ",\n        \"step\": " << bp.step << "\n";
		}
		else if (bp.spacing_type == SpectralGridSpacingType::Count)
		{
			ofs << ",\n        \"count\": " << bp.count << "\n";
		}
		else if (bp.spacing_type == SpectralGridSpacingType::ResolvingPower)
		{
			ofs << ",\n        \"resolving_power\": " << bp.resolving_power << "\n";
		}

		ofs << "      }\n";
	}

	ofs << "    },\n";

	ofs << "    \"grid\": {\n";
	ofs << "      \"type\": \"regular\",\n";
	ofs << "      \"n_zenith_angle\": 30\n";
	ofs << "    }\n";
	ofs << "  },\n";

	ofs << "  \"surface\": {\n";
	ofs << "    \"type\": \"" << enumToString_(config.atmosphere.surface.type, map_surface_type) << "\",\n";
	ofs << "    \"albedo\": " << config.atmosphere.surface.albedo << ",\n";
	ofs << "    \"emissivity\": " << config.atmosphere.surface.emissivity << ",\n";
	ofs << "    \"temperature\": " << config.atmosphere.surface.temperature << "\n";
	ofs << "  },\n";

	ofs << "  \"atmosphere\": {\n";

	ofs << "    \"background_gas\": {\n";
	ofs << "      \"diluent_species\": [";
	const auto& d_sp = config.atmosphere.background_gas.diluent_species;

	for (size_t k = 0; k < d_sp.size(); ++k)
	{
		ofs << "\"" << d_sp[k] << "\"";

		if (k + 1 < d_sp.size())
		{
			ofs << ", ";
		}
	}
	ofs << "],\n";
	ofs << "      \"diluent_ratio\": [";
	const auto& d_r = config.atmosphere.background_gas.diluent_ratio;

	for (size_t k = 0; k < d_r.size(); ++k)
	{
		ofs << d_r[k];

		if (k + 1 < d_r.size())
		{
			ofs << ", ";
		}
	}

	ofs << "]\n    },\n";

	ofs << "    \"layering\": {\n";
	ofs << "      \"unit\": \"m\",\n";
	ofs << "      \"z_edge\": [";

	for (size_t i = 0; i < config.atmosphere.layers.size(); ++i)
	{
		ofs << config.atmosphere.layers[i].altitude_bottom << ", ";
	}

	if (!config.atmosphere.layers.empty()) 
	{
		ofs << config.atmosphere.layers.back().altitude_top;
	}

	ofs << "]\n    },\n";

	auto writeProfile = [&](const std::string& name, const std::string& unit, auto extract_func)
	{
		ofs << "    \"" << name << "\": {\n";
		ofs << "      \"profile\": \"table\",\n";
		ofs << "      \"unit\": \"" << unit << "\",\n";
		ofs << "      \"table\": [";

		for (size_t i = 0; i < config.atmosphere.layers.size(); ++i)
		{
			ofs << extract_func(config.atmosphere.layers[i]);
			if (i + 1 < config.atmosphere.layers.size()) ofs << ", ";
		}

		ofs << "]\n    },\n";
	};

	writeProfile("temperature", "K", [](const atmosphere::Layer& l){ return l.temperature; });
	writeProfile("pressure", "Pa", [](const atmosphere::Layer& l){ return l.pressure; });
	writeProfile("number_density", "m-3", [](const atmosphere::Layer& l){ return l.number_density; });

	ofs << "    \"species\": [\n";
	for (size_t i = 0; i < config.atmosphere.species.size(); ++i)
	{
		const auto& sp = config.atmosphere.species[i];
		ofs << "      {\n";
		ofs << "        \"name\": \"" << sp.name << "\",\n";
		ofs << "        \"state\": \"" << enumToString_(sp.species_state, map_species_state) << "\",\n";
		ofs << "        \"type\": \"" << enumToString_(sp.species_type, map_species_type) << "\",\n";

		bool has_scatter = std::holds_alternative<atmosphere::RayleighScattering>(sp.scattering_model) || std::holds_alternative<atmosphere::MieScattering>(sp.scattering_model) || std::holds_alternative<atmosphere::HenyeyGreensteinScattering>(sp.scattering_model) || std::holds_alternative<atmosphere::ConstantScattering>(sp.scattering_model)|| std::holds_alternative<atmosphere::ExternalScattering>(sp.scattering_model);

		if (has_scatter)
		{
			ofs << "        \"scatter_type\": \"" << enumToString_(sp.scatter_type, map_scatter_type) << "\",\n";
		}

		ofs << "        \"vertical_profile\": {\n";
		ofs << "          \"unit\": \"dimensionless\",\n";
		ofs << "          \"table\": [";
		
		for (size_t j = 0; j < sp.vertical_mixing_ratio_profile.size(); ++j)
		{
			ofs << sp.vertical_mixing_ratio_profile[j];
			if (j + 1 < sp.vertical_mixing_ratio_profile.size()) ofs << ", ";
		}

		ofs << "]\n        }";

		if (std::holds_alternative<atmosphere::ConstantAbsorption>(sp.absorption_model))
		{
			auto abs = std::get<atmosphere::ConstantAbsorption>(sp.absorption_model);
			ofs << ",\n        \"absorption_cross_section\": { \"type\": \"constant\", \"value\": " << abs.cross_section << ", \"unit\": \"m2\" }";
		}
		else if (std::holds_alternative<atmosphere::ArrheniusAbsorption>(sp.absorption_model))
		{
			auto abs = std::get<atmosphere::ArrheniusAbsorption>(sp.absorption_model);
			ofs << ",\n        \"absorption_cross_section\": {\n";
			ofs << "          \"type\": \"external\",\n";
			ofs << "          \"filename\": \"" << abs.config.filename << "\",\n";
			ofs << "          \"varname_spectral\": \"" << abs.config.var_name_spectral << "\",\n";
			ofs << "          \"varname_temperature\": \"" << abs.config.var_name_temperature << "\",\n";
			ofs << "          \"varname_cross_section\": \"" << abs.config.var_name_cross_section << "\"\n";
			ofs << "        }";
		}
		else if (std::holds_alternative<atmosphere::HitranAbsorption>(sp.absorption_model))
		{
			auto abs = std::get<atmosphere::HitranAbsorption>(sp.absorption_model);
			ofs << ",\n        \"absorption_cross_section\": {\n";
			ofs << "          \"type\": \"hitran\",\n";
			ofs << "          \"molecule_id\": " << abs.molecule_id << ",\n";
			ofs << "          \"isotopologue_type\": \"" << enumToString_(abs.isotopologue_type, map_isotopologue_type) << "\",\n";

			if (abs.isotopologue_type == IsotopologueType::Defined && !abs.local_isotopologue_id.empty())
			{
				ofs << "          \"local_isotopologue_id\": [";

				for(size_t k = 0; k < abs.local_isotopologue_id.size(); ++k)
				{
					ofs << abs.local_isotopologue_id[k] << (k + 1 < abs.local_isotopologue_id.size() ? ", " : "");
				}

				ofs << "],\n";
			}

			ofs << "          \"abundance_type\": \"" << enumToString_(abs.abundance_type, map_isotopologue_abundance_type) << "\",\n";

			if (abs.abundance_type == IsotopologueAbundanceType::Defined && !abs.abundance.empty())
			{
				ofs << "          \"abundance\": [";

				for(size_t k = 0; k < abs.abundance.size(); ++k)
				{
					ofs << abs.abundance[k] << (k + 1 < abs.abundance.size() ? ", " : "");
				}

				ofs << "],\n";
			}

			ofs << "          \"is_normalize\": " << (abs.is_normalize ? "true" : "false");

			if (!abs.scalar.empty())
			{
				ofs << ",\n          \"scalar\": [";

				for(size_t k = 0; k < abs.scalar.size(); ++k)
				{
					ofs << abs.scalar[k] << (k + 1 < abs.scalar.size() ? ", " : "");
				}

				ofs << "]";
			}

			ofs << "\n        }";
		}

		if (std::holds_alternative<atmosphere::RayleighScattering>(sp.scattering_model))
		{
			auto ray = std::get<atmosphere::RayleighScattering>(sp.scattering_model);
			ofs << ",\n        \"refractive_index\": " << (ray.refractive_index.empty() ? 0.0 : ray.refractive_index[0]);
			ofs << ",\n        \"depolarization_factor\": " << ray.depolarization_factor;
			ofs << ",\n        \"number_density_reference\": { \"value\": " << ray.number_density_reference << ", \"unit\": \"m-3\" }";
		}
		else if (std::holds_alternative<atmosphere::ConstantScattering>(sp.scattering_model))
		{
			auto iso = std::get<atmosphere::ConstantScattering>(sp.scattering_model);
			ofs << ",\n        \"scattering_cross_section\": { \"value\": " << iso.cross_section << ", \"unit\": \"m2\" }";
		}
		else if (std::holds_alternative<atmosphere::HenyeyGreensteinScattering>(sp.scattering_model))
		{
			auto hg = std::get<atmosphere::HenyeyGreensteinScattering>(sp.scattering_model);
			ofs << ",\n        \"scattering_cross_section\": { \"value\": " << hg.cross_section << ", \"unit\": \"m2\" }";
			ofs << ",\n        \"asymmetry_factor\": " << hg.asymmetry_factor;
		}
		else if (std::holds_alternative<atmosphere::MieScattering>(sp.scattering_model))
		{
			auto mie = std::get<atmosphere::MieScattering>(sp.scattering_model);
			ofs << ",\n        \"refractive_index\": [" << mie.refractive_index[0] << ", " << mie.refractive_index[1] << "],\n";
			ofs << "        \"size_distribution\": {\n";
			ofs << "          \"unit\": \"m\",\n";
			ofs << "          \"n_sampling\": " << mie.particle_size_distribution.size() << ",\n";
			
			if (std::holds_alternative<atmosphere::DeltaDistribution>(mie.size_distribution))
			{
				auto d = std::get<atmosphere::DeltaDistribution>(mie.size_distribution);
				ofs << "          \"function\": \"delta\",\n";
				ofs << "          \"r\": " << d.r << "\n";
			}
			else if (std::holds_alternative<atmosphere::RectangularDistribution>(mie.size_distribution))
			{
				auto d = std::get<atmosphere::RectangularDistribution>(mie.size_distribution);
				ofs << "          \"function\": \"rectangular\",\n";
				ofs << "          \"r_mean\": " << d.r_mean << ",\n";
				ofs << "          \"width\": " << d.width << "\n";
			}
			else if (std::holds_alternative<atmosphere::LogNormalDistribution>(mie.size_distribution))
			{
				auto d = std::get<atmosphere::LogNormalDistribution>(mie.size_distribution);
				ofs << "          \"function\": \"log_normal\",\n";
				ofs << "          \"r_g\": " << d.r_g << ",\n";
				ofs << "          \"sigma_g\": " << d.sigma_g << ",\n";
				ofs << "          \"r_min\": " << d.r_min << ",\n";
				ofs << "          \"r_max\": " << d.r_max << "\n";
			}
			else if (std::holds_alternative<atmosphere::GammaDistribution>(mie.size_distribution))
			{
				auto d = std::get<atmosphere::GammaDistribution>(mie.size_distribution);
				ofs << "          \"function\": \"gamma\",\n";
				ofs << "          \"a\": " << d.a << ",\n";
				ofs << "          \"b\": " << d.b << "\n";
			}
			else if (std::holds_alternative<atmosphere::ModifiedGammaDistribution>(mie.size_distribution))
			{
				auto d = std::get<atmosphere::ModifiedGammaDistribution>(mie.size_distribution);
				ofs << "          \"function\": \"modified_gamma\",\n";
				ofs << "          \"r_c\": " << d.r_c << ",\n";
				ofs << "          \"alpha\": " << d.alpha << ",\n";
				ofs << "          \"gamma\": " << d.gamma << "\n";
			}
			else if (std::holds_alternative<atmosphere::PowerLawDistribution>(mie.size_distribution))
			{
				auto d = std::get<atmosphere::PowerLawDistribution>(mie.size_distribution);
				ofs << "          \"function\": \"power_law\",\n";
				ofs << "          \"delta\": " << d.delta << ",\n";
				ofs << "          \"r_1\": " << d.r1 << ",\n";
				ofs << "          \"r_2\": " << d.r2 << "\n";
			}

			ofs << "        }";
		}
		else if (std::holds_alternative<atmosphere::ExternalScattering>(sp.scattering_model))
		{
			auto ext = std::get<atmosphere::ExternalScattering>(sp.scattering_model);
			ofs << ",\n        \"scattering_cross_section\": { \"value\": " << ext.cross_section << ", \"unit\": \"m2\" }";
			ofs << ",\n        \"filename\": \"" << ext.config.filename << "\"";
			ofs << ",\n        \"varname_scattering_angle\": \"" << ext.config.var_name_scattering_angle << "\"";
			ofs << ",\n        \"varname_scattering_matrix\": \"" << ext.config.var_name_scattering_matrix << "\"";
		}

		ofs << "\n      }";

		if (i + 1 < config.atmosphere.species.size())
		{
			ofs << ",";
		}

		ofs << "\n";
	}

	ofs << "    ]\n";
	ofs << "  }\n";
	ofs << "}\n";
}

}
