#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <variant>
#include <optional>

#include "enums.hpp"
#include "hitran.hpp"

namespace paad::atmosphere
{

struct NetCDFCrossSectionConfig
{
	std::string filename;
	std::string var_name_spectral;
	std::string var_name_temperature;
	std::string var_name_cross_section;
	std::string unit_target_spectral;
	std::string unit_target_temperature;
	std::string unit_target_cross_section;
};

class ArrheniusCrossSectionModel
{
	private:
		std::vector<double> spectral_;
		std::vector<double> slope_;
		std::vector<double> intercept_;
		std::vector<bool> valid_flag_;
		
		template <typename T>
		T interpolateTemperature_(std::int64_t idx, T temperature) const
		{
			if (!valid_flag_[idx]) return T(0.0);
			using std::exp;
			return exp(intercept_[idx] + slope_[idx] / temperature);
		}

	public:
		void loadVectors(const std::vector<double>& spectral, const std::vector<double>& temperature, const std::vector<std::vector<double>>& cross_section);
		void loadNetCDF(const NetCDFCrossSectionConfig& config);
		
		template <typename T>
		T cross_section(double spectral, T temperature) const
		{
			if (spectral_.empty())
			{
				return T(0.0);
			}

			if (spectral < spectral_.front() || spectral > spectral_.back())
			{
				return T(0.0);
			}

			auto it = std::lower_bound(spectral_.begin(), spectral_.end(), spectral);

			if (it != spectral_.end() && *it == spectral)
			{
				std::int64_t idx = std::distance(spectral_.begin(), it);
				return interpolateTemperature_(idx, temperature);
			}

			auto it_next = it;
			auto it_prev = it - 1;

			std::int64_t idx_next = std::distance(spectral_.begin(), it_next);
			std::int64_t idx_prev = std::distance(spectral_.begin(), it_prev);

			T val_prev = interpolateTemperature_(idx_prev, temperature);
			T val_next = interpolateTemperature_(idx_next, temperature);

			if (!valid_flag_[idx_prev] && valid_flag_[idx_next])
			{
				return val_next;
			}

			if (valid_flag_[idx_prev] && !valid_flag_[idx_next])
			{
				return val_prev;
			}

			if (!valid_flag_[idx_prev] && !valid_flag_[idx_next])
			{
				return T(0.0);
			}

			double w_prev = *it_prev;
			double w_next = *it_next;
			double ratio = (spectral - w_prev) / (w_next - w_prev);

			return val_prev + ratio * (val_next - val_prev);
		}
};

struct HitranAbsorption
{
	int molecule_id;
	
	IsotopologueType isotopologue_type;
	std::vector<int> local_isotopologue_id;

	IsotopologueAbundanceType abundance_type;
	std::vector<double> abundance;

	std::vector<double> scalar;
	bool is_normalize;

	hitran::Isotopologue isotopologue_data;
	std::vector<hitran::Line> cached_lines;
};

struct DeltaDistribution
{
	double r;
};

struct LogNormalDistribution
{
	double r_g, sigma_g, r_min, r_max;
};

struct RectangularDistribution
{
	double r_mean, width;
};

struct GammaDistribution
{
	double a, b, r_min, r_max;
};

struct ModifiedGammaDistribution
{
	double r_c, alpha, gamma, r_min, r_max;
};

struct PowerLawDistribution
{
	double delta, r1, r2;
};

using SizeDistribution = std::variant<std::monostate, DeltaDistribution, LogNormalDistribution, RectangularDistribution, GammaDistribution, ModifiedGammaDistribution, PowerLawDistribution>;

struct ConstantAbsorption 
{ 
	double cross_section; 
};

struct ArrheniusAbsorption 
{ 
	NetCDFCrossSectionConfig config;
	ArrheniusCrossSectionModel model; 
};

using AbsorptionModel = std::variant<std::monostate, ConstantAbsorption, ArrheniusAbsorption, HitranAbsorption>;

struct NetCDFExternalScatteringConfig
{
	std::string filename;
	std::string var_name_scattering_angle = "scattering_angle";
	std::string var_name_scattering_matrix = "scattering_matrix";
};

class ExternalScatteringModel
{
	private:
		std::vector<double> scattering_angle_;
		std::vector<std::vector<std::vector<double>>> scattering_matrix_;

	public:
		void loadNetCDF(const NetCDFExternalScatteringConfig& config, PolarizationMode mode);
		double interpolateElement(double angle, int row, int col) const;
};

struct RayleighScattering 
{
	std::vector<double> refractive_index;
	double depolarization_factor;
	double number_density_reference;
};

struct MieScattering 
{
	std::vector<double> refractive_index;
	SizeDistribution size_distribution;
	std::vector<std::vector<double>> particle_size_distribution; 
	std::vector<std::vector<double>> weight_particle_size_distribution; 
};

struct HenyeyGreensteinScattering 
{
	double cross_section;
	double asymmetry_factor;
};

struct ConstantScattering 
{
	double cross_section;
};

struct ExternalScattering
{
	double cross_section;
	NetCDFExternalScatteringConfig config;
	ExternalScatteringModel model;
};

using ScatteringModel = std::variant<std::monostate, RayleighScattering, MieScattering, HenyeyGreensteinScattering, ConstantScattering, ExternalScattering>;

struct Species
{
	std::string name;
	SpeciesState species_state;
	SpeciesType species_type;
	ScatterType scatter_type;

	std::vector<double> vertical_number_density_profile;    
	std::vector<double> vertical_mixing_ratio_profile;  

	AbsorptionModel absorption_model;
	ScatteringModel scattering_model;
};

struct Layer
{
	double altitude_bottom;
	double altitude_top;
	double altitude;
	double temperature;
	double pressure;
	double number_density;
};

struct SurfaceModel
{
	double temperature;
	double albedo;
	double emissivity;
	SurfaceType type;
};

struct BackgroundGas
{
	std::vector<std::string> diluent_species;
	std::vector<double> diluent_ratio;
};

struct AtmosphereModel
{
	std::vector<Species> species;
	std::vector<Layer> layers;
	SurfaceModel surface;
	
	BackgroundGas background_gas;
};

void compileHitranData(AtmosphereModel& atmos, const std::string& hitran_filepath, double wavenumber_min, double wavenumber_max);

}
