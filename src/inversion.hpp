#pragma once

#include <vector>
#include <variant>

#include "enums.hpp"
#include "atmosphere.hpp"
#include "solver.hpp"
#include "adjoint.hpp"

namespace paad::core
{

struct DeltaSensitivity { std::vector<double> r; };
struct LogNormalSensitivity { std::vector<double> r_g; std::vector<double> sigma_g; };
struct RectangularSensitivity { std::vector<double> r_mean; std::vector<double> width; };
struct GammaSensitivity { std::vector<double> a; std::vector<double> b; };
struct ModifiedGammaSensitivity { std::vector<double> r_c; std::vector<double> alpha; std::vector<double> gamma; };
struct PowerLawSensitivity { std::vector<double> delta; std::vector<double> r1; std::vector<double> r2; };

using SizeDistributionSensitivity = std::variant<std::monostate, DeltaSensitivity, LogNormalSensitivity, RectangularSensitivity, GammaSensitivity, ModifiedGammaSensitivity, PowerLawSensitivity>;

struct ConstantAbsorptionSensitivity 
{ 
	std::vector<double> cross_section; 
};

struct ArrheniusAbsorptionSensitivity 
{ 
	;
};

struct HitranAbsorptionSensitivity 
{
	std::vector<double> abundance;
	std::vector<double> scalar;
};

using AbsorptionSensitivity = std::variant<std::monostate, ConstantAbsorptionSensitivity, ArrheniusAbsorptionSensitivity, HitranAbsorptionSensitivity >;

struct RayleighSensitivity 
{
	std::vector<double> refractive_index;
	std::vector<double> depolarization_factor;
	std::vector<double> number_density_reference;
};

struct MieSensitivity 
{
	std::vector<double> refractive_index_real;
	std::vector<double> refractive_index_imag;
	SizeDistributionSensitivity size_distribution;
};

struct HenyeyGreensteinSensitivity 
{
	std::vector<double> cross_section;
	std::vector<double> asymmetry_factor;
};

struct ConstantScatteringSensitivity 
{
	std::vector<double> cross_section;
};

struct ExternalScatteringSensitivity
{
	std::vector<double> cross_section;
};

using ScatteringSensitivity = std::variant<std::monostate, RayleighSensitivity, MieSensitivity, HenyeyGreensteinSensitivity, ConstantScatteringSensitivity, ExternalScatteringSensitivity>;

struct SpeciesSensitivity
{
	std::vector<double> number_density; 
	std::vector<double> mixing_ratio;

	AbsorptionSensitivity absorption;
	ScatteringSensitivity scattering;
};

struct SurfaceSensitivity
{
	double albedo = 0.0;
	double emissivity = 0.0;
	double temperature = 0.0;
};

struct AtmosphereSensitivity
{
	std::vector<double> temperature;
	std::vector<double> pressure;
	std::vector<double> number_density;
	
	std::vector<SpeciesSensitivity> species;
	
	SurfaceSensitivity surface;
};

AtmosphereSensitivity computeAtmosphericJacobian(const atmosphere::AtmosphereModel& atmos, const std::vector<OpticalSensitivity>& opt_sens_layers, const OpticalLayer& surface_opt_layer, const OpticalSensitivity& opt_sens_surface, double spectral, SpectralCoordinateDimension dim, int n_parallel);

}
