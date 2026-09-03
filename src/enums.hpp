#pragma once

#include <string>
#include <map>

namespace paad
{

enum class GridType{Regular, GaussRadau};
inline const std::map<std::string, GridType> map_grid_type = {
	{"GAUSSRADAU",  GridType::GaussRadau},
	{"GAUSS_RADAU", GridType::GaussRadau},
	{"REGULAR",     GridType::Regular}
};

enum class SpectralCoordinateDimension{Wavenumber, Wavelength, Frequency};
inline const std::map<std::string, SpectralCoordinateDimension> map_spectral_coordinate_dimension = {
	{"WAVENUMBER", SpectralCoordinateDimension::Wavenumber},
	{"WAVELENGTH", SpectralCoordinateDimension::Wavelength},
	{"FREQUENCY",  SpectralCoordinateDimension::Frequency}
};

enum class SpectralCoordinateType{Monochrome, Spectrum, Bandpass};
inline const std::map<std::string, SpectralCoordinateType> map_spectral_coordinate_type = {
	{"MONOCHROME", SpectralCoordinateType::Monochrome},
	{"SPECTRUM",   SpectralCoordinateType::Spectrum},
	{"BANDPASS",   SpectralCoordinateType::Bandpass}
};

enum class SpectralGridSpacingType { Step, Count, ResolvingPower };
inline const std::map<std::string, SpectralGridSpacingType> map_spectral_grid_spacing_type = {
	{"STEP",       SpectralGridSpacingType::Step},
	{"COUNT",   SpectralGridSpacingType::Count},
	{"RESOLVING_POWER",     SpectralGridSpacingType::ResolvingPower},
	{"RESOLVINGPOWER",     SpectralGridSpacingType::ResolvingPower}
};

enum class ILSType{None, Gaussian, Lorentzian, Boxcar, Custom};
inline const std::map<std::string, ILSType> map_ils_type = {
	{"NONE",       ILSType::None},
	{"GAUSSIAN",   ILSType::Gaussian},
	{"BOXCAR",     ILSType::Boxcar},
	{"RECTANGULAR",     ILSType::Boxcar},
	{"RECTANGLE",     ILSType::Boxcar}
};

enum class VerticalProfileInterpolation{Exponential, Linear};
inline const std::map<std::string, VerticalProfileInterpolation> map_vertical_profile_interpolation = {
	{"EXPONENTIAL", VerticalProfileInterpolation::Exponential},
	{"LINEAR",      VerticalProfileInterpolation::Linear}
};

enum class ScatterType{Rayleigh, Mie, HenyeyGreenstein, Isotropic, External};
inline const std::map<std::string, ScatterType> map_scatter_type = {
	{"RAYLEIGH",          ScatterType::Rayleigh},
	{"MIE",               ScatterType::Mie},
	{"HENYEY_GREENSTEIN", ScatterType::HenyeyGreenstein},
	{"HENYEYGREENSTEIN",  ScatterType::HenyeyGreenstein},
	{"HG",                ScatterType::HenyeyGreenstein},
	{"ISOTROPIC",         ScatterType::Isotropic},
	{"ISOTROPICAL",       ScatterType::Isotropic},
	{"EXTERNAL",          ScatterType::External}
};

enum class SpeciesState{Molecule, Aerosol};
inline const std::map<std::string, SpeciesState> map_species_state = {
	{"MOLECULE", SpeciesState::Molecule},
	{"GAS", SpeciesState::Molecule},
	{"AEROSOL",  SpeciesState::Aerosol}
};

enum class SpeciesType{Absorber, Scatterer, Extinction};
inline const std::map<std::string, SpeciesType> map_species_type = {
	{"ABSORBER",   SpeciesType::Absorber},
	{"SCATTERER",  SpeciesType::Scatterer},
	{"EXTINCTION", SpeciesType::Extinction},
	{"BOTH",       SpeciesType::Extinction}
};

enum class ParticleSizeDistribution{Delta, Rectangular, LogNormal, Gamma, ModifiedGamma, PowerLaw};
inline const std::map<std::string, ParticleSizeDistribution> map_particle_size_distribution = {
	{"DELTA",          ParticleSizeDistribution::Delta},
	{"RECTANGULAR",    ParticleSizeDistribution::Rectangular},
	{"LOG_NORMAL",     ParticleSizeDistribution::LogNormal},
	{"LOGNORMAL",      ParticleSizeDistribution::LogNormal},
	{"GAMMA",          ParticleSizeDistribution::Gamma},
	{"MODIFIED_GAMMA", ParticleSizeDistribution::ModifiedGamma},
	{"MODIFIEDGAMMA",  ParticleSizeDistribution::ModifiedGamma},
	{"POWER_LAW",      ParticleSizeDistribution::PowerLaw},
	{"POWERLAW",      ParticleSizeDistribution::PowerLaw}
};

enum class VerticalProfileType{MixingRatio, NumberDensity, ColumnNumberDensity};
inline const std::map<std::string, VerticalProfileType> map_vertical_profile_type = {
	{"MIXING_RATIO",          VerticalProfileType::MixingRatio},
	{"NUMBER_DENSITY",        VerticalProfileType::NumberDensity},
	{"NUMBERDENSITY",         VerticalProfileType::NumberDensity},
	{"COLUMN_NUMBER_DENSITY", VerticalProfileType::ColumnNumberDensity},
	{"COLUMNNUMBERDENSITY",   VerticalProfileType::ColumnNumberDensity}
};

enum class CrossSectionType{External, Constant, HITRAN};
inline const std::map<std::string, CrossSectionType> map_cross_section_type = {
	{"EXTERNAL", CrossSectionType::External},
	{"CONSTANT", CrossSectionType::Constant},
	{"HITRAN",   CrossSectionType::HITRAN}
};

enum class RefractiveIndexType{Constant, Spectral, Vertical, SpectralVertical};
inline const std::map<std::string, RefractiveIndexType> map_refractive_index_type = {
	{"CONSTANT",          RefractiveIndexType::Constant},
	{"SPECTRAL",          RefractiveIndexType::Spectral},
	{"VERTICAL",          RefractiveIndexType::Vertical},
	{"SPECTRALVERTICAL",  RefractiveIndexType::SpectralVertical},
	{"SPECTRAL_VERTICAL", RefractiveIndexType::SpectralVertical},
	{"VERTICALSPECTRAL",  RefractiveIndexType::SpectralVertical},
	{"VERTICAL_SPECTRAL", RefractiveIndexType::SpectralVertical},
	{"BOTH",              RefractiveIndexType::SpectralVertical},
	{"COUPLED",           RefractiveIndexType::SpectralVertical},
	{"COUPLED2D",         RefractiveIndexType::SpectralVertical},
	{"COUPLED_2D",        RefractiveIndexType::SpectralVertical},
	{"2D",                RefractiveIndexType::SpectralVertical}
};

enum class VerticalTemperatureProfile{VIRA_EQUATOR, VIRA_45, VIRA_60, Table, External};
inline const std::map<std::string, VerticalTemperatureProfile> map_vertical_temperature_profile = {
	{"VIRA_EQUATOR", VerticalTemperatureProfile::VIRA_EQUATOR},
	{"VIRAEQUATOR", VerticalTemperatureProfile::VIRA_EQUATOR},
	{"VIRA_45",      VerticalTemperatureProfile::VIRA_45},
	{"VIRA45",      VerticalTemperatureProfile::VIRA_45},
	{"VIRA_60",      VerticalTemperatureProfile::VIRA_60},
	{"VIRA60",      VerticalTemperatureProfile::VIRA_60},
	{"TABLE",        VerticalTemperatureProfile::Table},
	{"EXTERNAL",     VerticalTemperatureProfile::External}
};

enum class VerticalPressureProfile{VIRA_EQUATOR, VIRA_45, VIRA_60, Table, Hydrostatic, External};
inline const std::map<std::string, VerticalPressureProfile> map_vertical_pressure_profile = {
	{"VIRA_EQUATOR", VerticalPressureProfile::VIRA_EQUATOR},
	{"VIRAEQUATOR", VerticalPressureProfile::VIRA_EQUATOR},
	{"VIRA_45",      VerticalPressureProfile::VIRA_45},
	{"VIRA45",      VerticalPressureProfile::VIRA_45},
	{"VIRA_60",      VerticalPressureProfile::VIRA_60},
	{"VIRA60",      VerticalPressureProfile::VIRA_60},
	{"TABLE",        VerticalPressureProfile::Table},
	{"HYDROSTATIC",  VerticalPressureProfile::Hydrostatic},
	{"EXTERNAL",     VerticalPressureProfile::External}
};

enum class VerticalNumberDensityProfile{VIRA_EQUATOR, VIRA_45, VIRA_60, Table, External, IdealGas};
inline const std::map<std::string, VerticalNumberDensityProfile> map_vertical_number_density_profile = {
	{"VIRA_EQUATOR",  VerticalNumberDensityProfile::VIRA_EQUATOR},
	{"VIRAEQUATOR",  VerticalNumberDensityProfile::VIRA_EQUATOR},
	{"VIRA_45",       VerticalNumberDensityProfile::VIRA_45},
	{"VIRA45",       VerticalNumberDensityProfile::VIRA_45},
	{"VIRA_60",       VerticalNumberDensityProfile::VIRA_60},
	{"VIRA60",       VerticalNumberDensityProfile::VIRA_60},
	{"TABLE",         VerticalNumberDensityProfile::Table},
	{"EXTERNAL",      VerticalNumberDensityProfile::External},
	{"IDEALGAS",      VerticalNumberDensityProfile::IdealGas},
	{"IDEAL_GAS",     VerticalNumberDensityProfile::IdealGas},
	{"IDEALGASLAW",   VerticalNumberDensityProfile::IdealGas},
	{"IDEAL_GASLAW",   VerticalNumberDensityProfile::IdealGas},
	{"IDEALGAS_LAW",   VerticalNumberDensityProfile::IdealGas},
	{"IDEAL_GAS_LAW", VerticalNumberDensityProfile::IdealGas}
};

enum class SurfaceType{NoSurface, Lambert};
inline const std::map<std::string, SurfaceType> map_surface_type = {
	{"NO_SURFACE", SurfaceType::NoSurface},
	{"NOSURFACE",  SurfaceType::NoSurface},
	{"NONE",       SurfaceType::NoSurface},
	{"LAMBERT",    SurfaceType::Lambert},
	{"LAMBERTIAN", SurfaceType::Lambert}
};

enum class WindowFunctionType{Gauss, Rectangle, External, Table};
inline const std::map<std::string, WindowFunctionType> map_window_function_type = {
	{"GAUSS",       WindowFunctionType::Gauss},
	{"GAUSSIAN",    WindowFunctionType::Gauss},
	{"RECTANGLE",   WindowFunctionType::Rectangle},
	{"RECTANGULAR", WindowFunctionType::Rectangle},
	{"TABLE",       WindowFunctionType::Table},
	{"EXTERNAL",    WindowFunctionType::External}
};

enum class InstrumentFunctionType{Delta, Gauss, Rectangle, External};
inline const std::map<std::string, InstrumentFunctionType> map_instrument_function_type = {
	{"NONE",        InstrumentFunctionType::Delta},
	{"DELTA",       InstrumentFunctionType::Delta},
	{"GAUSS",       InstrumentFunctionType::Gauss},
	{"GAUSSIAN",    InstrumentFunctionType::Gauss},
	{"RECTANGLE",   InstrumentFunctionType::Rectangle},
	{"RECTANGULAR", InstrumentFunctionType::Rectangle},
	{"EXTERNAL",    InstrumentFunctionType::External}
};

enum class IsotopologueType{All, Defined};
inline const std::map<std::string, IsotopologueType> map_isotopologue_type = {
	{"ALL",     IsotopologueType::All},
	{"HITRAN",  IsotopologueType::All},
	{"DEFINED", IsotopologueType::Defined},
	{"TABLE",   IsotopologueType::Defined}
};

enum class IsotopologueAbundanceType{HITRAN, Defined};
inline const std::map<std::string, IsotopologueAbundanceType> map_isotopologue_abundance_type = {
	{"HITRAN",  IsotopologueAbundanceType::HITRAN},
	{"DEFINED", IsotopologueAbundanceType::Defined},
	{"TABLE",   IsotopologueAbundanceType::Defined}
};

enum class DeltaApproximationType{Disable, d_m1, d_m2, FWHM, Cumulative};
inline const std::map<std::string, DeltaApproximationType> map_delta_approximation_type = {
	{"NONE",       DeltaApproximationType::Disable},
	{"DISABLE",    DeltaApproximationType::Disable},
	{"D_M1",       DeltaApproximationType::d_m1},
	{"DM1",        DeltaApproximationType::d_m1},
	{"DELTA_M_1",  DeltaApproximationType::d_m1},
	{"DELTA_M1",   DeltaApproximationType::d_m1},
	{"D_M2",       DeltaApproximationType::d_m2},
	{"DM2",        DeltaApproximationType::d_m2},
	{"DELTA_M_2",  DeltaApproximationType::d_m2},
	{"DELTA_M2",   DeltaApproximationType::d_m2},
	{"FWHM",       DeltaApproximationType::FWHM},
	{"CUMULATIVE", DeltaApproximationType::Cumulative}
};

enum class InCellSuperSamplingType{Enable, Disable};
inline const std::map<std::string, InCellSuperSamplingType> map_in_cell_super_sampling_type = {
	{"ENABLE",  InCellSuperSamplingType::Enable},
	{"DISABLE", InCellSuperSamplingType::Disable}
};

enum class PolarizationMode { Scalar, Linear, FullStokes };
inline const std::map<std::string, PolarizationMode> map_polarization_mode = {
	{"SCALAR", PolarizationMode::Scalar},
	{"DISABLE",  PolarizationMode::Scalar},
	{"LINEAR",  PolarizationMode::Linear},
	{"FULL_STOKES",  PolarizationMode::FullStokes},
	{"FULL",    PolarizationMode::FullStokes}
};

enum class RunMode { Forward, Adjoint };
inline const std::map<std::string, RunMode> map_run_mode = {
	{"FORWARD", RunMode::Forward},
	{"ADJOINT", RunMode::Adjoint},
	{"BACKWARD", RunMode::Adjoint},
	{"JACOBIAN", RunMode::Adjoint}
};

}
