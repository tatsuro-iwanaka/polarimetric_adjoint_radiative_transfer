#pragma once

#include <vector>
#include <string>
#include <variant>
#include <Eigen/Dense>

#include "enums.hpp"

namespace paad::atmosphere
{
	struct AtmosphereModel;
}

namespace paad::core
{

class RadiativeLayer
{
	public:
		std::vector<Eigen::MatrixXd> reflectance_m_top;
		std::vector<Eigen::MatrixXd> reflectance_m_bottom;
		std::vector<Eigen::MatrixXd> transmittance_m_top;
		std::vector<Eigen::MatrixXd> transmittance_m_bottom;

		Eigen::VectorXd source_up;
		Eigen::VectorXd source_down;

		double optical_thickness;
		int n_doubling;
		
		bool is_surface = true;

		void resize(int n_theta, int n_mode, int n_stokes);
		void clear(void);
};

class OpticalLayer
{
	public:
		std::vector<Eigen::Matrix4d> scattering_matrix;
		std::vector<double> scattering_angle;
		std::vector<double> species_scattering_cross_section;
		std::vector<double> species_absorption_cross_section;
		std::vector<std::vector<Eigen::Matrix4d>> species_scattering_matrix;
		std::vector<bool> is_scattering_species;

		Eigen::Matrix4d surface_reflection_matrix;
		
		double optical_thickness;
		double single_scattering_albedo;
		double absorption_coefficient;
		double scattering_coefficient;
		double planck_function;
		double surface_albedo;
		double surface_emissivity;

		void clear(void);
};

struct MonochromeData
{
	std::vector<Eigen::MatrixXd> reflectance_m_top; 
	Eigen::VectorXd source_up;

	std::vector<std::vector<Eigen::Matrix4d>> scattering_matrix;
	std::vector<std::vector<std::vector<Eigen::Matrix4d>>> species_scattering_matrix;
	std::vector<std::vector<double>> species_absorption_cross_section; 
	std::vector<std::vector<double>> species_scattering_cross_section; 
	
	std::vector<double> absorption_coefficient; 
	std::vector<double> scattering_coefficient; 
	std::vector<double> single_scattering_albedo; 
	std::vector<double> optical_thickness; 
	std::vector<double> asymmetry_parameter; 
	std::vector<bool> is_scattering_species;
};

struct RadiativeTransferResult
{
	std::vector<MonochromeData> spectral_data;
	std::vector<double> theta_e;
	std::vector<double> theta_i;
	std::vector<double> phi;
	std::vector<double> altitude; 
	std::vector<double> altitude_top; 
	std::vector<double> altitude_bottom; 
	std::vector<double> physical_thickness; 
	std::vector<double> temperature; 
	std::vector<double> pressure; 
	std::vector<double> number_density; 
	
	std::vector<double> output_grid;

	int Ntheta;
	int Nphi;
	int Nmode;
};

struct InstrumentalFunction
{
	ILSType type = ILSType::None;
	double fwhm = 0.0;
	int oversample_factor = 3;
	double cutoff_sigma = 3.0;
};

struct MonochromeConfig
{
	double value = 0.0;
};

struct SpectrumConfig
{
	double min = 0.0;
	double max = 0.0;
	SpectralGridSpacingType spacing_type = SpectralGridSpacingType::Count;
	double step = 0.0;
	int count = 0;
	double resolving_power = 0.0;

	InstrumentalFunction ils;
};

struct BandpassConfig
{
	std::string filename;
	std::string varname_spectral;
	std::string varname_transmission;
	
	SpectralGridSpacingType spacing_type = SpectralGridSpacingType::Count;
	double step = 0.0;
	int count = 0;
	double resolving_power = 0.0;

	double cutoff_threshold = 1.0e-3;

	std::vector<double> grid;
	std::vector<double> values;
};

struct Spectral
{
	SpectralCoordinateType type;
	SpectralCoordinateDimension dimension; 

	std::variant<MonochromeConfig, SpectrumConfig, BandpassConfig> config;

	std::vector<double> output_grid;       
	std::vector<double> calculation_grid;  
};

struct Simulation
{
	std::string simulation_name;
	std::string directory_name;
	std::string result_name;
	std::string logfile_name;
	std::string adjoint_source_filepath;

	std::string hitran_filepath;

	double initial_optical_thickness = 1.0E-6;
	int n_scattering_angle = 5;
	int n_parallel = 1;
	
	PolarizationMode polarization_mode = PolarizationMode::FullStokes;

	RunMode run_mode = RunMode::Forward;
};

}
