#include "solver.hpp"

#include <cmath>
#include <numbers>
#include <algorithm>
#include <complex>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <random>
#include <sstream>
#include <omp.h>

#include "physics.hpp"
#include "forward.hpp"
#include "adjoint.hpp"
#include "mie.hpp"
#include "atmosphere.hpp"
#include "rayleigh.hpp"
#include "logger.hpp"

namespace paad::core
{

void RadiativeLayer::resize(int n_theta, int n_mode, int n_stokes)
{
	int dim = n_stokes * n_theta;
	reflectance_m_top.assign(n_mode + 1, Eigen::MatrixXd::Zero(dim, dim));
	reflectance_m_bottom.assign(n_mode + 1, Eigen::MatrixXd::Zero(dim, dim));
	transmittance_m_top.assign(n_mode + 1, Eigen::MatrixXd::Zero(dim, dim));
	transmittance_m_bottom.assign(n_mode + 1, Eigen::MatrixXd::Zero(dim, dim));

	source_up = Eigen::VectorXd::Zero(dim);
	source_down = Eigen::VectorXd::Zero(dim);
}

void RadiativeLayer::clear(void)
{
	reflectance_m_top.clear();
	reflectance_m_bottom.clear();
	transmittance_m_top.clear();
	transmittance_m_bottom.clear();
	source_up.resize(0);
	source_down.resize(0);
}

void OpticalLayer::clear(void)
{
	optical_thickness = 0.0;
	single_scattering_albedo = 0.0;
	absorption_coefficient = 0.0;
	scattering_coefficient = 0.0;
	planck_function = 0.0;

	scattering_angle.clear();
	scattering_matrix.clear();

	surface_albedo = 0.0;
	surface_emissivity = 0.0;
	surface_reflection_matrix = Eigen::Matrix4d::Zero();

	is_scattering_species.clear();
	species_scattering_cross_section.clear();
	species_absorption_cross_section.clear();

	for (auto& mat : species_scattering_matrix)
	{
		mat.clear();
	}

	species_scattering_matrix.clear();
}

double computeSimpsonIntegration(const std::vector<std::vector<double>>& f)
{
	int n = f.size();
	double result = 0.0;

	for(int i = 0; i < n; i++)
	{
		if(i == 0 || i == n - 1)
		{
			result += f[i][1];
		}
		else if(i % 2 == 0)
		{
			result += 2.0 * f[i][1];
		}
		else
		{
			result += 4.0 * f[i][1];
		}
	}

	return result * (f[1][0] - f[0][0]) / 3.0;
}

void normalizeScatteringMatrix(std::vector<Eigen::Matrix4d>& scattering_matrix)
{
	int n = scattering_matrix.size();

	if (n < 2)
	{
		return;
	}

	std::vector<std::vector<double>> integration_grid(n, std::vector<double>(2));
	double dtheta = std::numbers::pi / static_cast<double>(n - 1);

	for (int i = 0; i < n; ++i)
	{
		double theta = dtheta * static_cast<double>(i);
		double val = scattering_matrix[i](0, 0) * 2.0 * std::numbers::pi * std::sin(theta);
		integration_grid[i] = {theta, val};
	}

	double integral = computeSimpsonIntegration(integration_grid);
	double norm_factor = integral / (4.0 * std::numbers::pi);

	if (std::abs(norm_factor) > 1.0E-12)
	{
		for (int i = 0; i < n; ++i)
		{
			scattering_matrix[i] /= norm_factor;
		}
	}
}

Eigen::Matrix4d interpolateScatteringMatrix(const std::vector<Eigen::Matrix4d>& matrices, const std::vector<double>& angles, double target_angle)
{
	if (target_angle <= angles.front())
	{
		return matrices.front();
	}

	if (target_angle >= angles.back())
	{
		return matrices.back();
	}

	auto it = std::lower_bound(angles.begin(), angles.end(), target_angle);
	size_t idx = std::distance(angles.begin(), it);
	
	double t = (target_angle - angles[idx - 1]) / (angles[idx] - angles[idx - 1]);

	return matrices[idx - 1] + t * (matrices[idx] - matrices[idx - 1]);
}

std::string generateUniqueFilePath()
{
	thread_local std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count() + std::hash<std::thread::id>{}(std::this_thread::get_id()));
	std::uniform_int_distribution<uint64_t> dist;

	std::ostringstream oss;
	oss << "./paad_swap_" << std::hex << dist(rng) << ".bin";

	return oss.str();
}

void writeMatrix(std::ofstream& ofs, const Eigen::MatrixXd& mat)
{
	size_t rows = mat.rows();
	size_t cols = mat.cols();
	ofs.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
	ofs.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
	ofs.write(reinterpret_cast<const char*>(mat.data()), rows * cols * sizeof(double));
}

void readMatrix(std::ifstream& ifs, Eigen::MatrixXd& mat)
{
	size_t rows, cols;
	ifs.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
	ifs.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
	mat.resize(rows, cols);
	ifs.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(double));
}

void writeVector(std::ofstream& ofs, const Eigen::VectorXd& vec)
{
	size_t size = vec.size();
	ofs.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
	ofs.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(double));
}

void readVector(std::ifstream& ifs, Eigen::VectorXd& vec)
{
	size_t size;
	ifs.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	vec.resize(size);
	ifs.read(reinterpret_cast<char*>(vec.data()), size * sizeof(double));
}

void writeRadiativeLayer(std::ofstream& ofs, const RadiativeLayer& layer)
{
	ofs.write(reinterpret_cast<const char*>(&layer.optical_thickness), sizeof(double));
	ofs.write(reinterpret_cast<const char*>(&layer.is_surface), sizeof(bool));
	ofs.write(reinterpret_cast<const char*>(&layer.n_doubling), sizeof(int));

	size_t n_modes = layer.reflectance_m_top.size();
	ofs.write(reinterpret_cast<const char*>(&n_modes), sizeof(size_t));

	for (size_t m = 0; m < n_modes; ++m)
	{
		writeMatrix(ofs, layer.reflectance_m_top[m]);
		writeMatrix(ofs, layer.reflectance_m_bottom[m]);
		writeMatrix(ofs, layer.transmittance_m_top[m]);
		writeMatrix(ofs, layer.transmittance_m_bottom[m]);
	}

	writeVector(ofs, layer.source_up);
	writeVector(ofs, layer.source_down);
}

void readRadiativeLayer(std::ifstream& ifs, RadiativeLayer& layer)
{
	ifs.read(reinterpret_cast<char*>(&layer.optical_thickness), sizeof(double));
	ifs.read(reinterpret_cast<char*>(&layer.is_surface), sizeof(bool));
	ifs.read(reinterpret_cast<char*>(&layer.n_doubling), sizeof(int));

	size_t n_modes;
	ifs.read(reinterpret_cast<char*>(&n_modes), sizeof(size_t));

	layer.reflectance_m_top.resize(n_modes);
	layer.reflectance_m_bottom.resize(n_modes);
	layer.transmittance_m_top.resize(n_modes);
	layer.transmittance_m_bottom.resize(n_modes);

	for (size_t m = 0; m < n_modes; ++m)
	{
		readMatrix(ifs, layer.reflectance_m_top[m]);
		readMatrix(ifs, layer.reflectance_m_bottom[m]);
		readMatrix(ifs, layer.transmittance_m_top[m]);
		readMatrix(ifs, layer.transmittance_m_bottom[m]);
	}

	readVector(ifs, layer.source_up);
	readVector(ifs, layer.source_down);
}

struct TempFileGuard
{
	std::string path;
	
	~TempFileGuard()
	{
		if (!path.empty() && std::filesystem::exists(path))
		{
			std::error_code ec;
			std::filesystem::remove(path, ec);
			
			if (ec)
			{
				std::cerr << "[Warning] Failed to delete swap file: " << path << " (" << ec.message() << ")\n";
			}
		}
	}
};

RadiativeTransferSolver::RadiativeTransferSolver() {}
RadiativeTransferSolver::RadiativeTransferSolver(const geometry::Geometry& geo) : geometry_(geo) {}

void RadiativeTransferSolver::geometry(const geometry::Geometry& geo)
{
	geometry_ = geo;
}

void RadiativeTransferSolver::setPolarizationMode(PolarizationMode mode)
{
	polarization_mode_ = mode;

	if (mode == PolarizationMode::Scalar)
	{
		n_stokes_ = 1;
	}
	else if (mode == PolarizationMode::Linear)
	{
		n_stokes_ = 3;
	}
	else
	{
		n_stokes_ = 4;
	}
}

std::vector<OpticalLayer> RadiativeTransferSolver::setAtmosphericLayerCondition_(const atmosphere::AtmosphereModel& atmos, double spectral, SpectralCoordinateDimension dim)
{
	auto t_start = std::chrono::high_resolution_clock::now();
	int n_layer = atmos.layers.size();
	int n_species = atmos.species.size();
	int n_angle = n_scattering_angle_;

	double wavelength = (dim == SpectralCoordinateDimension::Wavelength) ? spectral : 1.0 / spectral;
	double wavenumber = (dim == SpectralCoordinateDimension::Wavenumber) ? spectral : 1.0 / spectral;

	std::vector<OpticalLayer> layers(n_layer);
	std::vector<double> common_angles(n_angle);

	for(int j = 0; j < n_angle; ++j)
	{
		common_angles[j] = std::numbers::pi * static_cast<double>(j) / static_cast<double>(n_angle - 1);
	}
	
	#pragma omp parallel for num_threads(n_parallel_fourier_)
	for(int i = 0; i < n_layer; ++i)
	{
		auto& layer = layers[i];
		layer.clear();
		layer.scattering_angle = common_angles;
		layer.scattering_matrix.assign(n_angle, Eigen::Matrix4d::Zero());
		layer.is_scattering_species.assign(n_species, false);
		layer.species_absorption_cross_section.assign(n_species, 0.0);
		layer.species_scattering_cross_section.assign(n_species, 0.0);
		layer.species_scattering_matrix.resize(n_species);

		double physical_thickness = atmos.layers[i].altitude_top - atmos.layers[i].altitude_bottom;
		double temp = atmos.layers[i].temperature;
		double pressure_total = atmos.layers[i].pressure;

		for(int j = 0; j < n_species; ++j)
		{
			auto& spec = atmos.species[j];
			double n_density = spec.vertical_number_density_profile[i];

			double sigma_sca = 0.0;
			double sigma_abs = 0.0;
			std::vector<Eigen::Matrix4d> F_s(n_angle, Eigen::Matrix4d::Zero());

			if (std::holds_alternative<atmosphere::MieScattering>(spec.scattering_model))
			{
				auto& mie = std::get<atmosphere::MieScattering>(spec.scattering_model);
				autodiff::complex<double> m(mie.refractive_index[0], mie.refractive_index[1]);
				
				auto res = mie::computeMieScatteringSizeDistribution(n_angle, wavelength, mie.weight_particle_size_distribution, m);

				sigma_sca = res.scattering_cross_section();
				sigma_abs += res.absorption_cross_section();

				for(int k = 0; k < n_angle; ++k)
				{
					F_s[k] = res.scattering_matrix(k);
				}
			}
			else if (std::holds_alternative<atmosphere::RayleighScattering>(spec.scattering_model))
			{
				auto& ray = std::get<atmosphere::RayleighScattering>(spec.scattering_model);
				double m_real = ray.refractive_index[0];
				auto res = rayleigh::computeRayleighScattering(n_angle, wavelength, m_real, ray.number_density_reference, ray.depolarization_factor);

				sigma_sca = res.scattering_cross_section(); 
				sigma_abs += res.absorption_cross_section();

				for(int k = 0; k < n_angle; ++k)
				{
					F_s[k] = res.scattering_matrix(k);
				}
			}
			else if (std::holds_alternative<atmosphere::HenyeyGreensteinScattering>(spec.scattering_model))
			{
				auto& hg = std::get<atmosphere::HenyeyGreensteinScattering>(spec.scattering_model);
				sigma_sca = hg.cross_section;
				double g = hg.asymmetry_factor;

				for(int k = 0; k < n_angle; ++k)
				{
					double theta = common_angles[k];
					double p11 = (1.0 - g * g) / std::pow(1.0 + g * g - 2.0 * g * std::cos(theta), 1.5);
					F_s[k](0, 0) = p11;
				}
			}
			else if (std::holds_alternative<atmosphere::ConstantScattering>(spec.scattering_model))
			{
				auto& iso = std::get<atmosphere::ConstantScattering>(spec.scattering_model);
				sigma_sca = iso.cross_section;

				for(int k = 0; k < n_angle; ++k)
				{
					F_s[k](0, 0) = 1.0;
				}
			}
			else if (std::holds_alternative<atmosphere::ExternalScattering>(spec.scattering_model))
			{
				auto& ext = std::get<atmosphere::ExternalScattering>(spec.scattering_model);
				sigma_sca = ext.cross_section;

				for(int k = 0; k < n_angle; ++k)
				{
					double theta = common_angles[k];

					for (int r = 0; r < n_stokes_; ++r)
					{
						for (int c = 0; c < n_stokes_; ++c)
						{
							F_s[k](r, c) = ext.model.interpolateElement(theta, r, c);
						}
					}
				}
			}

			if (std::holds_alternative<atmosphere::ConstantAbsorption>(spec.absorption_model))
			{
				auto& abs = std::get<atmosphere::ConstantAbsorption>(spec.absorption_model);
				sigma_abs += abs.cross_section;
			}
			else if (std::holds_alternative<atmosphere::ArrheniusAbsorption>(spec.absorption_model))
			{
				auto& abs = std::get<atmosphere::ArrheniusAbsorption>(spec.absorption_model);
				sigma_abs += abs.model.cross_section(spectral, temp);
			}
			else if (std::holds_alternative<atmosphere::HitranAbsorption>(spec.absorption_model))
			{
				auto& hit_abs = std::get<atmosphere::HitranAbsorption>(spec.absorption_model);
				
				double pressure_self = pressure_total * spec.vertical_mixing_ratio_profile[i];

				hitran::Diluent diluent;

				if (!atmos.background_gas.diluent_species.empty())
				{
					const auto& hc = atmos.background_gas;
					double air=0;
					double co2=0;
					double h2=0;
					double he=0;
					double h2o=0;

					for (size_t d = 0; d < hc.diluent_species.size(); ++d)
					{
						std::string sp = hc.diluent_species[d];
						std::transform(sp.begin(), sp.end(), sp.begin(), ::toupper);

						if (sp == "AIR")
						{
							air = hc.diluent_ratio[d];
						}
						else if (sp == "CO2")
						{
							co2 = hc.diluent_ratio[d];
						}
						else if (sp == "H2")
						{
							h2 = hc.diluent_ratio[d];
						}
						else if (sp == "HE")
						{
							he = hc.diluent_ratio[d];
						}
						else if (sp == "H2O")
						{
							h2o = hc.diluent_ratio[d];
						}
					}

					diluent = hitran::Diluent(air, co2, h2, he, h2o);
				}
				else
				{
					diluent = hitran::Diluent(1.0, 0.0, 0.0, 0.0, 0.0);
				}

				double cross_section_sum = 0.0;

				for (const auto& line : hit_abs.cached_lines)
				{
					cross_section_sum += line.computeCrossSection(wavenumber, temp, pressure_self, pressure_total, hit_abs.isotopologue_data, diluent);
				}

				sigma_abs += cross_section_sum;
			}

			layer.is_scattering_species[j] = (sigma_sca > 1e-30);
			layer.species_scattering_cross_section[j] = sigma_sca;
			layer.species_absorption_cross_section[j] = sigma_abs;
			layer.species_scattering_matrix[j] = F_s;

			double k_sca_s = sigma_sca * n_density;
			double k_abs_s = sigma_abs * n_density;

			layer.scattering_coefficient += k_sca_s;
			layer.absorption_coefficient += k_abs_s;

			for(int k = 0; k < n_angle; ++k)
			{
				layer.scattering_matrix[k] += k_sca_s * F_s[k];
			}
		}

		if (layer.scattering_coefficient > 1e-30)
		{
			for(int k = 0; k < n_angle; ++k)
			{
				layer.scattering_matrix[k] /= layer.scattering_coefficient;
			}
		}

		layer.optical_thickness = (layer.scattering_coefficient + layer.absorption_coefficient) * physical_thickness;

		if (layer.optical_thickness > 0.0)
		{
			layer.single_scattering_albedo = layer.scattering_coefficient / (layer.scattering_coefficient + layer.absorption_coefficient);
		}

		layer.planck_function = physics::computePlanckFunction(spectral, temp, dim);
	}
	
	auto t_end = std::chrono::high_resolution_clock::now();
	auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
	PAAD_DEBUG("setAtmosphericLayerCondition_ completed in " << dt << " ms (Spectral: " << spectral << ")");
	
	return layers;
}

OpticalLayer RadiativeTransferSolver::setSurfaceLayerCondition_(const atmosphere::AtmosphereModel& atmos, double spectral, SpectralCoordinateDimension dim)
{
	OpticalLayer layer;
	layer.surface_albedo = atmos.surface.albedo;
	layer.surface_emissivity = atmos.surface.emissivity;

	layer.planck_function = physics::computePlanckFunction(spectral, atmos.surface.temperature, dim);

	return layer;
}

RadiativeLayer RadiativeTransferSolver::initializeAtmosphericLayer_(const OpticalLayer& optical_layer)
{
	RadiativeLayer radiative_layer;
	const int N = geometry_.Ntheta;
	const int N_stokes = n_stokes_; 
	const int N_dim = N_stokes * N; 
	const int Nphi = geometry_.Nphi;

	radiative_layer.n_doubling = 0;
	double tau = optical_layer.optical_thickness;
	
	if (tau <= 0.0)
	{
		radiative_layer.optical_thickness = 0.0;
		radiative_layer.resize(N, geometry_.M, N_stokes); 
		return radiative_layer;
	}

	while(tau > initial_optical_thickness_)
	{
		tau /= 2.0;
		radiative_layer.n_doubling++;
	}

	radiative_layer.optical_thickness = tau;
	radiative_layer.resize(N, geometry_.M, N_stokes); 

	for(int e = 0; e < N; e++)
	{
		double emission = (1.0 - optical_layer.single_scattering_albedo) * physics::computeThermalEmission(optical_layer.planck_function, tau, geometry_.mu(e));
		radiative_layer.source_up(N_stokes * e) = emission;   
		radiative_layer.source_down(N_stokes * e) = emission; 
	}

	std::vector<Eigen::MatrixXd> R_top_phi(Nphi, Eigen::MatrixXd::Zero(N_dim, N_dim));
	std::vector<Eigen::MatrixXd> R_bottom_phi(Nphi, Eigen::MatrixXd::Zero(N_dim, N_dim));
	std::vector<Eigen::MatrixXd> T_top_phi(Nphi, Eigen::MatrixXd::Zero(N_dim, N_dim));
	std::vector<Eigen::MatrixXd> T_bottom_phi(Nphi, Eigen::MatrixXd::Zero(N_dim, N_dim));

	for(int p = 0; p < Nphi; p++)
	{
		for(int e = 0; e < N; e++)
		{
			for(int i = 0; i < N; i++)
			{
				auto compute_z = [&](double u_scat, double u_inc) -> Eigen::MatrixXd
				{
					double scattering_angle, rot1, rot2;
					double dphi = geometry_.phi[p];
					
					const double EPS = 1e-12;

					if (std::abs(std::abs(u_scat) - 1.0) < EPS && std::abs(std::abs(u_inc) - 1.0) < EPS) 
					{
						scattering_angle = (u_scat * u_inc > 0) ? 0.0 : std::numbers::pi;
						rot1 = 0.0;
						rot2 = (u_scat * u_inc > 0) ? dphi : std::numbers::pi - dphi; 
					}
					else if (std::abs(std::abs(u_scat) - 1.0) < EPS)
					{
						scattering_angle = std::acos(std::clamp(u_scat * u_inc, -1.0, 1.0));
						rot1 = 0.0;
						rot2 = (u_scat > 0) ? -dphi : dphi; 
					}
					else if (std::abs(std::abs(u_inc) - 1.0) < EPS)
					{
						scattering_angle = std::acos(std::clamp(u_scat * u_inc, -1.0, 1.0));
						rot1 = (u_inc > 0) ? dphi : -dphi;
						rot2 = 0.0;
					}
					else
					{
						geometry::computeScatteringGeometry(geometry_, u_scat, u_inc, dphi, scattering_angle, rot1, rot2);
					}

					Eigen::Matrix4d F = interpolateScatteringMatrix(optical_layer.scattering_matrix, optical_layer.scattering_angle, scattering_angle);
					return geometry::rotateMuellerMatrix(F, rot1, rot2).block(0, 0, N_stokes, N_stokes);
				};

				double alb_w = optical_layer.single_scattering_albedo * tau;
				double coeff = alb_w / (4.0 * geometry_.mu(i) * geometry_.mu(e));

				R_top_phi[p].block(N_stokes * e, N_stokes * i, N_stokes, N_stokes) = coeff * compute_z(-geometry_.mu(e), geometry_.mu(i));
				R_bottom_phi[p].block(N_stokes * e, N_stokes * i, N_stokes, N_stokes) = coeff * compute_z(geometry_.mu(e), -geometry_.mu(i));
				T_top_phi[p].block(N_stokes * e, N_stokes * i, N_stokes, N_stokes) = coeff * compute_z(geometry_.mu(e), geometry_.mu(i));
				T_bottom_phi[p].block(N_stokes * e, N_stokes * i, N_stokes, N_stokes) = coeff * compute_z(-geometry_.mu(e), -geometry_.mu(i));
			}
		}
	}

	radiative_layer.reflectance_m_top = geometry::computePackedFourierCoefficients(R_top_phi, geometry_);
	radiative_layer.reflectance_m_bottom = geometry::computePackedFourierCoefficients(R_bottom_phi, geometry_);
	radiative_layer.transmittance_m_top = geometry::computePackedFourierCoefficients(T_top_phi, geometry_);
	radiative_layer.transmittance_m_bottom = geometry::computePackedFourierCoefficients(T_bottom_phi, geometry_);

	return radiative_layer;
}

RadiativeLayer RadiativeTransferSolver::initializeSurfaceLayer_(const OpticalLayer& optical_layer)
{
	RadiativeLayer radiative_layer;
	const int N = geometry_.Ntheta;
	const int N_stokes = n_stokes_; 

	radiative_layer.resize(N, geometry_.M, N_stokes); 
	radiative_layer.is_surface = true;
	radiative_layer.n_doubling = 0;
	radiative_layer.optical_thickness = 1.0E100;

	double s = geometry_.WMU.sum();

	for(int e = 0; e < N; e++)
	{
		radiative_layer.source_up(N_stokes * e) = optical_layer.surface_emissivity * optical_layer.planck_function; 

		for(int i = 0; i < N; i++)
		{
			radiative_layer.reflectance_m_top[0](N_stokes * e, N_stokes * i) = optical_layer.surface_albedo / (2.0 * s); 
		}
	}

	return radiative_layer;
}

MonochromeData RadiativeTransferSolver::computeMonochrome(const atmosphere::AtmosphereModel& atmos, double spectral, SpectralCoordinateDimension dim, double initial_tau)
{
	auto t_start = std::chrono::high_resolution_clock::now();

	initial_optical_thickness_ = initial_tau;
	int n_layer = atmos.layers.size();

	auto atmos_layer_opt = setAtmosphericLayerCondition_(atmos, spectral, dim);

	int offset = (atmos.surface.type != SurfaceType::NoSurface) ? 1 : 0;
	std::vector<RadiativeLayer> atmos_layer_rad(n_layer + offset);

	MonochromeData result;
	result.absorption_coefficient.resize(n_layer);
	result.scattering_coefficient.resize(n_layer);
	result.optical_thickness.resize(n_layer);
	result.single_scattering_albedo.resize(n_layer);
	result.asymmetry_parameter.resize(n_layer);
	result.scattering_matrix.resize(n_layer);

	if (n_layer > 0)
	{
		result.is_scattering_species = atmos_layer_opt[0].is_scattering_species;
	}

	if(offset == 1)
	{
		auto surface_layer_opt = setSurfaceLayerCondition_(atmos, spectral, dim);
		atmos_layer_rad[0] = initializeSurfaceLayer_(surface_layer_opt);
	}

	for(int i = 0; i < n_layer; ++i)
	{
		auto t0_init = std::chrono::high_resolution_clock::now();
		int rad_idx = i + offset;

		atmos_layer_rad[rad_idx] = initializeAtmosphericLayer_(atmos_layer_opt[i]);
		atmos_layer_rad[rad_idx].is_surface = false;

		result.absorption_coefficient[i] = atmos_layer_opt[i].absorption_coefficient;
		result.scattering_coefficient[i] = atmos_layer_opt[i].scattering_coefficient;
		result.optical_thickness[i] = atmos_layer_opt[i].optical_thickness;
		result.single_scattering_albedo[i] = atmos_layer_opt[i].single_scattering_albedo;
		result.scattering_matrix[i] = atmos_layer_opt[i].scattering_matrix;
		
		result.species_absorption_cross_section.push_back(atmos_layer_opt[i].species_absorption_cross_section);
		result.species_scattering_cross_section.push_back(atmos_layer_opt[i].species_scattering_cross_section);
		result.species_scattering_matrix.push_back(atmos_layer_opt[i].species_scattering_matrix);
		
		auto t1_init = std::chrono::high_resolution_clock::now();
		auto dt_init = std::chrono::duration_cast<std::chrono::milliseconds>(t1_init - t0_init).count();

		PAAD_DEBUG("[Forward] Layer " << i << " initialization completed in " << dt_init << " ms (tau=" << atmos_layer_rad[rad_idx].optical_thickness << ", doublings=" << atmos_layer_rad[rad_idx].n_doubling << ")");
	}

	RadiativeLayer result_layer_rad = computeAtmosphere(atmos_layer_rad, geometry_, n_parallel_fourier_);

	result.reflectance_m_top = result_layer_rad.reflectance_m_top;
	result.source_up = result_layer_rad.source_up;

	auto t_end = std::chrono::high_resolution_clock::now();
	auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
	PAAD_INFO("computeMonochrome (Forward) completed in " << dt << " ms (Spectral: " << spectral << ")");

	return result;
}

RadiativeLayer computeAtmosphere(const std::vector<RadiativeLayer>& initial_layers, const geometry::Geometry& geo, int n_parallel_fourier)
{
	auto t_start = std::chrono::high_resolution_clock::now();
	int n_layer = initial_layers.size();
	RadiativeLayer result_layer;

	for(int i = 0; i < n_layer; ++i)
	{
		auto t0_comb = std::chrono::high_resolution_clock::now();
		RadiativeLayer current_layer = initial_layers[i];

		if(!current_layer.is_surface)
		{
			for(int j = 0; j < current_layer.n_doubling; ++j)
			{
				current_layer = doubleLayer(current_layer, geo, n_parallel_fourier);
			}
		}

		if(i == 0)
		{
			result_layer = current_layer;
		}
		else
		{
			result_layer = addLayer(result_layer, current_layer, geo, n_parallel_fourier);
		}
		
		auto t1_comb = std::chrono::high_resolution_clock::now();
		auto dt_comb = std::chrono::duration_cast<std::chrono::milliseconds>(t1_comb - t0_comb).count();
		PAAD_DEBUG("[Forward] Layer " << i << " combination (Doubling+Adding) completed in " << dt_comb << " ms");
	}
	
	auto t_end = std::chrono::high_resolution_clock::now();
	auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
	PAAD_INFO("computeAtmosphere (Adding-Doubling core) completed in " << dt << " ms. Layers combined: " << n_layer);

	return result_layer;
}

ForwardState RadiativeTransferSolver::computeForwardState(const atmosphere::AtmosphereModel& atmos, double spectral, SpectralCoordinateDimension dim, double initial_tau)
{
	auto t_start = std::chrono::high_resolution_clock::now();
	initial_optical_thickness_ = initial_tau;
	int n_layer = atmos.layers.size();

	auto atmos_layer_opt = setAtmosphericLayerCondition_(atmos, spectral, dim);
	int offset = (atmos.surface.type != SurfaceType::NoSurface) ? 1 : 0;

	ForwardState state;
	state.spectral_point = spectral;
	state.offset = offset;
	state.initial_optical_layers.resize(n_layer + offset);
	state.records.resize(n_layer + offset);
	state.swap_file_path = generateUniqueFilePath();

	PAAD_DEBUG("Opening swap file: " << state.swap_file_path);

	std::ofstream ofs(state.swap_file_path, std::ios::binary | std::ios::trunc);

	if (!ofs)
	{
		PAAD_ERROR("Failed to open swap file: " << state.swap_file_path);
		throw std::runtime_error("Failed to open swap file for Out-of-Core adjoint: " + state.swap_file_path);
	}

	PAAD_DEBUG("Swap file opened successfully.");

	RadiativeLayer cumulative_layer;

	for(int i = 0; i < n_layer + offset; ++i)
	{
		auto t0_init = std::chrono::high_resolution_clock::now();
		
		RadiativeLayer current_initial;

		if(offset == 1 && i == 0)
		{
			auto surface_layer_opt = setSurfaceLayerCondition_(atmos, spectral, dim);
			state.initial_optical_layers[0] = surface_layer_opt;
			current_initial = initializeSurfaceLayer_(surface_layer_opt);
		}
		else
		{
			int rad_idx = i;
			state.initial_optical_layers[rad_idx] = atmos_layer_opt[i - offset];
			current_initial = initializeAtmosphericLayer_(atmos_layer_opt[i - offset]);
			current_initial.is_surface = false;
		}

		state.records[i].initial_pos = ofs.tellp();
		writeRadiativeLayer(ofs, current_initial);

		PAAD_DEBUG("[Debug] Layer " << i << " | is_surface: " << (current_initial.is_surface ? "true" : "false") << " | n_doubling: " << current_initial.n_doubling);

		RadiativeLayer current_doubled = current_initial;
		
		if (!current_doubled.is_surface)
		{
			for(int j = 0; j < current_doubled.n_doubling; ++j)
			{
				if (j > 0 && j % 10000 == 0)
				{
					PAAD_DEBUG("[Debug] Layer " << i << " doubling progress: " << j << " / " << current_doubled.n_doubling);
				}
				
				current_doubled = doubleLayer(current_doubled, geometry_, n_parallel_fourier_);
			}
		}
		
		state.records[i].doubled_pos = ofs.tellp();
		writeRadiativeLayer(ofs, current_doubled);

		if(i == 0)
		{
			cumulative_layer = current_doubled;
		}
		else
		{
			cumulative_layer = addLayer(cumulative_layer, current_doubled, geometry_, n_parallel_fourier_);
		}

		state.records[i].added_pos = ofs.tellp();
		writeRadiativeLayer(ofs, cumulative_layer);
		
		auto t1_init = std::chrono::high_resolution_clock::now();
		auto dt_init = std::chrono::duration_cast<std::chrono::milliseconds>(t1_init - t0_init).count();

		PAAD_DEBUG("[Out-of-Core Forward] Layer " << i << " processed and swapped to disk in " << dt_init << " ms");
	}

	ofs.close();

	auto t_end = std::chrono::high_resolution_clock::now();
	auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
	PAAD_INFO("computeForwardState (Out-of-Core) completed in " << dt << " ms (Spectral: " << spectral << ")");

	return state;
}

InternalField RadiativeTransferSolver::computeInternalField(const ForwardState& state)
{
	InternalField field;
	int num_records = state.records.size();

	if (num_records == 0)
	{
		return field;
	}
	
	int K = num_records - 1;
	int dim = n_stokes_ * geometry_.Ntheta;

	field.I_plus_thm.assign(num_records, Eigen::VectorXd::Zero(dim));
	field.I_minus_thm.assign(num_records, Eigen::VectorXd::Zero(dim));
	field.I_plus_sca.assign(geometry_.M + 1, std::vector<Eigen::MatrixXd>(num_records, Eigen::MatrixXd::Zero(dim, dim)));
	field.I_minus_sca.assign(geometry_.M + 1, std::vector<Eigen::MatrixXd>(num_records, Eigen::MatrixXd::Zero(dim, dim)));

	std::ifstream ifs(state.swap_file_path, std::ios::binary);

	if (!ifs)
	{
		throw std::runtime_error("Failed to open swap file for InternalField computation: " + state.swap_file_path);
	}

	core::RadiativeLayer layer_toa;
	ifs.seekg(state.records[K].added_pos);
	core::readRadiativeLayer(ifs, layer_toa);
	
	field.I_minus_thm[K] = Eigen::VectorXd::Zero(dim);
	field.I_plus_thm[K] = layer_toa.source_up;

	for (int m = 0; m <= geometry_.M; ++m)
	{
		field.I_minus_sca[m][K] = Eigen::MatrixXd::Identity(dim, dim); 
		field.I_plus_sca[m][K]  = layer_toa.reflectance_m_top[m];
	}

	for (int k = K; k >= 1; --k)
	{
		core::RadiativeLayer layer_top;
		core::RadiativeLayer layer_bottom;

		ifs.seekg(state.records[k].doubled_pos);
		core::readRadiativeLayer(ifs, layer_top);
		
		ifs.seekg(state.records[k - 1].added_pos);
		core::readRadiativeLayer(ifs, layer_bottom);

		if (geometry_.M >= 0)
		{
			computeInternalRadianceVector(layer_top, layer_bottom, field.I_minus_thm[k], geometry_, n_stokes_, field.I_minus_thm[k - 1], field.I_plus_thm[k - 1]);
		}

		#pragma omp parallel for num_threads(n_parallel_fourier_)
		for (int m = 0; m <= geometry_.M; ++m)
		{
			computeInternalRadianceMatrix(layer_top, layer_bottom, field.I_minus_sca[m][k], geometry_, m, n_stokes_, field.I_minus_sca[m][k - 1], field.I_plus_sca[m][k - 1]);
		}
	}
	
	ifs.close();
	return field;
}

std::vector<OpticalSensitivity> RadiativeTransferSolver::computeAdjointMonochrome(const ForwardState& state, const RadiativeLayer& adjoint_source, const InternalField* fwd_field, const InternalField* adj_field)
{
	auto t_start = std::chrono::high_resolution_clock::now();
	int total_layers = state.records.size();
	int K = total_layers - 1;
	int dim = n_stokes_ * geometry_.Ntheta;

	std::vector<OpticalSensitivity> opt_sens(total_layers);
	TempFileGuard file_guard{state.swap_file_path}; 

	std::ifstream ifs(state.swap_file_path, std::ios::binary);

	if (!ifs)
	{
		throw std::runtime_error("Failed to read swap file for Out-of-Core adjoint: " + state.swap_file_path);
	}

	RadiativeLayer adj_current = adjoint_source; 

	if (fwd_field && adj_field && geometry_.M >= 0)
	{
		adj_current.source_up += adj_field->I_plus_thm[K];

		for (int m = 0; m <= geometry_.M; ++m)
		{
			adj_current.reflectance_m_top[m] += adj_field->I_plus_sca[m][K];
		}
	}

	Eigen::VectorXd cur_adj_I_minus_thm_k = Eigen::VectorXd::Zero(dim);
	std::vector<Eigen::MatrixXd> cur_adj_I_minus_sca_k(geometry_.M + 1, Eigen::MatrixXd::Zero(dim, dim));

	RadiativeLayer layer_bottom;
	RadiativeLayer layer_top;
	RadiativeLayer initial_doubling;

	for (int i = K; i >= 1; --i)
	{
		auto t0_adj = std::chrono::high_resolution_clock::now();
		
		ifs.seekg(state.records[i - 1].added_pos);
		readRadiativeLayer(ifs, layer_bottom);

		ifs.seekg(state.records[i].doubled_pos);
		readRadiativeLayer(ifs, layer_top);

		auto adj_pair = addLayer_adjoint(layer_bottom, layer_top, geometry_, adj_current, n_parallel_fourier_);
		RadiativeLayer adj_bot = adj_pair[0];
		RadiativeLayer adj_top = adj_pair[1];

		if (fwd_field && adj_field)
		{
			if (geometry_.M >= 0)
			{
				Eigen::VectorXd adj_I_plus_k_1 = adj_field->I_plus_thm[i - 1];
				Eigen::VectorXd adj_I_minus_k_1 = adj_field->I_minus_thm[i - 1] + cur_adj_I_minus_thm_k;

				auto res_thm = computeInternalRadianceVector_adjoint(layer_top, layer_bottom, fwd_field->I_minus_thm[i], fwd_field->I_minus_thm[i - 1], adj_I_plus_k_1, adj_I_minus_k_1, geometry_, n_stokes_);

				adj_top.optical_thickness += res_thm.adj_layer_top.optical_thickness;
				adj_bot.optical_thickness += res_thm.adj_layer_bottom.optical_thickness;
				adj_top.source_down += res_thm.adj_layer_top.source_down;
				adj_bot.source_up += res_thm.adj_layer_bottom.source_up;
				adj_top.reflectance_m_bottom[0] += res_thm.adj_layer_top.reflectance_m_bottom[0];
				adj_bot.reflectance_m_top[0] += res_thm.adj_layer_bottom.reflectance_m_top[0];
				adj_top.transmittance_m_top[0] += res_thm.adj_layer_top.transmittance_m_top[0];

				cur_adj_I_minus_thm_k = res_thm.adj_I_minus_k_vec;
			}

			#pragma omp parallel for num_threads(n_parallel_fourier_)
			for (int m = 0; m <= geometry_.M; ++m)
			{
				Eigen::MatrixXd adj_I_plus_k_1 = adj_field->I_plus_sca[m][i - 1];
				Eigen::MatrixXd adj_I_minus_k_1 = adj_field->I_minus_sca[m][i - 1] + cur_adj_I_minus_sca_k[m];

				auto res_sca = computeInternalRadianceMatrix_adjoint(layer_top, layer_bottom, fwd_field->I_minus_sca[m][i], fwd_field->I_minus_sca[m][i - 1], adj_I_plus_k_1, adj_I_minus_k_1, geometry_, m, n_stokes_);

				#pragma omp critical
				{
					adj_top.optical_thickness += res_sca.adj_layer_top.optical_thickness;
					adj_bot.optical_thickness += res_sca.adj_layer_bottom.optical_thickness;
					adj_top.reflectance_m_bottom[m] += res_sca.adj_layer_top.reflectance_m_bottom[m];
					adj_bot.reflectance_m_top[m] += res_sca.adj_layer_bottom.reflectance_m_top[m];
					adj_top.transmittance_m_top[m] += res_sca.adj_layer_top.transmittance_m_top[m];
					cur_adj_I_minus_sca_k[m] = res_sca.adj_I_minus_k_mat;
				}
			}
		}

		adj_current = adj_bot;
		RadiativeLayer adj_single = adj_top;
		
		ifs.seekg(state.records[i].initial_pos);
		readRadiativeLayer(ifs, initial_doubling);
		
		std::vector<RadiativeLayer> doubling_states(initial_doubling.n_doubling + 1);
		doubling_states[0] = initial_doubling;

		for (int d = 0; d < initial_doubling.n_doubling; ++d)
		{
			doubling_states[d + 1] = doubleLayer(doubling_states[d], geometry_, n_parallel_fourier_);
		}
		
		for (int d = initial_doubling.n_doubling - 1; d >= 0; --d)
		{
			adj_single = doubleLayer_adjoint(doubling_states[d], geometry_, adj_single, n_parallel_fourier_);
		}

		opt_sens[i] = computeInitializationSensitivities(adj_single, initial_doubling, state.initial_optical_layers[i].single_scattering_albedo, state.initial_optical_layers[i].planck_function, geometry_, n_scattering_angle_);

		double scale_tau = 1.0 / std::pow(2.0, initial_doubling.n_doubling);
		opt_sens[i].optical_thickness *= scale_tau;
		
		auto t1_adj = std::chrono::high_resolution_clock::now();
		auto dt_adj = std::chrono::duration_cast<std::chrono::milliseconds>(t1_adj - t0_adj).count();

		PAAD_DEBUG("[Out-of-Core Adjoint] Layer " << i << " backpropagation completed in " << dt_adj << " ms");
	}

	auto t0_adj0 = std::chrono::high_resolution_clock::now();
	RadiativeLayer adj_single0 = adj_current;

	ifs.seekg(state.records[0].initial_pos);
	readRadiativeLayer(ifs, initial_doubling);

	if (!initial_doubling.is_surface) 
	{
		std::vector<RadiativeLayer> doubling_states(initial_doubling.n_doubling + 1);
		doubling_states[0] = initial_doubling;

		for (int d = 0; d < initial_doubling.n_doubling; ++d)
		{
			doubling_states[d + 1] = doubleLayer(doubling_states[d], geometry_, n_parallel_fourier_);
		}

		for (int d = initial_doubling.n_doubling - 1; d >= 0; --d)
		{
			adj_single0 = doubleLayer_adjoint(doubling_states[d], geometry_, adj_single0, n_parallel_fourier_);
		}
	}

	opt_sens[0] = computeInitializationSensitivities(adj_single0, initial_doubling, state.initial_optical_layers[0].single_scattering_albedo, state.initial_optical_layers[0].planck_function, geometry_, n_scattering_angle_);

	double scale_tau_0 = 1.0 / std::pow(2.0, initial_doubling.n_doubling);
	opt_sens[0].optical_thickness *= scale_tau_0;

	if (state.offset == 1)
	{
		double s = geometry_.WMU.sum();
		double grad_albedo = 0.0;
		double grad_emissivity_planck = 0.0;
		
		for(int e = 0; e < geometry_.Ntheta; e++)
		{
			for(int j = 0; j < geometry_.Ntheta; j++)
			{
				grad_albedo += adj_single0.reflectance_m_top[0](n_stokes_ * e, n_stokes_ * j) / (2.0 * s);
			}

			grad_emissivity_planck += adj_single0.source_up(n_stokes_ * e);
		}
		
		opt_sens[0].surface_albedo = grad_albedo;
		opt_sens[0].surface_emissivity = grad_emissivity_planck * state.initial_optical_layers[0].planck_function;
		opt_sens[0].planck_function = grad_emissivity_planck * state.initial_optical_layers[0].surface_emissivity;
	}
	
	ifs.close();

	auto t1_adj0 = std::chrono::high_resolution_clock::now();
	auto dt_adj0 = std::chrono::duration_cast<std::chrono::milliseconds>(t1_adj0 - t0_adj0).count();
	PAAD_DEBUG("[Out-of-Core Adjoint] Layer 0 (Surface/Base) backpropagation completed in " << dt_adj0 << " ms");

	auto t_end = std::chrono::high_resolution_clock::now();
	auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
	PAAD_INFO("computeAdjointMonochrome (Out-of-Core Backpropagation) completed in " << dt << " ms. Layers: " << total_layers);

	return opt_sens;
}

}
