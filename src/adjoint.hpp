#pragma once

#include <vector>
#include <utility>
#include <Eigen/Dense>

#include "types.hpp"
#include "geometry.hpp"

namespace paad::core
{

struct OpticalSensitivity
{
	double optical_thickness;
	double single_scattering_albedo;
	double planck_function;
	std::vector<std::pair<double, Eigen::Matrix4d>> scattering_phase_matrix;

	double surface_albedo = 0.0;
	double surface_emissivity = 0.0;
	double surface_temperature = 0.0;
};

template <typename T>
void accumulate_gradient_to_grid(double theta_val, const T& grad_val, const std::vector<double>& theta_grid, std::vector<T>& grad_P_array);
RadiativeLayer doubleLayer_adjoint(const RadiativeLayer& layer, const geometry::Geometry& geometry, const RadiativeLayer& adj_result, int n_parallel_fourier);
std::vector<RadiativeLayer> addLayer_adjoint(const RadiativeLayer& layer_bottom, const RadiativeLayer& layer_top, const geometry::Geometry& geometry, const RadiativeLayer& adj_result, int n_parallel_fourier);
OpticalSensitivity computeInitializationSensitivities(const RadiativeLayer& adj_layer, const RadiativeLayer& fwd_matrix_layer, double single_scattering_albedo, double planck_function, const geometry::Geometry& geo, int n_theta);

struct InternalRadianceAdjointResult {
    RadiativeLayer adj_layer_top;
    RadiativeLayer adj_layer_bottom;
    Eigen::VectorXd adj_I_minus_k_vec;
    Eigen::MatrixXd adj_I_minus_k_mat;
};

InternalRadianceAdjointResult computeInternalRadianceVector_adjoint(
    const RadiativeLayer& layer_top, const RadiativeLayer& layer_bottom,
    const Eigen::VectorXd& I_minus_k, const Eigen::VectorXd& I_minus_k_1,
    const Eigen::VectorXd& adj_I_plus_k_1, const Eigen::VectorXd& adj_I_minus_k_1,
    const geometry::Geometry& geo, int n_stokes);

InternalRadianceAdjointResult computeInternalRadianceMatrix_adjoint(
    const RadiativeLayer& layer_top, const RadiativeLayer& layer_bottom,
    const Eigen::MatrixXd& I_minus_k, const Eigen::MatrixXd& I_minus_k_1,
    const Eigen::MatrixXd& adj_I_plus_k_1, const Eigen::MatrixXd& adj_I_minus_k_1,
    const geometry::Geometry& geo, int m, int n_stokes);
	
}
