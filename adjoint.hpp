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

RadiativeLayer doubleLayer_adjoint(const RadiativeLayer& layer, const geometry::Geometry& geometry, const RadiativeLayer& adj_result);

std::vector<RadiativeLayer> addLayer_adjoint(const RadiativeLayer& layer_bottom, const RadiativeLayer& layer_top, const geometry::Geometry& geometry, const RadiativeLayer& adj_result);

OpticalSensitivity computeInitializationSensitivities(const RadiativeLayer& adj_layer, const RadiativeLayer& fwd_matrix_layer, double single_scattering_albedo, double planck_function, const geometry::Geometry& geo, int n_theta);

}
