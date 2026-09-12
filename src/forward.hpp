#pragma once

#include <Eigen/Dense>
#include "types.hpp"
#include "geometry.hpp"

namespace paad::core
{

Eigen::MatrixXd expandDiagonal(const Eigen::VectorXd& vec, int n_stokes);
Eigen::MatrixXd expandWMU(const Eigen::MatrixXd& WMU_small, int n_stokes);

RadiativeLayer doubleLayer(const RadiativeLayer& layer, const geometry::Geometry& geometry, int n_parallel_fourier);
RadiativeLayer addLayer(const RadiativeLayer& layer_bottom, const RadiativeLayer& layer_top, const geometry::Geometry& geometry, int n_parallel_fourier);

void computeInternalRadianceVector(const RadiativeLayer& layer_top, const RadiativeLayer& layer_bottom, const Eigen::VectorXd& I_minus_k, const geometry::Geometry& geo, int n_stokes, Eigen::VectorXd& I_minus_k_1, Eigen::VectorXd& I_plus_k_1);

void computeInternalRadianceMatrix(const RadiativeLayer& layer_top, const RadiativeLayer& layer_bottom, const Eigen::MatrixXd& I_minus_k, const geometry::Geometry& geo, int m, int n_stokes, Eigen::MatrixXd& I_minus_k_1, Eigen::MatrixXd& I_plus_k_1);
}
