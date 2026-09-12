#pragma once

#include <utility>

#include "types.hpp"
#include "atmosphere.hpp"
#include "geometry.hpp"
#include "inversion.hpp"

namespace paad::io
{

void exportResultNetCDF(const core::RadiativeTransferResult& result, const core::Simulation& sim_info, const atmosphere::AtmosphereModel& atmosphere, const geometry::Geometry& geometry, const core::Spectral& spectral_info);

void exportJacobianNetCDF(const std::string& filename, const std::vector<core::AtmosphereSensitivity>& jacobians, const atmosphere::AtmosphereModel& atmosphere, const core::Spectral& spectral_info);

std::pair<std::vector<std::vector<Eigen::MatrixXd>>, std::vector<Eigen::VectorXd>> importAdjointSourceNetCDF(const std::string& filepath, const geometry::Geometry& geom, const core::Spectral& spectral_info, PolarizationMode pol_mode);

}
