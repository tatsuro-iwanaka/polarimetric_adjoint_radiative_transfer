#pragma once

#include "solver.hpp"
#include "geometry.hpp"
#include <vector>
#include <Eigen/Dense>

namespace paad::sfi
{

struct SFISensitivity
{
	core::InternalField adj_field;
	std::vector<core::OpticalSensitivity> direct_gradients;
};

class SFIIntegrator
{
public:
	SFIIntegrator(const geometry::Geometry& geo, int n_stokes, int threads);

	Eigen::VectorXd computeThermalEmission(const core::InternalField& field, const std::vector<core::OpticalLayer>& opt_layers, double mu_obs);
	Eigen::MatrixXd computeReflectanceMatrix(const core::InternalField& field, const std::vector<core::OpticalLayer>& opt_layers, double mu_obs, double mu_0, double dphi);
	SFISensitivity computeAdjoint(const core::InternalField& field, const std::vector<core::OpticalLayer>& opt_layers, const Eigen::VectorXd& grad_I_thm, const Eigen::MatrixXd& grad_R_mat, double mu_obs, double mu_0, double dphi);

private:
	geometry::Geometry geo_;
	int n_stokes_;
	int threads_;
};

}
