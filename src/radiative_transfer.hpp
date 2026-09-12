#pragma once

#include <string>
#include <vector>

#include "types.hpp"
#include "geometry.hpp"
#include "solver.hpp"
#include "atmosphere.hpp"
#include "inversion.hpp"
#include "sfi.hpp"

namespace paad
{

class RadiativeTransfer
{
	private:
		atmosphere::AtmosphereModel atmosphere_model_;
		core::RadiativeTransferSolver radiative_transfer_solver_;
		geometry::Geometry geometry_;
		core::RadiativeTransferResult result_;
		core::Spectral spectral_;
		core::Simulation simulation_;

		std::vector<core::AtmosphereSensitivity> jacobians_;

		std::vector<core::RadiativeLayer> adjoint_sources_;
		bool has_custom_adjoint_source_ = false;

		std::vector<Eigen::VectorXd> sfi_grad_I_thm_;
		std::vector<Eigen::MatrixXd> sfi_grad_R_mat_;
		bool has_custom_sfi_gradient_ = false;

		std::string config_filename_;

	public:
		RadiativeTransfer();
		RadiativeTransfer(const std::string& config_filename);
		
		void loadConfiguration(void);
		void setupSpectralGrid(void);
		void setup(void);
		void run(void);
		void exportResult(void) const;
		const core::RadiativeTransferResult& getResult() const { return result_; }
		const geometry::Geometry& getGeometry() const { return geometry_; }
		
		const std::vector<core::AtmosphereSensitivity>& getJacobians() const { return jacobians_; }

		void setAdjointSourceFourier(const std::vector<std::vector<Eigen::MatrixXd>>& F_mode_list, const std::vector<Eigen::VectorXd>& emission_up_list = {});
		void setAdjointSourceSFI(const std::vector<Eigen::VectorXd>& grad_thm, const std::vector<Eigen::MatrixXd>& grad_ref);
};

}
