#pragma once

#include <vector>
#include "types.hpp"
#include "geometry.hpp"
#include "enums.hpp"
#include "adjoint.hpp"
#include <Eigen/Dense>

namespace paad::core
{

struct LayerDiskRecord
{
    std::streampos initial_pos;
    std::streampos doubled_pos;
    std::streampos added_pos;
};

struct ForwardState
{
    double spectral_point;
    int offset;
    std::vector<OpticalLayer> initial_optical_layers;
    
    std::string swap_file_path;
    std::vector<LayerDiskRecord> records;
};

struct InternalField
{
    std::vector<Eigen::VectorXd> I_plus_thm;  
    std::vector<Eigen::VectorXd> I_minus_thm; 
    std::vector<std::vector<Eigen::MatrixXd>> I_plus_sca;  
    std::vector<std::vector<Eigen::MatrixXd>> I_minus_sca; 
};

class RadiativeTransferSolver
{
    private:
        geometry::Geometry geometry_;
        double initial_optical_thickness_ = 1.0e-6;
        int n_scattering_angle_ = 901;

        PolarizationMode polarization_mode_ = PolarizationMode::FullStokes;
        int n_stokes_ = 4;
        
        int n_parallel_fourier_ = 1;

        RadiativeLayer initializeAtmosphericLayer_(const OpticalLayer& optical_layer);
        RadiativeLayer initializeSurfaceLayer_(const OpticalLayer& optical_layer);

        std::vector<OpticalLayer> setAtmosphericLayerCondition_(const atmosphere::AtmosphereModel& atmos, double spectral, SpectralCoordinateDimension dim);
        OpticalLayer setSurfaceLayerCondition_(const atmosphere::AtmosphereModel& atmos, double spectral, SpectralCoordinateDimension dim);

    public:
        RadiativeTransferSolver();
        RadiativeTransferSolver(const geometry::Geometry& geo);

        void geometry(const geometry::Geometry& geo);
        void setScatteringAngleResolution(int n) { n_scattering_angle_ = n; }
        
        void setPolarizationMode(PolarizationMode mode);

        void setFourierParallelThreads(int n_threads) { n_parallel_fourier_ = n_threads; }

        MonochromeData computeMonochrome(const atmosphere::AtmosphereModel& atmos, double spectral, SpectralCoordinateDimension dim, double initial_tau);
        ForwardState computeForwardState(const atmosphere::AtmosphereModel& atmos, double spectral, SpectralCoordinateDimension dim, double initial_tau);
        InternalField computeInternalField(const ForwardState& state);
        std::vector<OpticalSensitivity> computeAdjointMonochrome(const ForwardState& state, const RadiativeLayer& adjoint_source, const InternalField* fwd_field = nullptr, const InternalField* adj_field = nullptr);
};

RadiativeLayer computeAtmosphere(const std::vector<RadiativeLayer>& initial_layers, const geometry::Geometry& geo, int n_parallel_fourier);

Eigen::Matrix4d interpolateScatteringMatrix(const std::vector<Eigen::Matrix4d>& matrices, const std::vector<double>& angles, double target_angle);

void writeMatrix(std::ofstream& ofs, const Eigen::MatrixXd& mat);
void readMatrix(std::ifstream& ifs, Eigen::MatrixXd& mat);
void writeVector(std::ofstream& ofs, const Eigen::VectorXd& vec);
void readVector(std::ifstream& ifs, Eigen::VectorXd& vec);
void writeRadiativeLayer(std::ofstream& ofs, const RadiativeLayer& layer);
void readRadiativeLayer(std::ifstream& ifs, RadiativeLayer& layer);

}
