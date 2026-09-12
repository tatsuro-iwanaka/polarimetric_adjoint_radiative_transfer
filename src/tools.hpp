#pragma once

#include <string>
#include <vector>

namespace paad::tools
{

enum class ObservableType{Radiance, DegreeOfPolarization};

struct ObservationPoint
{
	double theta_e;
	double theta_i;
	double phi;
	
	double I;
	double Q;
	double U;
	double V;
	
	double weight = 1.0;
	double solar_flux = 0.0;
};

void generateAdjointSourceFromObservations(const std::string& forward_filepath, const std::vector<ObservationPoint>& observations, const std::string& output_filepath, ObservableType obs_type);
void generateSensitivityAdjointSource(const std::string& forward_filepath, const std::string& output_filepath, double target_theta_e, double target_theta_i, double target_phi, int target_stokes, bool is_thermal = false, double solar_flux = 1.0);

}
