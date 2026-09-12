#include <iostream>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tools.hpp"

std::vector<paad::tools::ObservationPoint> loadObservationsCSV(const std::string& filepath)
{
	std::vector<paad::tools::ObservationPoint> obs_list;
	std::ifstream ifs(filepath);

	if (!ifs.is_open())
	{
		throw std::runtime_error("Cannot open observation CSV file: " + filepath);
	}

	std::string line;
	std::getline(ifs, line);

	while (std::getline(ifs, line))
	{
		if (line.empty() || line[0] == '#')
		{
			continue;
		}
		
		std::stringstream ss(line);
		std::string token;
		std::vector<double> vals;

		while (std::getline(ss, token, ','))
		{
			vals.push_back(std::stod(token));
		}

		if (vals.size() >= 9)
		{
			paad::tools::ObservationPoint obs;
			obs.theta_e = vals[0];
			obs.theta_i = vals[1];
			obs.phi = vals[2];
			obs.I = vals[3];
			obs.Q = vals[4];
			obs.U = vals[5];
			obs.V = vals[6];
			obs.weight = vals[7];
			obs.solar_flux = vals[8];
			obs_list.push_back(obs);
		}
	}
	return obs_list;
}

int main(int argc, char** argv)
{
	try
	{
		if (argc < 2)
		{
			std::cerr << "Usage:\n";
			std::cerr << "  " << argv[0] << " obs <forward_result.nc> <observations.csv> <output_adjoint.nc> <type: radiance|pol>\n";
			std::cerr << "  " << argv[0] << " sens <forward_result.nc> <output_adjoint.nc> <theta_e> <theta_i> <phi> <stokes_idx> <is_thermal:0|1> <solar_flux>\n";
			return 1;
		}

		std::string mode = argv[1];

		if (mode == "obs")
		{
			if (argc < 6)
			{
				throw std::runtime_error("Insufficient arguments for 'obs' mode.");
			}

			std::string forward_nc = argv[2];
			std::string obs_csv = argv[3];
			std::string output_nc = argv[4];
			std::string type_str = argv[5];

			paad::tools::ObservableType obs_type = (type_str == "pol") ? paad::tools::ObservableType::DegreeOfPolarization : paad::tools::ObservableType::Radiance;

			auto obs_list = loadObservationsCSV(obs_csv);
			paad::tools::generateAdjointSourceFromObservations(forward_nc, obs_list, output_nc, obs_type);
		}
		else if (mode == "sens")
		{
			if (argc < 10)
			{
				throw std::runtime_error("Insufficient arguments for 'sens' mode.");
			}

			std::string forward_nc = argv[2];
			std::string output_nc  = argv[3];
			double theta_e = std::stod(argv[4]);
			double theta_i = std::stod(argv[5]);
			double phi = std::stod(argv[6]);
			int stokes_idx = std::stoi(argv[7]);
			bool is_therm = (std::stoi(argv[8]) != 0);
			double solar_f = std::stod(argv[9]);

			paad::tools::generateSensitivityAdjointSource(forward_nc, output_nc, theta_e, theta_i, phi, stokes_idx, is_therm, solar_f);
		}
		else
		{
			throw std::runtime_error("Unknown mode: " + mode);
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "Fatal Error: " << e.what() << '\n';
		return 1;
	}

	return 0;
}
