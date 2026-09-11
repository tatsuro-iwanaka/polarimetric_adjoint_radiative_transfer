#include "atmosphere.hpp"
#include "units.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <netcdf>

namespace paad::atmosphere
{

void ArrheniusCrossSectionModel::loadVectors(const std::vector<double>& spectral, const std::vector<double>& temperature, const std::vector<std::vector<double>>& cross_section)
{
    spectral_ = spectral;
    size_t num_specs = spectral_.size();
    size_t num_temps = temperature.size();

    slope_.resize(num_specs);
    intercept_.resize(num_specs);
    valid_flag_.resize(num_specs, false);

    int valid_count = 0;

    for (size_t i = 0; i < num_specs; ++i)
    {
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
        int n = 0;

        for (size_t j = 0; j < num_temps; ++j)
        {
            double val = cross_section[i][j];

            if (val > 0.0 && temperature[j] > 0.0) // ゼロ除算回避を追加
            { 
                double x = 1.0 / temperature[j];
                double y = std::log(val);
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_xx += x * x;
                n++;
            }
        }

        if (n >= 2)
        {
            double denominator = static_cast<double>(n) * sum_xx - sum_x * sum_x;

            if (std::abs(denominator) > 0.0)
            {
                slope_[i] = (static_cast<double>(n) * sum_xy - sum_x * sum_y) / denominator;
                intercept_[i] = (sum_xx * sum_y - sum_xy * sum_x) / denominator;
                valid_flag_[i] = true;
                valid_count++;
            }
        } 
        else if (n == 1)
        {
            slope_[i] = 0.0;
            intercept_[i] = sum_y;
            valid_flag_[i] = true;
            valid_count++;
        }
        else
        {
            slope_[i] = 0.0;
            intercept_[i] = -1.0E300;
            valid_flag_[i] = false;
        }
    }
}

void ArrheniusCrossSectionModel::loadNetCDF(const NetCDFCrossSectionConfig& config)
{
    try
    {
        netCDF::NcFile dataFile(config.filename, netCDF::NcFile::read);

        auto getVarSafe = [&](const std::string& name) -> netCDF::NcVar
        {
            netCDF::NcVar var = dataFile.getVar(name);
            if(var.isNull()) throw std::runtime_error("NetCDF Variable '" + name + "' not found in " + config.filename);
            return var;
        };

        auto getScaleFromAttribute = [](const netCDF::NcVar& var) -> double
        {
            try
            {
                netCDF::NcVarAtt att = var.getAtt("units");
                if (!att.isNull())
                {
                    std::string unit_str;
                    att.getValues(unit_str);
                    return units::getUnitInfo(unit_str).to_si;
                }
            }
            catch (...)
            {
                ;
            }
            return 1.0;
        };

        netCDF::NcVar v_spec = getVarSafe(config.var_name_spectral);
        netCDF::NcVar v_temp = getVarSafe(config.var_name_temperature);
        netCDF::NcVar v_cross_section = getVarSafe(config.var_name_cross_section);

        size_t num_spec = v_spec.getDim(0).getSize();
        size_t num_temp = v_temp.getDim(0).getSize();

        std::vector<double> buf_spec(num_spec);
        std::vector<double> buf_temperature(num_temp);
        std::vector<double> buf_cross_section(num_spec * num_temp);

        v_spec.getVar(buf_spec.data());
        v_temp.getVar(buf_temperature.data());
        v_cross_section.getVar(buf_cross_section.data());

        double scale_spec = getScaleFromAttribute(v_spec); 
        double scale_temp = getScaleFromAttribute(v_temp);
        double scale_cross_section = getScaleFromAttribute(v_cross_section);

        std::vector<double> spectral(num_spec);
        std::vector<double> temperature(num_temp);
        std::vector<std::vector<double>> cross_section(num_spec, std::vector<double>(num_temp));

        for(size_t i = 0; i < num_spec; ++i)
        {
            spectral[i] = buf_spec[i] * scale_spec;

            for(size_t j = 0; j < num_temp; ++j)
            {
                cross_section[i][j] = buf_cross_section[i * num_temp + j] * scale_cross_section;
            }
        }
        for(size_t j = 0; j < num_temp; ++j)
        {
            temperature[j] = buf_temperature[j] * scale_temp;
        }

        loadVectors(spectral, temperature, cross_section);
    }
    catch(const netCDF::exceptions::NcException& e)
    {
        throw std::runtime_error("NetCDF Error in " + config.filename + ": " + std::string(e.what()));
    }
}

void ExternalScatteringModel::loadNetCDF(const NetCDFExternalScatteringConfig& config, PolarizationMode mode)
{
	try
	{
		netCDF::NcFile dataFile(config.filename, netCDF::NcFile::read);

		auto getVarSafe = [&](const std::string& name) -> netCDF::NcVar
		{
			netCDF::NcVar var = dataFile.getVar(name);
			if(var.isNull()) throw std::runtime_error("NetCDF Variable '" + name + "' not found in " + config.filename);
			return var;
		};

		netCDF::NcVar v_angle = getVarSafe(config.var_name_scattering_angle);
		netCDF::NcVar v_matrix = getVarSafe(config.var_name_scattering_matrix);

		size_t num_angle = v_angle.getDim(0).getSize();
		
		int expected_stokes_dim = (mode == PolarizationMode::Scalar) ? 1 : 4;

		if (v_matrix.getDimCount() != 3)
		{
			throw std::runtime_error("Scattering matrix must have 3 dimensions (angle, stokes_out, stokes_in).");
		}

		if (v_matrix.getDim(1).getSize() != expected_stokes_dim || v_matrix.getDim(2).getSize() != expected_stokes_dim)
		{
			throw std::runtime_error("Scattering matrix stokes dimensions mismatch with polarization mode.");
		}

		std::vector<double> buf_angle(num_angle);
		std::vector<double> buf_matrix(num_angle * expected_stokes_dim * expected_stokes_dim);

		v_angle.getVar(buf_angle.data());
		v_matrix.getVar(buf_matrix.data());

		scattering_angle_ = buf_angle;
		scattering_matrix_.resize(num_angle, std::vector<std::vector<double>>(expected_stokes_dim, std::vector<double>(expected_stokes_dim)));

		size_t idx = 0;

		for (size_t i = 0; i < num_angle; ++i)
		{
			for (int r = 0; r < expected_stokes_dim; ++r)
			{
				for (int c = 0; c < expected_stokes_dim; ++c)
				{
					scattering_matrix_[i][r][c] = buf_matrix[idx++];
				}
			}
		}
	}
	catch(const netCDF::exceptions::NcException& e)
	{
		throw std::runtime_error("NetCDF Error in " + config.filename + ": " + std::string(e.what()));
	}
}

double ExternalScatteringModel::interpolateElement(double angle, int row, int col) const
{
	if (scattering_angle_.empty())
	{
		return 0.0;
	}

	if (angle <= scattering_angle_.front())
	{
		return scattering_matrix_.front()[row][col];
	}

	if (angle >= scattering_angle_.back())
	{
		return scattering_matrix_.back()[row][col];
	}

	auto it = std::lower_bound(scattering_angle_.begin(), scattering_angle_.end(), angle);

	if (it != scattering_angle_.end() && *it == angle)
	{
		size_t idx = std::distance(scattering_angle_.begin(), it);
		return scattering_matrix_[idx][row][col];
	}

	auto it_next = it;
	auto it_prev = it - 1;

	size_t idx_next = std::distance(scattering_angle_.begin(), it_next);
	size_t idx_prev = std::distance(scattering_angle_.begin(), it_prev);

	double ang_prev = *it_prev;
	double ang_next = *it_next;
	double val_prev = scattering_matrix_[idx_prev][row][col];
	double val_next = scattering_matrix_[idx_next][row][col];

	double ratio = (angle - ang_prev) / (ang_next - ang_prev);

	return val_prev + ratio * (val_next - val_prev);
}

void compileHitranData(AtmosphereModel& atmos, const std::string& hitran_filepath, double wavenumber_min, double wavenumber_max)
{
	if (hitran_filepath.empty())
	{
		return;
	}

	netCDF::NcFile nc_hitran(hitran_filepath, netCDF::NcFile::read);

	std::vector<Species> expanded_species;

	for (const auto& s_orig : atmos.species)
	{
		if (!std::holds_alternative<HitranAbsorption>(s_orig.absorption_model))
		{
			expanded_species.push_back(s_orig);
			continue;
		}

		const auto& hit_config = std::get<HitranAbsorption>(s_orig.absorption_model);
		
		std::vector<hitran::Isotopologue> isos;

		if (hit_config.isotopologue_type == IsotopologueType::All)
		{
			isos = hitran::isos_for_molecule(hit_config.molecule_id);
		}
		else if (hit_config.isotopologue_type == IsotopologueType::Defined)
		{
			for (int loc_id : hit_config.local_isotopologue_id)
			{
				int glob_id = hitran::global_from_mol_local(hit_config.molecule_id, loc_id);
				isos.push_back(hitran::iso_from_global(glob_id));
			}
		}

		if (!hit_config.abundance.empty())
		{
			for (size_t j = 0; j < isos.size(); ++j)
			{
				isos[j].abundance = hit_config.abundance[j];
			}
		}

		if (!hit_config.scalar.empty())
		{
			for (size_t j = 0; j < isos.size(); ++j)
			{
				isos[j].abundance *= hit_config.scalar[j];
			}
		}

		if (hit_config.is_normalize)
		{
			double norm_const = 0.0;

			for (const auto& iso : isos)
			{
				norm_const += iso.abundance;
			}
			
			if (norm_const > 0.0)
			{
				for (auto& iso : isos)
				{
					iso.abundance /= norm_const;
				}
			}
		}

		for (const auto& iso : isos)
		{
			Species s_new = s_orig;
			s_new.name = s_orig.name + "_" + iso.isotopic_formula;

			HitranAbsorption hit_new = hit_config;
			hit_new.isotopologue_data = iso;
			
			hit_new.isotopologue_data.loadQTable(nc_hitran);
			hit_new.cached_lines = hitran::loadLines(nc_hitran, hit_new.isotopologue_data, wavenumber_min, wavenumber_max, false);

			for (size_t k = 0; k < s_new.vertical_mixing_ratio_profile.size(); ++k)
			{
				s_new.vertical_mixing_ratio_profile[k] *= iso.abundance;
				s_new.vertical_number_density_profile[k] *= iso.abundance;
			}

			s_new.absorption_model = hit_new;
			expanded_species.push_back(s_new);
		}
	}

	atmos.species = std::move(expanded_species);
}

}
