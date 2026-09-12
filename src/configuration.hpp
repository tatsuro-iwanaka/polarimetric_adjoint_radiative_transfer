#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <stdexcept>
#include <fstream>

#include "types.hpp"
#include "geometry.hpp"
#include "json.hpp"
#include "string.hpp"
#include "atmosphere.hpp"

namespace paad::core { struct Simulation; struct Spectral; }
namespace paad::atmosphere { struct AtmosphereModel; }

namespace paad::configuration
{

struct ConfigurationData
{
	core::Simulation simulation;
	core::Spectral spectral;
	geometry::Geometry geometry;
	atmosphere::AtmosphereModel atmosphere;
};

class ConfigParser
{
	public:
		ConfigurationData import_json(const std::string& filepath);
		void export_json(const ConfigurationData& config, const std::string& filepath);

	private:
		utils::json::JsonNode root_;

		core::Simulation parseSimulation_();
		core::Spectral parseSpectral_();
		geometry::Geometry parseGeometry_();
		atmosphere::AtmosphereModel parseAtmosphere_();

		template<typename EnumT>
		static EnumT parseEnum_(const std::string& s, const std::map<std::string, EnumT>& mapping)
		{
			std::string s_u = utils::toUpper(s);
			auto it = mapping.find(s_u);
			if (it != mapping.end()) return it->second;
			throw std::runtime_error("Unknown TypeName: " + s);
		}

		template<typename EnumT>
		static std::string enumToString_(EnumT val, const std::map<std::string, EnumT>& mapping)
		{
			for (const auto& pair : mapping)
			{
				if (pair.second == val) return utils::toLower(pair.first);
			}
			return "unknown";
		}
};

}
