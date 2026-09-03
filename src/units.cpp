#include "units.hpp"
#include "string.hpp"
#include <stdexcept>

namespace paad::units
{

UnitInfo getUnitInfo(const std::string& unit_input)
{
	std::string u = utils::toLower(unit_input);

	if (u == "km") return { UnitDim::Length, 1.0E3 };
	if (u == "m") return { UnitDim::Length, 1.0 };
	if (u == "cm") return { UnitDim::Length, 1.0E-2 };
	if (u == "mm") return { UnitDim::Length, 1.0E-3 };
	if (u == "um" || u == "micron") return { UnitDim::Length, 1.0E-6 };
	if (u == "nm") return { UnitDim::Length, 1.0E-9 };
	
	if (u == "km2") return { UnitDim::Area, 1.0E6 };
	if (u == "m2") return { UnitDim::Area, 1.0 };
	if (u == "cm2") return { UnitDim::Area, 1.0E-4 };
	if (u == "mm2") return { UnitDim::Area, 1.0E-6 };
	if (u == "um2" || u == "micron2") return { UnitDim::Area, 1.0E-12 };
	if (u == "nm2") return { UnitDim::Area, 1.0E-18 };
	
	if (u == "kg") return { UnitDim::Mass, 1.0 };
	if (u == "g") return { UnitDim::Mass, 1.0E-3 };
	
	if (u == "k") return { UnitDim::Temperature, 1.0};
	
	if (u == "pa" || u == "pascal") return { UnitDim::Pressure, 1.0};
	if (u == "hpa" || u == "mbar") return { UnitDim::Pressure, 1.0E2}; 
	if (u == "bar") return { UnitDim::Pressure, 1.0E5}; 
	if (u == "atm") return { UnitDim::Pressure, 101325.0};
	if (u == "torr" || u == "mmhg") return { UnitDim::Pressure, 101325.0 / 760.0};
	
	if (u == "km-3" || u == "1/km3") return { UnitDim::NumberDensity, 1.0E-9 };
	if (u == "m-3" || u == "1/m3") return { UnitDim::NumberDensity, 1.0 };
	if (u == "cm-3" || u == "1/cm3") return { UnitDim::NumberDensity, 1.0E6 };
	if (u == "mm-3" || u == "1/mm3") return { UnitDim::NumberDensity, 1.0E9 };
	if (u == "um-3" || u == "1/um3" || u == "micron-3" || u == "1/micron3") return { UnitDim::NumberDensity, 1.0E18 };
	if (u == "nm-3" || u == "1/nm3") return { UnitDim::NumberDensity, 1.0E27 };
	
	if (u == "km-2" || u == "1/km2") return { UnitDim::ColumnNumberDensity, 1.0E-6 };
	if (u == "m-2" || u == "1/m2") return { UnitDim::ColumnNumberDensity, 1.0 };
	if (u == "cm-2" || u == "1/cm2") return { UnitDim::ColumnNumberDensity, 1.0E4 };
	if (u == "mm-2" || u == "1/mm2") return { UnitDim::ColumnNumberDensity, 1.0E6 };
	if (u == "um-2" || u == "1/um2" || u == "micron-2" || u == "1/micron2") return { UnitDim::ColumnNumberDensity, 1.0E12 };
	if (u == "nm-2" || u == "1/nm2") return { UnitDim::ColumnNumberDensity, 1.0E18 };
	
	if (u == "km-1" || u == "1/km") return { UnitDim::Wavenumber, 1.0E-3 };
	if (u == "m-1" || u == "1/m") return { UnitDim::Wavenumber, 1.0 };
	if (u == "cm-1" || u == "1/cm" || u == "kayser") return { UnitDim::Wavenumber, 1.0E2 };
	if (u == "um-1" || u == "1/um" || u == "micron-1"   || u == "1/micron") return { UnitDim::Wavenumber, 1.0E6 };
	if (u == "nm-1" || u == "1/nm") return { UnitDim::Wavenumber, 1.0E9 };
	
	if (u == "1" || u == "none" || u == "mol/mol" || u == "dimensionless") return { UnitDim::Dimensionless, 1.0 };
	if (u == "ppm") return { UnitDim::Dimensionless, 1.0E-6 };
	if (u == "ppb") return { UnitDim::Dimensionless, 1.0E-9 };
	if (u == "ppt") return { UnitDim::Dimensionless, 1.0E-12 };
	
	throw std::runtime_error("Unknown unit: " + unit_input);
}

double scaleUnit(const std::string& unit_input, const std::string& unit_output)
{
	UnitInfo inInfo = getUnitInfo(unit_input);
	UnitInfo outInfo = getUnitInfo(unit_output);

	if (inInfo.dim != outInfo.dim)
	{
		throw std::runtime_error("Unit dimension mismatch: cannot convert from `" + unit_input + "` to `" + unit_output + "`");
	}
	
	return inInfo.to_si / outInfo.to_si;
}

}
