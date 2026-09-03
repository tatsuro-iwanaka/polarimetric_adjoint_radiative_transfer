#pragma once

#include <string>

namespace paad::units
{

enum class UnitDim
{
	Length, 
	Area, 
	Mass, 
	Temperature, 
	Pressure, 
	NumberDensity, 
	ColumnNumberDensity, 
	Wavenumber, 
	Dimensionless
};

class UnitInfo
{	
	public:
		UnitDim dim;
		double to_si;
};

UnitInfo getUnitInfo(const std::string& unit_input);
double scaleUnit(const std::string& unit_input, const std::string& unit_output);

}
