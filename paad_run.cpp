#include <iostream>
#include <exception>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numbers>

#include "paad.hpp"

int main(int argc, char** argv)
{
	try
	{
		if (argc <= 1)
		{
			std::cerr << "Usage: " << argv[0] << " <config.json>\n";
			return 1;
		}

		paad::RadiativeTransfer rt(argv[1]);

		rt.run();
		
		rt.exportResult();
	}
	catch (const std::exception& e)
	{
		std::cerr << "Fatal Error: " << e.what() << '\n';
		return 1;
	}

	return 0;
}
