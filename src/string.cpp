#include "string.hpp"

#include <cctype>
#include <algorithm>

namespace paad::utils
{

std::string toUpper(std::string s)
{
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::toupper(c); });
	return s;
}

std::string toLower(std::string s)
{
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
	return s;
}

std::vector<std::string> splitString(const std::string& str, char del) 
{
	std::vector<std::string> result;
	std::string::size_type first = 0;
	std::string::size_type last = str.find_first_of(del);

	while (first < str.size())
	{
		std::string subStr(str, first, last - first);
		result.push_back(subStr);

		first = last + 1;
		last = str.find_first_of(del, first);

		if (last == std::string::npos)
		{
			last = str.size();
		}
	}

	return result;
}

}
