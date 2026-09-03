#pragma once

#include <string>
#include <vector>

namespace paad::utils
{

std::string toUpper(std::string s);
std::string toLower(std::string s);
std::vector<std::string> splitString(const std::string& str, char del);

}
