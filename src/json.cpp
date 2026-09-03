#include "json.hpp"

#include <sstream>
#include <cctype>
#include <functional>

namespace paad::utils::json
{

const JsonNode& JsonNode::operator[](const std::string& key) const
{
	if (type != Type::Object || !obj_val)
	{
		throw std::runtime_error("Not an object");
	}

	auto it = obj_val->find(key);

	if (it == obj_val->end())
	{
		throw std::runtime_error("Key not found: " + key);
	}

	return it->second;
}

JsonNode& JsonNode::operator[](const std::string& key)
{
	if (type != Type::Object)
	{
		type = Type::Object;
		obj_val = std::make_shared<std::map<std::string, JsonNode>>();
	}

	return (*obj_val)[key];
}

const JsonNode& JsonNode::operator[](size_t index) const
{
	if (type != Type::Array || !arr_val)
	{
		throw std::runtime_error("Not an array");
	}

	if (index >= arr_val->size())
	{
		throw std::runtime_error("Index out of bounds");
	}

	return (*arr_val)[index];
}

void JsonNode::push_back(const JsonNode& val)
{
	if (type != Type::Array)
	{
		type = Type::Array;
		arr_val = std::make_shared<std::vector<JsonNode>>();
	}

	arr_val->push_back(val);
}

bool JsonNode::has(const std::string& key) const
{
	if (type != Type::Object || !obj_val)
	{
		return false;
	}

	return obj_val->find(key) != obj_val->end();
}

size_t JsonNode::size() const
{
	if (type == Type::Array && arr_val)
	{
		return arr_val->size();
	}

	if (type == Type::Object && obj_val)
	{
		return obj_val->size();
	}

	return 0;
}

double JsonNode::get_number() const
{
	if (type != Type::Number)
	{
		throw std::runtime_error("Not a number");
	}

	return num_val;
}

std::string JsonNode::get_string() const
{
	if (type != Type::String)
	{
		throw std::runtime_error("Not a string");
	}

	return str_val;
}

bool JsonNode::get_bool() const
{
	if (type != Type::Bool)
	{
		throw std::runtime_error("Not a bool");
	}

	return bool_val;
}

JsonNode parse(const std::string& text)
{
	size_t pos = 0;

	auto skipWhitespace = [&]()
	{
		while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
		{
			pos++;
		}
	};

	std::function<JsonNode()> parseValue = [&]() -> JsonNode
	{
		skipWhitespace();

		if (pos >= text.size())
		{
			throw std::runtime_error("Unexpected end of JSON input");
		}

		char c = text[pos];
		if (c == '{')
		{
			pos++;
			JsonNode obj = JsonNode::make_object();
			skipWhitespace();

			if (pos < text.size() && text[pos] == '}')
			{
				pos++; return obj;
			}
			
			while (pos < text.size())
			{
				skipWhitespace();

				if (text[pos] != '"')
				{
					throw std::runtime_error("Expected string key in object");
				}

				std::string key = parseValue().get_string();
				
				skipWhitespace();
				
				if (text[pos] != ':')
				{
					throw std::runtime_error("Expected ':'");
				}

				pos++;
				
				obj[key] = parseValue();
				
				skipWhitespace();

				if (pos < text.size() && text[pos] == ',')
				{
					pos++;
				}
				else if(pos < text.size() && text[pos] == '}')
				{
					pos++; return obj;
				}
				else
				{
					throw std::runtime_error("Expected ',' or '}' in object");
				}
			}

			throw std::runtime_error("Unclosed object");
		}
		else if (c == '[')
		{
			pos++;
			JsonNode arr = JsonNode::make_array();
			skipWhitespace();

			if (pos < text.size() && text[pos] == ']')
			{
				pos++; return arr;
			}
			
			while (pos < text.size())
			{
				arr.push_back(parseValue());
				skipWhitespace();

				if (pos < text.size() && text[pos] == ',')
				{
					pos++;
				}
				else if (pos < text.size() && text[pos] == ']')
				{
					pos++; return arr;
				}
				else
				{
					throw std::runtime_error("Expected ',' or ']' in array");
				}
			}

			throw std::runtime_error("Unclosed array");
		}
		else if (c == '"')
		{
			pos++;
			std::string s;

			while (pos < text.size() && text[pos] != '"')
			{
				if (text[pos] == '\\')
				{
					pos++;

					if (pos < text.size())
					{
						s += text[pos];
					}
				}
				else
				{
					s += text[pos];
				}

				pos++;
			}

			pos++;

			return JsonNode(s);
		}
		else if (c == 't' || c == 'f')
		{
			if (text.compare(pos, 4, "true") == 0)
			{
				pos += 4; return JsonNode(true);
			}

			if (text.compare(pos, 5, "false") == 0)
			{
				pos += 5; return JsonNode(false);
			}

			throw std::runtime_error("Invalid boolean");
		}
		else if (c == 'n')
		{
			if (text.compare(pos, 4, "null") == 0)
			{
				pos += 4; return JsonNode();
			}

			throw std::runtime_error("Invalid null");
		}
		else
		{
			size_t start = pos;

			while (pos < text.size() && (std::isdigit(text[pos]) || text[pos] == '-' || text[pos] == '.' || text[pos] == 'e' || text[pos] == 'E' || text[pos] == '+'))
			{
				pos++;
			}

			if (start == pos)
			{
				throw std::runtime_error("Invalid JSON character");
			}

			return JsonNode(std::stod(text.substr(start, pos - start)));
		}
	};

	return parseValue();
}

static std::string dumpNodeInternal(const JsonNode& node, int indent, int current_indent)
{
	std::string indStr(current_indent, ' ');
	
	if (node.type == JsonNode::Type::Null)
	{
		return "null";
	}

	if (node.type == JsonNode::Type::Bool)
	{
		return node.bool_val ? "true" : "false";
	}

	if (node.type == JsonNode::Type::Number)
	{
		std::ostringstream oss;
		oss << node.num_val;
		return oss.str();
	}

	if (node.type == JsonNode::Type::String)
	{
		return "\"" + node.str_val + "\"";
	}

	if (node.type == JsonNode::Type::Array)
	{
		if (!node.arr_val || node.arr_val->empty())
		{
			return "[]";
		}

		std::string res = "[\n";

		for (size_t i = 0; i < node.arr_val->size(); ++i)
		{
			res += std::string(current_indent + indent, ' ') + dumpNodeInternal((*node.arr_val)[i], indent, current_indent + indent);

			if (i < node.arr_val->size() - 1) \
			{
				res += ",";
			}

			res += "\n";
		}

		res += indStr + "]";

		return res;
	}
	if (node.type == JsonNode::Type::Object)
	{
		if (!node.obj_val || node.obj_val->empty())
		{
			return "{}";
		}

		std::string res = "{\n";
		size_t count = 0;

		for (auto it = node.obj_val->begin(); it != node.obj_val->end(); ++it, ++count)
		{
			res += std::string(current_indent + indent, ' ') + "\"" + it->first + "\": " + dumpNodeInternal(it->second, indent, current_indent + indent);

			if (count < node.obj_val->size() - 1)
			{
				res += ",";
			}
			
			res += "\n";
		}

		res += indStr + "}";

		return res;
	}
	return "";
}

std::string dump(const JsonNode& node, int indent)
{
	return dumpNodeInternal(node, indent, 0);
}

}
