#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

namespace paad::utils::json
{

class JsonNode
{
	public:
		enum class Type { Null, Bool, Number, String, Array, Object };
		Type type;

		std::string str_val;
		double num_val{0.0};
		bool bool_val{false};
		std::shared_ptr<std::vector<JsonNode>> arr_val;
		std::shared_ptr<std::map<std::string, JsonNode>> obj_val;

		JsonNode() : type(Type::Null) {}
		JsonNode(bool b) : type(Type::Bool), bool_val(b) {}
		JsonNode(double d) : type(Type::Number), num_val(d) {}
		JsonNode(const std::string& s) : type(Type::String), str_val(s) {}

		static JsonNode make_array() { JsonNode n; n.type = Type::Array; n.arr_val = std::make_shared<std::vector<JsonNode>>(); return n; }
		static JsonNode make_object() { JsonNode n; n.type = Type::Object; n.obj_val = std::make_shared<std::map<std::string, JsonNode>>(); return n; }

		const JsonNode& operator[](const std::string& key) const;
		JsonNode& operator[](const std::string& key);
		const JsonNode& operator[](size_t index) const;
		void push_back(const JsonNode& val);

		bool has(const std::string& key) const;
		size_t size() const;

		double get_number() const;
		std::string get_string() const;
		bool get_bool() const;
};

JsonNode parse(const std::string& text);
std::string dump(const JsonNode& node, int indent = 4);

}
