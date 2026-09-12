#pragma once

#include <string>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

namespace paad::utils
{

enum class LogLevel
{
	INFO,
	WARNING,
	ERROR,
	DEBUG
};

class Logger
{
	public:
		static Logger& getInstance();

		void setLogFile(const std::string& filepath);
		void log(LogLevel level, const std::string& message);

		Logger(const Logger&) = delete;
		Logger& operator=(const Logger&) = delete;

	private:
		Logger() = default;
		~Logger();

		std::string levelToString(LogLevel level) const;

		std::unique_ptr<std::ofstream> file_stream_;
		std::mutex mutex_;
};


#define PAAD_INFO(msg) do { std::stringstream ss; ss << msg; paad::utils::Logger::getInstance().log(paad::utils::LogLevel::INFO, ss.str()); } while(0)
#define PAAD_WARN(msg) do { std::stringstream ss; ss << msg; paad::utils::Logger::getInstance().log(paad::utils::LogLevel::WARNING, ss.str()); } while(0)
#define PAAD_ERROR(msg) do { std::stringstream ss; ss << msg; paad::utils::Logger::getInstance().log(paad::utils::LogLevel::ERROR, ss.str()); } while(0)
#define PAAD_DEBUG(msg) do { std::stringstream ss; ss << msg; paad::utils::Logger::getInstance().log(paad::utils::LogLevel::DEBUG, ss.str()); } while(0)

}
