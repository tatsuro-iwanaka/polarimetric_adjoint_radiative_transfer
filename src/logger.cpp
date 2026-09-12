#include "logger.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

namespace paad::utils
{

Logger& Logger::getInstance()
{
	static Logger instance;
	return instance;
}

Logger::~Logger()
{
	if (file_stream_ && file_stream_->is_open())
	{
		file_stream_->close();
	}
}

void Logger::setLogFile(const std::string& filepath)
{
	std::lock_guard<std::mutex> lock(mutex_);

	if (file_stream_ && file_stream_->is_open())
	{
		file_stream_->close();
	}
	
	file_stream_ = std::make_unique<std::ofstream>(filepath, std::ios::out | std::ios::trunc);

	if (!file_stream_->is_open())
	{
		std::cerr << "[Logger Error] Cannot open log file: " << filepath << std::endl;
	}
}

std::string Logger::levelToString(LogLevel level) const
{
	switch (level)
	{
		case LogLevel::INFO: return "INFO";
		case LogLevel::WARNING: return "WARNING";
		case LogLevel::ERROR: return "ERROR";
		case LogLevel::DEBUG: return "DEBUG";
		default: return "UNKNOWN";
	}
}

void Logger::log(LogLevel level, const std::string& message)
{
	std::lock_guard<std::mutex> lock(mutex_);
	
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);
	
	std::stringstream ss;
	ss << "[" << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S") << "] " << "[" << levelToString(level) << "] " << message << "\n";
	
	if (file_stream_ && file_stream_->is_open())
	{
		*file_stream_ << ss.str();
		file_stream_->flush();
	}
	
	if (level == LogLevel::ERROR)
	{
		std::cerr << ss.str();
	}
	else
	{
		std::cout << ss.str();
	}
}

}
