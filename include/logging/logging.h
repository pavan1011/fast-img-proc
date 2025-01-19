#ifndef LOGGING_H
#define LOGGING_H

// Generated using configure_file from fast_img_proc/CMakeLists.txt
#include "version.h"

#include <format>
#include <string_view>
#include <mutex>
#include <queue>
#include <thread>
#include <fstream>

enum class LogLevel {
    DEBUG,
    PROFILE,
    INFO,
    WARN,
    ERROR
};

class Logger {
private:
    struct LogMessage {
        std::string message;
        LogLevel level;
        std::string_view file;
        int line;
        bool newline;
    };

    std::queue<LogMessage> message_queue;
    std::mutex queue_mutex;
    std::ofstream log_file;
    std::thread logging_thread;
    bool should_terminate{false};
    
    static constexpr std::string_view level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::PROFILE: return "PROFILE";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARN: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

    void process_log_queue();
    void output_log(const LogMessage& msg);
    bool should_log(LogLevel level) const;

public:
    Logger(const std::string& filename);
    ~Logger();

    template<typename... Args>
    void log(LogLevel level, const std::string_view file, int line,
             bool newline,  std::format_string<Args...> fmt, Args&&... args) {
        if (should_log(level)) {  // Compile-time optimization happens here
            auto formatted = std::vformat(fmt.get(), std::make_format_args(std::forward<Args>(args)...));
            std::lock_guard<std::mutex> lock(queue_mutex);
            message_queue.emplace(LogMessage{
                std::move(formatted), level, file, line, newline
            });
        }
    }

    static Logger& instance() {
        static Logger logger(std::string(PROJECT_NAME) + ".log");
        return logger;
    }
};

// Macro to log messages to new line
#define LOG(level, fmt, ...) \
    Logger::instance().log(LogLevel::level, \
                          std::string_view(__FILE__), __LINE__, \
                          true, fmt, ##__VA_ARGS__)

// Macro to log messages without new line
#define LOG_NNL(level, fmt, ...) \
    Logger::instance().log(LogLevel::level, \
                          std::string_view(__FILE__), __LINE__, \
                          false, fmt, ##__VA_ARGS__)

#endif // LOGGING_H