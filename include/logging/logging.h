/**
 * @file logging.h
 * @brief Interface to display log messages based on build type
*/
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

/**
 * @enum LogLevel
 * @brief Defines severity levels for log messages
 * 
 * Levels are ordered from least to most severe:
 * - DEBUG: Detailed information for debugging
 * - PROFILE: Performance and timing information
 * - INFO: General information about program execution
 * - WARN: Warnings that don't prevent execution
 * - ERROR: Serious issues that may affect execution
 */
enum class LogLevel {
    DEBUG,
    PROFILE,
    INFO,
    WARN,
    ERROR
};

/**
 * @class Logger
 * @brief Thread-safe singleton logger with message queuing
 * 
 * Provides asynchronous logging capabilities with different severity levels.
 * Messages are queued and processed in a separate thread to minimize impact
 * on the main program execution.
 */
class Logger {
private:

    /**
     * @brief Constructs a logger that writes to specified file
     * @param filename Name of the log file
     */
    Logger(const std::string& filename);

    /**
     * @brief Copy constructor (deleted)
     * 
     * Deleted to enforce the Singleton pattern by preventing
     * creation of additional copies
     */
    Logger(const Logger&) = delete;

    /**
     * @brief Copy assignment operator (deleted)
     * 
     * Deleted to enforce the Singleton pattern by preventing
     * assignment between Logger instances
     * @return Reference to this Logger
     */
    Logger& operator=(const Logger&) = delete;

    /**
     * @struct Logger::LogMessage
     * @brief Container for a single log message and its metadata
     * 
     * @var message The formatted log message
     * @var level Severity level of the message
     * @var file Source file where the log was called
     * @var line Line number in source file
     * @var newline Whether to append newline to message
     */
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
    
    /**
     * @brief Converts LogLevel to its string representation
     * @param level The LogLevel to convert
     * @return String view of the level name
     */
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
    ~Logger();

    template<typename... Args>

    /**
     * @brief Logs a formatted message with specified severity
     * @tparam Args Template parameter pack for format arguments
     * @param level Severity level of the message
     * @param file Source file name
     * @param line Source line number
     * @param newline Whether to append newline
     * @param fmt Format string
     * @param args Arguments for format string
     */
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

    /**
     * @brief Returns singleton instance of the logger
     * @return Reference to the global logger instance
     */
    static Logger& instance() {
        static Logger logger(std::string(PROJECT_NAME) + ".log");
        return logger;
    }
};

/**
 * @def LOG(level, fmt, ...)
 * @brief Macro for logging messages with automatic newline
 * @param level Severity level (DEBUG, PROFILE, INFO, WARN, ERROR)
 * @param fmt Format string
 * @param ... Variable arguments for format string
 */
#define LOG(level, fmt, ...) \
    Logger::instance().log(LogLevel::level, \
                          std::string_view(__FILE__), __LINE__, \
                          true, fmt, ##__VA_ARGS__)

/**
 * @def LOG_NNL(level, fmt, ...)
 * @brief Macro for logging messages without newline
 * @param level Severity level (DEBUG, PROFILE, INFO, WARN, ERROR)
 * @param fmt Format string
 * @param ... Variable arguments for format string
 */
#define LOG_NNL(level, fmt, ...) \
    Logger::instance().log(LogLevel::level, \
                          std::string_view(__FILE__), __LINE__, \
                          false, fmt, ##__VA_ARGS__)

#endif // LOGGING_H