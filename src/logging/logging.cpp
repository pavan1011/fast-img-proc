#include "logging/logging.h"
#include <iostream>
#include <chrono>

Logger::Logger(const std::string& filename) : log_file(filename, std::ios::app) {
    if (!log_file.is_open()) {
        throw std::runtime_error("Failed to open log file: " + filename);
    }
    logging_thread = std::thread(&Logger::process_log_queue, this);
}

Logger::~Logger() {
    should_terminate = true;
    if (logging_thread.joinable()) {
        logging_thread.join();
    }
    if (log_file.is_open()) {
        log_file.close();
    }
}

void Logger::process_log_queue() {
    while (!should_terminate) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (!message_queue.empty()) {
            auto msg = message_queue.front();
            message_queue.pop();
            lock.unlock();
            output_log(msg);
        }
    }
}

void Logger::output_log(const LogMessage& msg) {
    auto timestamp = std::chrono::system_clock::now();

    if (msg.level == LogLevel::WARN || msg.level == LogLevel::ERROR) {
        if (!msg.newline){
            log_file << msg.message;
            std::cerr << msg.message;
        }else{
            auto formatted = std::format("[{}] [{}] {} v{} : {} \n",
                level_to_string(msg.level),
                std::format("{:%Y-%m-%d %H:%M:%S}", timestamp),
                std::string(PROJECT_NAME),
                std::string(PROJECT_VERSION_STRING),
                msg.message
            );
            log_file << formatted;
            log_file.flush();
            std::cerr << formatted;
        }
        
    } else {
        if (!msg.newline){
            log_file << msg.message;
            std::cerr << msg.message;
        }else{
            auto formatted = std::format("[{}] [{}] {}:{} {}\n",
                level_to_string(msg.level),
                msg.newline ? std::format("{:%Y-%m-%d %H:%M:%S}", timestamp) : "",
                msg.file,
                msg.line,
                msg.message);
            log_file << formatted;
            log_file.flush();
            std::cout << formatted;
        }
    }    
}

bool Logger::should_log(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:
            #ifdef DEBUG_BUILD
            return true;
            #else
            return false;
            #endif
            
        case LogLevel::PROFILE:
            #ifdef PROFILE_BUILD
            return true;
            #else
            return false;
            #endif
            
        case LogLevel::INFO:
            #if defined(DEBUG_BUILD) || defined(VERBOSE_BUILD)
            return true;
            #else
            return false;
            #endif
            
        case LogLevel::WARN:
        case LogLevel::ERROR:
            return true;
    }
    return false;
}