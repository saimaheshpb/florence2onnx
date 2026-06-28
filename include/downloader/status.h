#pragma once

#include <string>

namespace florence2 {

class IStatus {
public:
    virtual ~IStatus() = default;

    virtual void set_error(const std::string& error) = 0;
    virtual void set_message(const std::string& message) = 0;
    virtual void set_progress(float progress) = 0;

    virtual std::string get_error() const = 0;
    virtual std::string get_message() const = 0;
    virtual float get_progress() const = 0;
};

// Concrete implementation of IStatus
class DownloadStatus : public IStatus {
private:
    std::string error_;
    std::string message_;
    float progress_ = 0.0f;

public:
    void set_error(const std::string& error) override { error_ = error; }
    void set_message(const std::string& message) override { message_ = message; }
    void set_progress(float progress) override { progress_ = progress; }

    std::string get_error() const override { return error_; }
    std::string get_message() const override { return message_; }
    float get_progress() const override { return progress_; }
};

} // namespace florence2