#pragma once

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <streambuf>
#include <string>

class TeeBuf : public std::streambuf {
public:
    TeeBuf(std::streambuf* primary, std::streambuf* secondary)
        : primary_(primary), secondary_(secondary) {}

protected:
    int overflow(int c) override {
        if (c == EOF) return 0;
        const int a = primary_->sputc(static_cast<char>(c));
        const int b = secondary_->sputc(static_cast<char>(c));
        return (a == EOF || b == EOF) ? EOF : c;
    }

    int sync() override {
        int r = 0;
        if (primary_->pubsync() != 0) r = -1;
        if (secondary_->pubsync() != 0) r = -1;
        return r;
    }

    std::streamsize xsputn(const char* s, std::streamsize n) override {
        primary_->sputn(s, n);
        secondary_->sputn(s, n);
        return n;
    }

private:
    std::streambuf* primary_;
    std::streambuf* secondary_;
};

class ConsoleLogTee {
public:
    bool enable(const std::string& path) {
        if (enabled_) return true;

        std::error_code ec;
        const auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent, ec);

        file_.open(path, std::ios::out | std::ios::trunc);
        if (!file_.is_open()) {
            std::cerr << "[ConsoleLog] failed to open: " << path << std::endl;
            return false;
        }

        orig_cout_buf_ = std::cout.rdbuf();
        orig_cerr_buf_ = std::cerr.rdbuf();
        cout_tee_ = std::make_unique<TeeBuf>(orig_cout_buf_, file_.rdbuf());
        cerr_tee_ = std::make_unique<TeeBuf>(orig_cerr_buf_, file_.rdbuf());
        std::cout.rdbuf(cout_tee_.get());
        std::cerr.rdbuf(cerr_tee_.get());

        enabled_ = true;
        log_path_ = path;
        return true;
    }

    bool enableAuto(const std::string& report_dir, const std::string& seq) {
        char buf[32];
        const std::time_t now = std::time(nullptr);
        std::strftime(buf, sizeof(buf), "%F-%H%M%S", std::localtime(&now));

        std::string seq_id = "00";
        if (seq.size() >= 3) seq_id = seq.substr(1, 2);

        std::string dir = report_dir;
        while (!dir.empty() && dir.back() == '/') dir.pop_back();
        const std::string path = dir + "/seq" + seq_id + "_run_" + buf + ".log";
        return enable(path);
    }

    const std::string& path() const { return log_path_; }
    bool isEnabled() const { return enabled_; }

    ~ConsoleLogTee() {
        if (!enabled_) return;
        std::cout.flush();
        std::cerr.flush();
        std::cout.rdbuf(orig_cout_buf_);
        std::cerr.rdbuf(orig_cerr_buf_);
        file_.close();
    }

private:
    std::ofstream file_;
    std::streambuf* orig_cout_buf_ = nullptr;
    std::streambuf* orig_cerr_buf_ = nullptr;
    std::unique_ptr<TeeBuf> cout_tee_;
    std::unique_ptr<TeeBuf> cerr_tee_;
    std::string log_path_;
    bool enabled_ = false;
};
