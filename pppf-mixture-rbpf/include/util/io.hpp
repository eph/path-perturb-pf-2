#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace util {

inline void ensure_dir(const std::filesystem::path& p) {
  if (p.empty()) return;
  std::error_code ec;
  std::filesystem::create_directories(p, ec);
  if (ec) throw std::runtime_error("ensure_dir failed: " + p.string());
}

inline void write_text_file(const std::filesystem::path& path, const std::string& content) {
  ensure_dir(path.parent_path());
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to open: " + path.string());
  out << content;
}

inline void write_csv_header(std::ofstream& out, const std::vector<std::string>& cols) {
  for (std::size_t i = 0; i < cols.size(); ++i) {
    if (i) out << ",";
    out << cols[i];
  }
  out << "\n";
}

template <typename... Ts>
inline void write_csv_row(std::ofstream& out, const Ts&... xs) {
  bool first = true;
  auto write_one = [&](const auto& v) {
    if (!first) out << ",";
    first = false;
    out << v;
  };
  (write_one(xs), ...);
  out << "\n";
}

inline std::string json_escape(const std::string& s) {
  std::ostringstream oss;
  for (char c : s) {
    switch (c) {
      case '\"':
        oss << "\\\"";
        break;
      case '\\':
        oss << "\\\\";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << c;
    }
  }
  return oss.str();
}

}  // namespace util

