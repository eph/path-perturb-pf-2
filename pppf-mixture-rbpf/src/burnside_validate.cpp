#include "models/burnside.hpp"
#include "util/io.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string format_tag(double v) {
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << std::setprecision(6) << v;
  std::string s = oss.str();
  // Trim trailing zeros and a trailing dot.
  while (!s.empty() && s.back() == '0') s.pop_back();
  if (!s.empty() && s.back() == '.') s.pop_back();
  if (s.empty()) s = "0";
  return s;
}
}  // namespace

struct CliOptions {
  int T = 40;
  std::vector<double> shock_sizes = {0.5, 1.0, 2.0};
  std::filesystem::path out_dir = "output/burnside";
};

std::vector<double> parse_csv_doubles(const std::string& s) {
  std::vector<double> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    out.push_back(std::stod(tok));
  }
  return out;
}

CliOptions parse_args(int argc, char** argv) {
  CliOptions opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto need = [&](const std::string& flag) {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      return std::string(argv[++i]);
    };
    if (a == "--T") {
      opt.T = std::stoi(need(a));
    } else if (a == "--shock_sizes") {
      opt.shock_sizes = parse_csv_doubles(need(a));
      if (opt.shock_sizes.empty()) throw std::invalid_argument("shock_sizes must be nonempty");
    } else if (a == "--out_dir") {
      opt.out_dir = need(a);
    } else if (a == "--help" || a == "-h") {
      std::cout << "Usage: burnside_validate [--T int] [--shock_sizes a,b,c] [--out_dir path]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown flag: " + a);
    }
  }
  return opt;
}

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);

    models::BurnsideParams p;
    p.beta = 0.99;
    p.gamma = 2.0;
    p.rho = 0.95;
    p.mu = 0.0;
    p.sigma = 0.006;

    const int T = cli.T;
    const std::vector<double> shock_sizes = cli.shock_sizes;

    const std::filesystem::path out_dir = cli.out_dir;
    util::ensure_dir(out_dir);

    // Clean prior IRF artifacts to keep output deterministic.
    for (const auto& ent : std::filesystem::directory_iterator(out_dir)) {
      if (!ent.is_regular_file()) continue;
      const auto name = ent.path().filename().string();
      if (name.rfind("irf_", 0) != 0) continue;
      const auto ext = ent.path().extension().string();
      if (ext == ".csv" || ext == ".png") {
        std::error_code ec;
        std::filesystem::remove(ent.path(), ec);
      }
    }

    for (double s : shock_sizes) {
      const double eps0 = s * p.sigma;
      std::vector<double> x(T);
      for (int t = 0; t < T; ++t) x[t] = p.mu + std::pow(p.rho, t) * eps0;

      const std::string tag = format_tag(s);
      const std::filesystem::path csv_path = out_dir / ("irf_" + tag + ".csv");
      std::ofstream out(csv_path);
      if (!out) throw std::runtime_error("failed to open: " + csv_path.string());
      util::write_csv_header(out, {"t", "x", "v_exact", "v_ce", "v_ut"});

      for (int t = 0; t < T; ++t) {
        const double v_exact = models::burnside_v_exact(p, x[t]);
        const double v_ce = models::burnside_v_ce(p, x[t]);
        const double v_ut = models::burnside_v_ut_one_step(p, x[t]);
        util::write_csv_row(out, t, x[t], v_exact, v_ce, v_ut);
      }

      std::cout << "Wrote " << csv_path.string() << "\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "burnside_validate error: " << e.what() << "\n";
    return 1;
  }
}
