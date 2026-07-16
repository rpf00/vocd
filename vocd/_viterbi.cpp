// _viterbi.cpp — compiled Viterbi proposal pass for VOCD.
//
// This is an accelerator, not a second implementation.  vocd/core.py is the
// specification; every line here mirrors it, and tests/test_equivalence.py
// asserts the two produce identical proposals on randomised input.  If they
// ever disagree, this file is wrong.
//
// Only the proposal stage is compiled.  The verification gate stays in Python
// so that SciPy remains the definition of the p-values: an asymptotic
// Kolmogorov series in C++ disagrees with SciPy's exact two-sample
// distribution by up to five orders of magnitude in the far tail, which is
// exactly where VOCD's thresholds sit.  The proposal stage has no such
// hazard -- it is pure arithmetic with no distributional approximation.
//
// State is carried across calls, so `step()` is O(S^2) and the streaming and
// batch paths share one engine, as in Python.
//
// Build:
//   c++ -O3 -std=c++17 -shared -fPIC $(python3 -m pybind11 --includes) \
//       vocd/_viterbi.cpp -o vocd/_viterbi$(python3-config --extension-suffix)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kLog2Pi = 1.8378770664093453;  // log(2*pi)
constexpr double kVarFloor = 1e-12;
constexpr double kClusterVarFloor = 1e-3;

inline double huber(double r, double delta) {
    const double a = std::abs(r);
    return (a <= delta) ? 0.5 * r * r : delta * (a - 0.5 * delta);
}

// Mirrors core._emission.
inline void emission(double y, const std::vector<double>& mu,
                     const std::vector<double>& var, double delta,
                     std::vector<double>& out) {
    for (std::size_t s = 0; s < mu.size(); ++s) {
        const double v = std::max(var[s], kVarFloor);
        const double r = (y - mu[s]) / std::sqrt(v);
        out[s] = huber(r, delta) + 0.5 * (kLog2Pi + std::log(v));
    }
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const std::size_t n = v.size();
    std::sort(v.begin(), v.end());
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Mirrors core._init_regimes, including the deterministic empty-cluster rule:
// reseed to the observation farthest from every current centre, ties broken by
// lowest index.  No RNG anywhere, so results are bit-identical to Python.
void init_regimes(const std::vector<double>& d, int S,
                  std::vector<double>& mu, std::vector<double>& var,
                  int iters = 20) {
    const int n = static_cast<int>(d.size());
    mu.assign(S, 0.0);
    var.assign(S, 1.0);
    if (n == 0) return;

    double mean = 0.0;
    for (double x : d) mean += x;
    mean /= n;
    double gvar = 0.0;
    for (double x : d) gvar += (x - mean) * (x - mean);
    gvar /= n;

    if (n < S) {
        const double mn = *std::min_element(d.begin(), d.end());
        const double mx = *std::max_element(d.begin(), d.end());
        for (int s = 0; s < S; ++s) {
            mu[s] = (S == 1) ? mn : mn + (mx - mn) * s / double(S - 1);
            var[s] = gvar + 1e-6;
        }
        return;
    }

    const double med = median_of(d);
    std::vector<double> absdev(n);
    for (int i = 0; i < n; ++i) absdev[i] = std::abs(d[i] - med);
    double mad = median_of(absdev);
    if (mad < 1e-6) mad = std::sqrt(gvar) + 1e-6;

    std::vector<double> centers(S);
    for (int s = 0; s < S; ++s) {
        centers[s] = med + (S == 1 ? -2.0 * mad
                                   : -2.0 * mad + 4.0 * mad * s / double(S - 1));
    }

    std::vector<int> z(n);
    auto assign = [&]() {
        for (int i = 0; i < n; ++i) {
            int best = 0;
            double bd = kInf;
            for (int s = 0; s < S; ++s) {
                const double e = (d[i] - centers[s]) * (d[i] - centers[s]);
                if (e < bd) { bd = e; best = s; }
            }
            z[i] = best;
        }
    };

    std::vector<double> members;
    for (int it = 0; it < iters; ++it) {
        assign();
        for (int s = 0; s < S; ++s) {
            members.clear();
            for (int i = 0; i < n; ++i) if (z[i] == s) members.push_back(d[i]);
            if (!members.empty()) {
                centers[s] = median_of(members);
            } else {
                // Farthest point from its nearest centre (deterministic).
                int far = 0;
                double fd = -1.0;
                for (int i = 0; i < n; ++i) {
                    double nd = kInf;
                    for (int c = 0; c < S; ++c) {
                        const double e = (d[i] - centers[c]) * (d[i] - centers[c]);
                        if (e < nd) nd = e;
                    }
                    if (nd > fd) { fd = nd; far = i; }
                }
                centers[s] = d[far];
            }
        }
    }

    assign();
    for (int s = 0; s < S; ++s) {
        mu[s] = centers[s];
        members.clear();
        for (int i = 0; i < n; ++i) if (z[i] == s) members.push_back(d[i]);
        if (members.size() > 1) {
            std::vector<double> dev(members.size());
            for (std::size_t i = 0; i < members.size(); ++i)
                dev[i] = std::abs(members[i] - mu[s]);
            const double cmad = median_of(dev);
            var[s] = std::max((1.4826 * cmad) * (1.4826 * cmad), kClusterVarFloor);
        } else {
            var[s] = gvar + 1e-6;
        }
        var[s] = std::max(var[s], kClusterVarFloor);
    }
}

}  // namespace

// Mirrors core._ViterbiPass.  Same buffering, same replay, same recurrence.
class ViterbiPass {
public:
    ViterbiPass(int S, double pen_base, int min_dwell, int init_window,
                int merge_window, double huber_delta, int confirm_k)
        : S_(S), pen_base_(pen_base), min_dwell_(min_dwell),
          init_window_(init_window), merge_window_(merge_window),
          huber_delta_(huber_delta), confirm_k_(confirm_k),
          emis_(S), new_dp_(S), new_dwell_(S) {}

    // Feed one observation; returns change-points confirmed at this step.
    std::vector<int> update(double x) {
        if (!ready_) {
            buf_.push_back(x);
            if (static_cast<int>(buf_.size()) < init_window_) return {};
            return start();
        }
        std::vector<int> out;
        const int cp = step(x);
        if (cp >= 0) out.push_back(cp);
        return out;
    }

    // Force initialisation on a series shorter than init_window.
    std::vector<int> flush() {
        if (!ready_ && !buf_.empty()) return start();
        return {};
    }

private:
    std::vector<int> start() {
        init_regimes(buf_, S_, mu_, var_);
        dp_.assign(S_, 0.0);
        dwell_.assign(S_, 1);
        ready_ = true;
        std::vector<int> out;
        for (double v : buf_) {
            const int cp = step(v);
            if (cp >= 0) out.push_back(cp);
        }
        buf_.clear();
        return out;
    }

    int step(double x) {
        emission(x, mu_, var_, huber_delta_, emis_);

        for (int s = 0; s < S_; ++s) {
            double best_cost = kInf;
            int best_k = s;
            for (int k = 0; k < S_; ++k) {
                if (k != s) {
                    if (dwell_[k] < min_dwell_) continue;
                    if (emis_[k] - emis_[s] <= 0.0) continue;
                }
                const double c = dp_[k] + (k == s ? 0.0 : pen_base_);
                if (c < best_cost) { best_cost = c; best_k = k; }  // ties -> low k
            }
            new_dp_[s] = best_cost + emis_[s];
            new_dwell_[s] = (best_k == s) ? dwell_[s] + 1 : 1;
        }
        dp_ = new_dp_;
        dwell_ = new_dwell_;

        int best_state = 0;
        double bv = kInf;
        for (int s = 0; s < S_; ++s) if (dp_[s] < bv) { bv = dp_[s]; best_state = s; }

        const int t = t_++;
        int cp_out = -1;
        if (best_state != confirmed_state_) {
            if (pending_state_ == best_state) {
                ++pending_count_;
            } else {
                pending_state_ = best_state;
                pending_count_ = 1;
                pending_cp_ = t;
            }
            if (pending_count_ >= confirm_k_) {
                if (pending_cp_ - last_cp_ >= merge_window_) {
                    cp_out = pending_cp_;
                    last_cp_ = pending_cp_;
                }
                confirmed_state_ = pending_state_;
                pending_state_ = -1;
                pending_count_ = 0;
                pending_cp_ = -1;
            }
        } else {
            pending_state_ = -1;
            pending_count_ = 0;
            pending_cp_ = -1;
        }
        return cp_out;
    }

    int S_, min_dwell_, init_window_, merge_window_, confirm_k_;
    double pen_base_, huber_delta_;
    std::vector<double> buf_, mu_, var_, dp_, emis_, new_dp_;
    std::vector<int> dwell_, new_dwell_;
    bool ready_ = false;
    int t_ = 0, confirmed_state_ = 0, last_cp_ = 0;
    int pending_state_ = -1, pending_count_ = 0, pending_cp_ = -1;
};

PYBIND11_MODULE(_viterbi, m) {
    m.doc() = "Compiled Viterbi proposal pass for VOCD (mirrors vocd.core).";
    py::class_<ViterbiPass>(m, "ViterbiPass")
        .def(py::init<int, double, int, int, int, double, int>(),
             py::arg("S"), py::arg("pen_base"), py::arg("min_dwell"),
             py::arg("init_window"), py::arg("merge_window"),
             py::arg("huber_delta"), py::arg("confirm_k"))
        .def("update", &ViterbiPass::update, py::arg("x"))
        .def("flush", &ViterbiPass::flush);
}
