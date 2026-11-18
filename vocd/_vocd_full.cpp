// _vocd_full.cpp — VOCD `detect` entirely in C++.
//
// One call runs everything vocd.core.detect does: fit the regime bank on the
// init window, run the precision and recall Viterbi passes, pool their
// proposals, and score every candidate with the MW + exact-KS gate
// (_gate_impl.hpp).  Thresholding (`VOCDResult.changepoints`) stays in
// Python: it is a handful of comparisons and is where the error budget is
// chosen, after detection.
//
// vocd/core.py is the specification.  This file mirrors it line for line and
// reproduces NumPy's arithmetic where NumPy's choices matter for the last
// bit, so candidates and p-values are identical to the Python engine:
//
//   * np.linspace: y[i] = i*step + start, last element set to `stop` exactly
//   * np.var / np.std: pairwise summation (NumPy's 8-accumulator block sum)
//   * np.median: mean of the two middle elements
//   * _emission: 0.5 * log((2*pi) * var), evaluated in that association
//   * np.argmin: first index on ties, everywhere
//
// Candidates are scored on the full series at the end of the stream.  That is
// what core.detect's streaming loop + flush produce: a proposal whose window
// has filled is tested on exactly `verify_w` points either side whenever it is
// tested, and one that has not is tested on the truncated right window at
// flush, so the p-values are the same.  (Only the online alarm list depends on
// scoring order, and core.detect discards it.)
//
// Build (done by vocd/_fastdetect.py; any C++17 compiler):
//   c++ -O2 -std=c++17 -ffp-contract=off -shared -fPIC -o _vocd_full.so _vocd_full.cpp
// (-ffp-contract=off: clang would otherwise fuse a*b+c into FMA, which
//  rounds differently from NumPy and breaks bit-identity)

#include "_gate_impl.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <vector>

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kTwoPi = 6.283185307179586;      // 2.0 * np.pi (Python float)
constexpr double kVarFloor = 1e-12;
constexpr double kClusterVarFloor = 1e-3;

// ── NumPy-exact reductions ────────────────────────────────────────────

// numpy/core/src/umath/loops_utils.h.src: pairwise_sum for contiguous doubles
double np_pairwise_sum(const double* a, int n) {
    if (n < 8) {
        double res = 0.0;
        for (int i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        int i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int n2 = n / 2;
    n2 -= n2 % 8;
    return np_pairwise_sum(a, n2) + np_pairwise_sum(a + n2, n - n2);
}

double np_var(const std::vector<double>& d) {          // ddof = 0
    const int n = static_cast<int>(d.size());
    const double mean = np_pairwise_sum(d.data(), n) / n;
    std::vector<double> sq(n);
    for (int i = 0; i < n; ++i) { const double x = d[i] - mean; sq[i] = x * x; }
    return np_pairwise_sum(sq.data(), n) / n;
}

double np_median(std::vector<double> v) {
    const std::size_t n = v.size();
    std::sort(v.begin(), v.end());
    return (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// ── core._init_regimes ────────────────────────────────────────────────

// Returns false for the branches we leave to Python (n == 0 or n < S).
bool init_regimes(const std::vector<double>& d, int S,
                  std::vector<double>& mu, std::vector<double>& var, int iters = 20) {
    const int n = static_cast<int>(d.size());
    if (n == 0 || n < S) return false;

    const double med = np_median(d);
    std::vector<double> absdev(n);
    for (int i = 0; i < n; ++i) absdev[i] = std::fabs(d[i] - med);
    double mad = np_median(absdev);
    if (mad < 1e-6) mad = std::sqrt(np_var(d)) + 1e-6;

    // np.linspace(-2*mad, 2*mad, S) + med
    std::vector<double> centers(S);
    {
        const double start = -2.0 * mad, stop = 2.0 * mad;
        if (S == 1) {
            centers[0] = med + start;
        } else {
            const double step = (stop - start) / static_cast<double>(S - 1);
            for (int i = 0; i < S; ++i) centers[i] = med + (static_cast<double>(i) * step + start);
            centers[S - 1] = med + stop;
        }
    }

    std::vector<int> z(n);
    auto assign = [&]() {
        for (int i = 0; i < n; ++i) {
            int best = 0; double bd = kInf;
            for (int s = 0; s < S; ++s) {
                const double e = (d[i] - centers[s]) * (d[i] - centers[s]);
                if (e < bd) { bd = e; best = s; }          // first index on ties
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
                centers[s] = np_median(members);
            } else {
                // farthest observation from its nearest CURRENT centre
                int far = 0; double fd = -1.0;
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
    mu = centers;
    var.assign(S, 0.0);
    const double global_var = np_var(d) + 1e-6;
    for (int s = 0; s < S; ++s) {
        members.clear();
        for (int i = 0; i < n; ++i) if (z[i] == s) members.push_back(d[i]);
        if (members.size() > 1) {
            std::vector<double> dev(members.size());
            for (std::size_t i = 0; i < members.size(); ++i) dev[i] = std::fabs(members[i] - mu[s]);
            const double cmad = np_median(dev);
            const double m = 1.4826 * cmad;
            var[s] = std::max(m * m, kClusterVarFloor);
        } else {
            var[s] = global_var;
        }
        var[s] = std::max(var[s], kClusterVarFloor);
    }
    return true;
}

// ── core._ViterbiPass ─────────────────────────────────────────────────

struct Pass {
    int S, confirm_k, min_dwell, merge_window, init_window;
    double huber_delta, pen_base;

    std::vector<double> buf, mu, var, dp, emis, new_dp;
    std::vector<double> sd, logc;      // per-regime constants: sqrt(var), 0.5*log(2*pi*var)
    std::vector<int> dwell, new_dwell;
    bool ready = false;
    int t = 0, confirmed_state = 0, last_cp = 0;
    int pending_state = -1, pending_count = 0, pending_cp = -1;
    bool init_failed = false;

    Pass(int S_, int ck, double hd, double pb, int md, int mw, int iw)
        : S(S_), confirm_k(ck), min_dwell(md), merge_window(mw), init_window(iw),
          huber_delta(hd), pen_base(pb), emis(S_), new_dp(S_), new_dwell(S_) {}

    // core._emission recomputes sqrt(var) and 0.5*log(2*pi*var) at every
    // step; the bank is fixed after init, so the values are identical each
    // time and are cached once (bit-exact: same inputs, same libm outputs).
    void cache_constants() {
        sd.resize(S); logc.resize(S);
        for (int s = 0; s < S; ++s) {
            const double v = std::max(var[s], kVarFloor);
            sd[s] = std::sqrt(v);
            logc[s] = 0.5 * std::log(kTwoPi * v);
        }
    }

    void emission(double y) {
        for (int s = 0; s < S; ++s) {
            const double r = (y - mu[s]) / sd[s];
            const double a = std::fabs(r);
            const double h = (a <= huber_delta) ? 0.5 * r * r
                                                : huber_delta * (a - 0.5 * huber_delta);
            emis[s] = h + logc[s];
        }
    }

    // returns the confirmed change-point at this step, or -1
    int step(double x) {
        emission(x);
        for (int s = 0; s < S; ++s) {
            int best_k = -1; double best_cost = kInf;
            for (int k = 0; k < S; ++k) {
                double c;
                if (k == s) {
                    c = dp[k];
                } else {
                    const bool allowed = (dwell[k] >= min_dwell) && ((emis[k] - emis[s]) > 0.0);
                    if (!allowed) continue;
                    c = dp[k] + pen_base;
                }
                if (c < best_cost) { best_cost = c; best_k = k; }     // np.argmin: first
            }
            new_dp[s] = best_cost + emis[s];
            new_dwell[s] = (best_k == s) ? dwell[s] + 1 : 1;
        }
        dp.swap(new_dp);
        dwell.swap(new_dwell);

        int best_state = 0;
        for (int s = 1; s < S; ++s) if (dp[s] < dp[best_state]) best_state = s;
        const int tt = t;
        ++t;

        int cp_out = -1;
        if (best_state != confirmed_state) {
            if (pending_state == best_state) {
                ++pending_count;
            } else {
                pending_state = best_state; pending_count = 1; pending_cp = tt;
            }
            if (pending_count >= confirm_k) {
                if (pending_cp - last_cp >= merge_window) { cp_out = pending_cp; last_cp = pending_cp; }
                confirmed_state = pending_state;
                pending_state = -1; pending_count = 0; pending_cp = -1;
            }
        } else {
            pending_state = -1; pending_count = 0; pending_cp = -1;
        }
        return cp_out;
    }

    void start(std::vector<int>& out) {
        if (!init_regimes(buf, S, mu, var)) { init_failed = true; return; }
        cache_constants();
        dp.assign(S, 0.0);
        dwell.assign(S, 1);
        ready = true;
        for (double v : buf) { const int cp = step(v); if (cp >= 0) out.push_back(cp); }
        buf.clear();
    }

    void update(double x, std::vector<int>& out) {
        if (!ready) {
            buf.push_back(x);
            if (static_cast<int>(buf.size()) < init_window) return;
            start(out);
            return;
        }
        const int cp = step(x);
        if (cp >= 0) out.push_back(cp);
    }

    void flush(std::vector<int>& out) {
        if (!ready && !buf.empty()) start(out);
    }
};

}  // namespace

extern "C" {

// Runs core.detect.  Outputs, sorted by cp:
//   out_cp[K], out_pmw[K], out_pks[K], out_code[K] where code is the gate's
//   return code (1 both p-values here; 2 p_mw needs SciPy; 0 untestable or
//   Python path for both).  Returns K, or -1 if a regime bank could not be
//   fitted here (series shorter than the bank: caller uses the Python engine).
int vocd_detect_full(const double* y, int n,
                     int S1, int ck1, double hd1, double pb1, int md1, int mw1, int iw1,
                     int S2, int ck2, double hd2, double pb2, int md2, int mw2, int iw2,
                     int verify_w, int min_side,
                     int* out_cp, double* out_pmw, double* out_pks, int* out_code,
                     int max_out) {
    Pass ppv(S1, ck1, hd1, pb1, md1, mw1, iw1);
    Pass tpr(S2, ck2, hd2, pb2, md2, mw2, iw2);
    std::vector<int> props;
    std::unordered_set<int> seen;
    std::vector<int> cands;
    auto add = [&](int loc) { if (seen.insert(loc).second) cands.push_back(loc); };

    for (int i = 0; i < n; ++i) {
        props.clear(); ppv.update(y[i], props); for (int c : props) add(c);
        props.clear(); tpr.update(y[i], props); for (int c : props) add(c);
        if (ppv.init_failed || tpr.init_failed) return -1;
    }
    props.clear(); ppv.flush(props); for (int c : props) add(c);
    props.clear(); tpr.flush(props); for (int c : props) add(c);
    if (ppv.init_failed || tpr.init_failed) return -1;

    std::sort(cands.begin(), cands.end());
    const int K = static_cast<int>(cands.size());
    for (int i = 0; i < K && i < max_out; ++i) {
        double p_mw = 0.0, p_ks = 0.0;
        const int code = gate_impl::gate(y, n, cands[i], verify_w, min_side, p_mw, p_ks);
        out_cp[i] = cands[i];
        out_code[i] = code;
        out_pmw[i] = (code == 1) ? p_mw : 0.0;
        out_pks[i] = (code >= 1) ? p_ks : 0.0;
    }
    return K;
}

}  // extern "C"
