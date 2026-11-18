// _cpp_alarms.cpp — C++ ports of the FOCuS and NP-FOCuS alarm detectors.
// Loaded by alternative_models/_cppalarms.py (built on first use).
//
// Two implementations of each, selected by the `pruned` flag:
//
//   pruned = 0  BRUTE-FORCE PORT.  A line-by-line port of the Python
//               reference in alternative_models/{focus,npfocus}.py, which
//               recomputes the Page-CUSUM/GLR statistic by an O(k) scan over
//               every candidate at every step (O(n^2) per segment).  Same
//               arithmetic in the same order, so detections are bit-identical
//               to the Python reference.  This is the like-for-like "same
//               algorithm, same language" comparison with VOCD's C++ backend.
//
//   pruned = 1  FUNCTIONAL PRUNING (Romano, Eckley, Fearnhead & Rigaill 2023).
//               The published FOCuS algorithm: the statistic Q_t = max over
//               tau of (S_t - S_tau)^2 / (2 (t - tau)) is the maximum over a
//               set of quadratics in the change magnitude mu, and candidates
//               whose quadratic is nowhere on the upper envelope are pruned
//               permanently.  Expected O(log n) candidates per step.  Produces
//               the SAME Q and the SAME argmax tau as the brute-force scan
//               (a pruned candidate can never attain the maximum), so the
//               detections must again match the Python reference; the Python
//               driver asserts this.
//
// NP-FOCuS (Romano, Eckley & Fearnhead 2024) runs the same recursion on the
// centred Bernoulli increments 1{x <= q_k} - p_k for K quantiles q_k fixed
// from the segment burn-in, and takes the maximum over quantiles.
//
// Both detectors are driven by a port of alternative_models/_restart.py
// (restart after each alarm, re-feed only already-seen samples, min_seg
// guard), so the whole timed call is the same procedure the Python harness
// times.  The alarm threshold h is passed in from Python (it is one closed-
// form evaluation, computed inside the timed region there too).
//
// Functional pruning for ONE side (mu > 0) with cumulative sum S:
//   a candidate tau has a = S_t - S_tau, b = t - tau and quadratic
//   f(mu) = a mu - b mu^2 / 2.  Differences a_j - a_l and b_j - b_l are
//   constant over time, so pairwise dominance never changes.  Kept
//   candidates, oldest first, have strictly decreasing a and b.  At step t,
//   before pushing the new candidate tau = t-1 (a = 0 at creation):
//     A) pop the newest while its a <= 0            (dominated by the restart)
//     B) pop the newest q while, with p the one below it,
//        (a_p - a_q) / (b_p - b_q) >= a_q / b_q      (q never on the envelope)
//   then push tau = t-1.  Q+ is the max of a^2/(2b) over kept candidates with
//   a > 0; the mu < 0 side runs the same on -S; Q = max of the two.  Ties go
//   to the smallest tau, as np.argmax does.
//
// Build (any C++17 compiler; _cppalarms.py does this automatically):
//   c++ -O2 -std=c++17 -ffp-contract=off -shared -fPIC -o _cpp_alarms.so _cpp_alarms.cpp
// (-ffp-contract=off: clang would otherwise fuse a*b+c into FMA, which
//  rounds differently from NumPy and breaks bit-identity)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

// ── helpers that must match NumPy bit-for-bit ─────────────────────────

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;          // np.median: mean of middle two
}

// alternative_models._base.robust_scale
double robust_scale(const std::vector<double>& x) {
    if (x.empty()) return 1.0;
    const double med = median_of(x);
    std::vector<double> dev(x.size());
    for (size_t i = 0; i < x.size(); ++i) dev[i] = std::fabs(x[i] - med);
    const double s = 1.4826 * median_of(dev);
    if (s > 1e-9) return s;
    double m = 0.0;
    for (double v : x) m += v;
    m /= static_cast<double>(x.size());
    double var = 0.0;
    for (double v : x) var += (v - m) * (v - m);
    var /= static_cast<double>(x.size());
    const double r = std::sqrt(var) + 1e-9;
    return r != 0.0 ? r : 1.0;
}

// np.quantile(..., method="linear"), including NumPy's t >= 0.5 lerp branch
double np_quantile_linear(std::vector<double> v, double level) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    const double pos = level * static_cast<double>(n - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const double t = pos - static_cast<double>(lo);
    const double a = v[lo];
    const double b = v[std::min(lo + 1, n - 1)];
    const double diff = b - a;
    if (t >= 0.5) return b - diff * (1.0 - t);
    return a + diff * t;
}

// ── one-sided functional-pruning candidate stack ─────────────────────

struct OneSided {
    std::vector<double> S;        // signed cumulative sum, S[0] = 0
    std::vector<double> Stau;     // S at each kept candidate tau
    std::vector<int>    tau;      // kept candidates, oldest first
    long long           visits = 0;   // sum of stack sizes over steps (stats)
    long long           steps  = 0;
    size_t              peak   = 0;

    void reset() { S.assign(1, 0.0); Stau.clear(); tau.clear(); }

    // Process residual r_t (t = 1-indexed count of residuals so far + 1).
    void step(double r) {
        const int t = static_cast<int>(S.size());      // this is residual #t
        const double Sprev = S.back();                 // S_{t-1}
        if (t >= 2) {                                  // candidate tau = t-1 (tau >= 1)
            // A) restart dominance
            while (!tau.empty() && (Sprev - Stau.back()) <= 0.0) {
                tau.pop_back(); Stau.pop_back();
            }
            // B) envelope dominance
            while (tau.size() >= 2) {
                const size_t q = tau.size() - 1, p = q - 1;
                const double aq = Sprev - Stau[q], ap = Sprev - Stau[p];
                const double bq = static_cast<double>((t - 1) - tau[q]);
                const double bp = static_cast<double>((t - 1) - tau[p]);
                if ((ap - aq) * bq >= aq * (bp - bq)) { tau.pop_back(); Stau.pop_back(); }
                else break;
            }
            tau.push_back(t - 1);
            Stau.push_back(Sprev);
        }
        S.push_back(Sprev + r);
        visits += static_cast<long long>(tau.size());
        ++steps;
        if (tau.size() > peak) peak = tau.size();
    }

    // Best a^2/(2b) over kept candidates with a > 0; ties -> smallest tau.
    void best(double& val, int& arg) const {
        const int t = static_cast<int>(S.size()) - 1;
        const double St = S.back();
        val = -1.0; arg = -1;
        for (size_t i = 0; i < tau.size(); ++i) {
            const double a = St - Stau[i];
            if (a <= 0.0) continue;
            const double q = (a * a) / (2.0 * static_cast<double>(t - tau[i]));
            if (q > val) { val = q; arg = tau[i]; }
        }
    }
};

// ── FOCuS segment detector (Gaussian change in mean) ─────────────────

struct FocusSeg {
    int burnin; bool pruned;
    int seg_start = 0;
    std::vector<double> buf;
    double mu = 0.0, scale = 1.0;
    std::vector<double> csum;          // S[0]=0 .. S[k]   (brute force)
    OneSided plus, minus;              // (pruned)

    FocusSeg(int burnin_, bool pruned_) : burnin(burnin_), pruned(pruned_) {}

    void reset(int s) {
        seg_start = s; buf.clear(); csum.assign(1, 0.0);
        plus.reset(); minus.reset();
    }

    void rebuild_from_buf() {         // burn-in: fix mu/scale, rebuild sums
        mu = median_of(buf);
        scale = robust_scale(buf);
        csum.assign(1, 0.0);
        plus.reset(); minus.reset();
        for (double v : buf) {
            const double r = (v - mu) / scale;
            csum.push_back(csum.back() + r);
            if (pruned) { plus.step(r); minus.step(-r); }
        }
    }

    // returns p in {0,1}; loc = absolute index or -1
    void update(double x, double h, double& p, int& loc) {
        buf.push_back(x);
        const int k = static_cast<int>(buf.size());
        if (k <= burnin) { rebuild_from_buf(); p = 1.0; loc = -1; return; }
        const double r = (x - mu) / scale;
        double Q; int tau_best;
        if (!pruned) {
            csum.push_back(csum.back() + r);
            const double Sk = csum[k];
            Q = -1.0; tau_best = 1;
            for (int tau = 1; tau <= k - 1; ++tau) {
                const double d = Sk - csum[tau];
                const double q = (d * d) / (2.0 * static_cast<double>(k - tau));
                if (q > Q) { Q = q; tau_best = tau; }
            }
        } else {
            plus.step(r); minus.step(-r);
            double vp, vm; int tp, tm;
            plus.best(vp, tp); minus.best(vm, tm);
            Q = 0.0; tau_best = 1;                 // np.argmax of all-zero -> tau=1
            if (vp > Q || (vp == Q && tp >= 0 && tp < tau_best)) { Q = vp; tau_best = tp; }
            if (vm > Q || (vm == Q && tm >= 0 && tm < tau_best)) { Q = vm; tau_best = tm; }
        }
        p = (Q >= h) ? 0.0 : 1.0;
        loc = seg_start + tau_best;
    }
};

// ── NP-FOCuS segment detector (K quantile indicator streams) ─────────

struct NPFocusSeg {
    int burnin; bool pruned; int K;
    std::vector<double> levels;
    int seg_start = 0;
    std::vector<double> buf;
    std::vector<double> thr;                       // quantile thresholds
    std::vector<std::vector<double>> csum;         // per quantile, S[0]=0..S[k]
    std::vector<OneSided> plus, minus;             // per quantile (pruned)

    NPFocusSeg(int burnin_, bool pruned_, const double* lv, int K_)
        : burnin(burnin_), pruned(pruned_), K(K_), levels(lv, lv + K_),
          csum(K_), plus(K_), minus(K_) {}

    void reset(int s) {
        seg_start = s; buf.clear(); thr.assign(K, 0.0);
        for (int q = 0; q < K; ++q) { csum[q].assign(1, 0.0); plus[q].reset(); minus[q].reset(); }
    }

    void rebuild_from_buf() {
        for (int q = 0; q < K; ++q) {
            thr[q] = np_quantile_linear(buf, levels[q]);
            csum[q].assign(1, 0.0); plus[q].reset(); minus[q].reset();
            for (double v : buf) {
                const double inc = (v <= thr[q] ? 1.0 : 0.0) - levels[q];
                csum[q].push_back(csum[q].back() + inc);
                if (pruned) { plus[q].step(inc); minus[q].step(-inc); }
            }
        }
    }

    void update(double x, double h, double& p, int& loc) {
        buf.push_back(x);
        const int k = static_cast<int>(buf.size());
        if (k <= burnin) { rebuild_from_buf(); p = 1.0; loc = -1; return; }
        double Q = -1.0; int tau_best = 1;
        if (!pruned) {
            for (int q = 0; q < K; ++q) {
                const double inc = (x <= thr[q] ? 1.0 : 0.0) - levels[q];
                csum[q].push_back(csum[q].back() + inc);
            }
            for (int tau = 1; tau <= k - 1; ++tau) {   // max over quantiles, then first argmax
                double qmax = -1.0;
                for (int q = 0; q < K; ++q) {
                    const double d = csum[q][k] - csum[q][tau];
                    const double v = (d * d) / (2.0 * static_cast<double>(k - tau));
                    if (v > qmax) qmax = v;
                }
                if (qmax > Q) { Q = qmax; tau_best = tau; }
            }
        } else {
            Q = 0.0; tau_best = 1;
            for (int q = 0; q < K; ++q) {
                const double inc = (x <= thr[q] ? 1.0 : 0.0) - levels[q];
                plus[q].step(inc); minus[q].step(-inc);
                double vp, vm; int tp, tm;
                plus[q].best(vp, tp); minus[q].best(vm, tm);
                if (vp > Q || (vp == Q && tp >= 0 && tp < tau_best)) { Q = vp; tau_best = tp; }
                if (vm > Q || (vm == Q && tm >= 0 && tm < tau_best)) { Q = vm; tau_best = tm; }
            }
        }
        p = (Q >= h) ? 0.0 : 1.0;
        loc = seg_start + tau_best;
    }
};

// ── port of alternative_models/_restart.py ───────────────────────────

template <class Det>
int run_restart(Det& det, const double* x, int n, double h, double alpha,
                int min_seg, int* out, int max_out) {
    std::vector<int> cps;
    int seg_start = 0;
    det.reset(0);
    double p; int loc;
    for (int t = 0; t < n; ++t) {
        det.update(x[t], h, p, loc);
        const bool fired = (p < alpha) && (t - seg_start + 1 >= 2 * min_seg);
        if (fired) {
            int cp = (loc >= 0 && loc > seg_start) ? loc : t;
            cp = std::min(std::max(cp, seg_start + 1), n - 1);
            cps.push_back(cp);
            seg_start = cp;
            det.reset(seg_start);
            for (int s = cp + 1; s <= t; ++s) det.update(x[s], h, p, loc);
        }
    }
    std::sort(cps.begin(), cps.end());
    cps.erase(std::unique(cps.begin(), cps.end()), cps.end());
    const int m = static_cast<int>(std::min<size_t>(cps.size(), max_out));
    for (int i = 0; i < m; ++i) out[i] = cps[i];
    return static_cast<int>(cps.size());
}

}  // namespace

extern "C" {

// stats (may be null): [0] = mean kept candidates per step (all stacks),
//                       [1] = peak kept candidates in any one stack.
int focus_detect(const double* x, int n, double h, double alpha, int min_seg,
                 int burnin, int pruned, int* out, int max_out, double* stats) {
    FocusSeg det(burnin, pruned != 0);
    const int m = run_restart(det, x, n, h, alpha, min_seg, out, max_out);
    if (stats) {
        const double steps = static_cast<double>(std::max<long long>(1, det.plus.steps));
        stats[0] = static_cast<double>(det.plus.visits + det.minus.visits) / steps;
        stats[1] = static_cast<double>(std::max(det.plus.peak, det.minus.peak));
    }
    return m;
}

int npfocus_detect(const double* x, int n, double h, double alpha, int min_seg,
                   int burnin, const double* levels, int K, int pruned,
                   int* out, int max_out, double* stats) {
    NPFocusSeg det(burnin, pruned != 0, levels, K);
    const int m = run_restart(det, x, n, h, alpha, min_seg, out, max_out);
    if (stats) {
        long long visits = 0, steps_ll = 1; size_t peak = 0;
        for (int q = 0; q < K; ++q) {
            visits += det.plus[q].visits + det.minus[q].visits;
            steps_ll = std::max(steps_ll, det.plus[q].steps);
            peak = std::max({peak, det.plus[q].peak, det.minus[q].peak});
        }
        const double steps = static_cast<double>(steps_ll);
        stats[0] = static_cast<double>(visits) / steps;
        stats[1] = static_cast<double>(peak);
    }
    return m;
}

}  // extern "C"
