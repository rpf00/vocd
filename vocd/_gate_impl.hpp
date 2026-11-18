// _gate_impl.hpp — the VOCD verification-gate arithmetic (MW + exact KS),
// shared by _gate.cpp (compiled gate for the Python engine) and
// _vocd_full.cpp (fully compiled detector).  See _gate.cpp for the
// bit-identity contract with vocd/_faststats.py and SciPy.
#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace gate_impl {


const double SQRT2 = std::sqrt(2.0);

// scipy.stats._stats_py._compute_prob_outside_square, verbatim arithmetic
inline double prob_outside_square(int n, int h) {
    double P = 0.0;
    int k = static_cast<int>(std::floor(static_cast<double>(n) / h));
    while (k >= 0) {
        double p1 = 1.0;
        for (int j = 0; j < h; ++j) {
            p1 = static_cast<double>(n - k * h - j) * p1
                 / static_cast<double>(n + k * h + j + 1);
        }
        P = p1 * (1.0 - P);
        k -= 1;
    }
    return 2 * P;
}

// scipy.stats._stats_pythran._compute_outer_prob_inside_method, verbatim
// (Hodges / Viehmann lattice recursion for 1 - P(inside), unequal sizes)
inline double outer_prob_inside(int m, int n, int g, int h) {
    if (m < n) std::swap(m, n);
    const int mg = m / g, ng = n / g;
    int minj = 0;
    int maxj = std::min(static_cast<int>(std::ceil(static_cast<double>(h) / mg)), n + 1);
    int curlen = maxj - minj;
    const int lenA = std::min(2 * maxj + 2, n + 1);
    std::vector<double> A(lenA, 1.0);
    for (int j = minj; j < maxj; ++j) A[j] = 0.0;
    for (int i = 1; i <= m; ++i) {
        const int lastminj = minj, lastlen = curlen;
        minj = std::max(static_cast<int>(std::floor(static_cast<double>(ng * i - h) / mg)) + 1, 0);
        minj = std::min(minj, n);
        maxj = std::min(static_cast<int>(std::ceil(static_cast<double>(ng * i + h) / mg)), n + 1);
        if (maxj <= minj) return 1.0;
        double val = (minj == 0) ? 0.0 : 1.0;
        for (int jj = 0; jj < maxj - minj; ++jj) {
            const int j = jj + minj;
            val = (A[jj + minj - lastminj] * i + val * j) / static_cast<double>(i + j);
            A[jj] = val;
        }
        curlen = maxj - minj;
        if (lastlen > curlen)
            for (int k = maxj - minj; k < maxj - minj + (lastlen - curlen); ++k) A[k] = 1.0;
    }
    return A[maxj - minj - 1];
}

// scipy.special.binom for small integer arguments: exact product form
inline double binom_int(int n, int k) {
    double num = 1.0, den = 1.0;
    for (int i = 1; i <= k; ++i) { num *= static_cast<double>(n - k + i); den *= static_cast<double>(i); }
    return num / den;
}

// scipy.stats._mannwhitneyu._MWU.sf: exact null distribution of U (no ties),
// via the sigma-array recurrence of configurations; n1 <= n2 by convention.
inline double mwu_exact_sf(long long k, int n1, int n2) {
    if (n1 > n2) std::swap(n1, n2);
    const long long nn = static_cast<long long>(n1) * n2;
    long long kc = nn - k;
    const bool flip = k < kc;
    if (flip) kc = k;
    const int maxu = static_cast<int>(kc);
    // sigma array, index j <-> d-multiples of (j+1)
    std::vector<long long> sfull(maxu + 1, 0);
    for (int d = 1; d <= n1; ++d) for (int i = d; i <= maxu; i += d) sfull[i] += d;
    for (int d = n2 + 1; d <= n2 + n1; ++d) for (int i = d; i <= maxu; i += d) sfull[i] -= d;
    // configurations (exact integers; magnitudes here are far below 2^53)
    std::vector<double> conf(maxu + 1, 0.0);
    conf[0] = 1.0;
    for (int u = 1; u <= maxu; ++u) {
        double dot = 0.0;                         // np.dot: float64 accumulation
        for (int i = 0; i < u; ++i) dot += conf[i] * static_cast<double>(sfull[u - i]);
        conf[u] = dot / u;
    }
    const double total = binom_int(n1 + n2, n1);
    double cdf = 0.0;                             // np.cumsum: sequential
    for (int i = 0; i <= maxu; ++i) cdf += conf[i] / total;
    const double pmf = conf[maxu] / total;
    return flip ? (1.0 - cdf + pmf) : cdf;
}

// _faststats.mannwhitney_p, asymptotic branch (both n > 8)
struct RankStats { double rank_sum_a; double tie; bool has_ties; };

inline RankStats rank_stats(const double* a, int n1, const double* b, int n2) {
    const int n = n1 + n2;
    // pooled with origin flag, sorted by value (stable: ties keep order, which
    // does not affect average ranks)
    std::vector<std::pair<double, int>> v(n);
    for (int i = 0; i < n1; ++i) v[i] = {a[i], 0};
    for (int i = 0; i < n2; ++i) v[n1 + i] = {b[i], 1};
    std::stable_sort(v.begin(), v.end(),
                     [](const auto& x, const auto& y) { return x.first < y.first; });

    double rank_sum_a = 0.0;   // exact: half-integers below 2^53
    double tie = 0.0;          // sum of c^3 - c over tie groups (exact ints)
    bool has_ties = false;
    int start = 0;
    while (start < n) {
        int end = start + 1;
        while (end < n && v[end].first == v[start].first) ++end;
        const double avg = (start + end - 1) / 2.0 + 1.0;   // 1-based mean rank
        for (int i = start; i < end; ++i)
            if (v[i].second == 0) rank_sum_a += avg;
        const double c = static_cast<double>(end - start);
        if (end - start > 1) has_ties = true;
        tie += c * c * c - c;
        start = end;
    }

    return {rank_sum_a, tie, has_ties};
}

inline double mannwhitney_p(const double* a, int n1, const double* b, int n2) {
    const int n = n1 + n2;
    const RankStats rs = rank_stats(a, n1, b, n2);
    const double u1 = rs.rank_sum_a - n1 * (n1 + 1) / 2.0;
    const double u2 = static_cast<double>(n1 * n2) - u1;
    const double u = std::max(u1, u2);
    const double mu = n1 * n2 / 2.0;
    const double var = static_cast<double>(n1 * n2) / 12.0
                       * ((n + 1) - rs.tie / (n * (n - 1.0)));
    if (var <= 0) return 1.0;
    const double z = (u - mu - 0.5) / std::sqrt(var);
    const double sf = 0.5 * std::erfc(z / SQRT2);
    return std::min(1.0, 2.0 * sf);
}

// SciPy mannwhitneyu(method='exact'): min(n1,n2) <= 8 and no ties
inline double mannwhitney_p_exact(const double* a, int n1, const double* b, int n2) {
    const RankStats rs = rank_stats(a, n1, b, n2);
    const double u1 = rs.rank_sum_a - n1 * (n1 + 1) / 2.0;
    const double u2 = static_cast<double>(n1 * n2) - u1;
    const long long u = static_cast<long long>(std::max(u1, u2));   // integer-valued
    const double p = 2.0 * mwu_exact_sf(u, n1, n2);
    return std::min(1.0, std::max(0.0, p));
}

// _faststats.ks2samp_p: exact two-sided p-value (square and unequal sizes)
inline double ks2samp_p_exact(const double* a, int n1, const double* b, int n2) {
    std::vector<double> sa(a, a + n1), sb(b, b + n2);
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    double d = 0.0;
    auto probe = [&](double x) {
        const double ca = static_cast<double>(std::upper_bound(sa.begin(), sa.end(), x) - sa.begin()) / n1;
        const double cb = static_cast<double>(std::upper_bound(sb.begin(), sb.end(), x) - sb.begin()) / n2;
        const double diff = std::fabs(ca - cb);
        if (diff > d) d = diff;
    };
    for (double x : sa) probe(x);
    for (double x : sb) probe(x);

    // _attempt_exact_2kssamp
    const int g = std::gcd(n1, n2);
    const int lcm = (n1 / g) * n2;
    const int h = static_cast<int>(std::nearbyint(d * lcm));   // np.round: half-even
    if (h == 0) return 1.0;
    double prob = (n1 == n2) ? prob_outside_square(n1, h)
                             : outer_prob_inside(n1, n2, g, h);
    if (!(prob >= 0.0 && prob <= 1.0)) {
        // SciPy would report failure and fall back to the public function;
        // signal with a negative value so the shim takes the Python path.
        return -1.0;
    }
    return std::min(1.0, std::max(0.0, prob));   // np.clip
}


// One candidate.  Returns 1 = both p-values computed (p_mw, p_ks);
//                        2 = p_ks computed, p_mw needs SciPy (tiny window with ties);
//                        0 = untestable, or exact KS failed (Python path for both).
inline int gate(const double* y, int T, int cp, int w, int min_side,
                double& p_mw, double& p_ks) {
    const int a0 = std::max(0, cp - w), a1 = cp;
    const int b0 = cp, b1 = std::min(T, cp + w);
    const int n1 = a1 - a0, n2 = b1 - b0;
    if (n1 < min_side || n2 < min_side) return 0;
    if (n1 * n2 > 10000) return 0;
    p_ks = ks2samp_p_exact(y + a0, n1, y + b0, n2);
    if (p_ks < 0.0) return 0;
    if (n1 > 8 && n2 > 8) { p_mw = mannwhitney_p(y + a0, n1, y + b0, n2); return 1; }
    const RankStats rs = rank_stats(y + a0, n1, y + b0, n2);
    if (rs.has_ties) return 2;
    p_mw = mannwhitney_p_exact(y + a0, n1, y + b0, n2);
    return 1;
}

}  // namespace gate_impl
