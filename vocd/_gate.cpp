// _gate.cpp — compiled VOCD verification gate (Mann-Whitney + Kolmogorov-Smirnov).
//
// Computes exactly what vocd/_faststats.py computes, in the same floating-
// point order, so p-values are bit-identical and detections do not change:
//
//   Mann-Whitney U, two-sided: average ranks with ties, tie-corrected normal
//   approximation with continuity correction, upper tail via erfc.  This is
//   SciPy's `method='asymptotic'` branch, which `_faststats` uses when both
//   samples have > 8 observations (SciPy's own `auto` rule).
//
//   Kolmogorov-Smirnov two-sample, two-sided, EXACT p-value for n1 == n2:
//   SciPy's `_compute_prob_outside_square` (Horner form of the alternating
//   binomial series), reached through `_attempt_exact_2kssamp`.  Rank sums
//   and lattice counts are exact integers / half-integers, so the only
//   rounding happens in the final few scalar operations, which are replicated
//   verbatim.
//
// Scope.  The Python shim (vocd/_fastgate.py) sends a candidate here only when
// both windows have >= 9 points AND the windows are equal in size.  Inside a
// series that is every candidate (both windows are `verify_w` wide); the
// exceptions are candidates within `verify_w` of the end of the series, tested
// on the final flush with a truncated right window.  Those go through the
// unchanged `_faststats` path (SciPy's exact small-sample MW and its compiled
// unequal-size KS), so the gate as a whole is exact by construction rather
// than by approximation.
//
// Build (done automatically by vocd/_fastgate.py; any C++17 compiler):
//   c++ -O2 -std=c++17 -ffp-contract=off -shared -fPIC -o _gate.so _gate.cpp
// (-ffp-contract=off: clang would otherwise fuse a*b+c into FMA, which
//  rounds differently from NumPy and breaks bit-identity)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "_gate_impl.hpp"

extern "C" {

// Returns 1 if both p-values were computed here (out[0]=p_mw, out[1]=p_ks);
//         2 if only p_ks was (min(n1,n2) <= 8 WITH ties: SciPy's asymptotic
//           MW there uses cephes' ndtr, which is not bit-reproducible with
//           libm, so the shim computes p_mw through SciPy's own routine);
//         0 if untestable or the exact KS failed (shim takes the Python path).
int vocd_gate(const double* y, int T, int cp, int w, int min_side, double* out) {
    double p_mw = 0.0, p_ks = 0.0;
    const int code = gate_impl::gate(y, T, cp, w, min_side, p_mw, p_ks);
    if (code >= 1) out[1] = p_ks;
    if (code == 1) out[0] = p_mw;
    return code;
}

}  // extern "C"
