import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def run_statistical_tests(
    proposed_results: List[float],
    baseline_results: List[float],
    metric_name: str = "metric",
    alpha: float = 0.05,
) -> Dict:

    arr_p = np.array(proposed_results, dtype=float)
    arr_b = np.array(baseline_results, dtype=float)

    n = min(len(arr_p), len(arr_b))
    arr_p, arr_b = arr_p[:n], arr_b[:n]

    # ------------------------------------------------------------------
    # Normality check (Shapiro-Wilk)
    # ------------------------------------------------------------------
    _, sw_p_val = stats.shapiro(arr_p - arr_b) if n >= 3 else (None, 0.0)
    use_parametric = sw_p_val > alpha if n >= 3 else True

    # ------------------------------------------------------------------
    # Paired t-test
    # ------------------------------------------------------------------
    t_stat, t_pval = stats.ttest_rel(arr_p, arr_b) if n >= 2 else (0.0, 1.0)

    # ------------------------------------------------------------------
    # Wilcoxon signed-rank test (non-parametric fallback)
    # ------------------------------------------------------------------
    if n >= 6:
        try:
            w_stat, w_pval = stats.wilcoxon(arr_p, arr_b)
        except ValueError:
            w_stat, w_pval = 0.0, 1.0
    else:
        w_stat, w_pval = 0.0, 1.0

    # Select primary test
    primary_pval = t_pval if use_parametric else w_pval
    significant  = primary_pval < alpha

    # ------------------------------------------------------------------
    # Effect size (Cohen's d)
    # ------------------------------------------------------------------
    diff   = arr_p - arr_b
    cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-9)


    mean_diff = float(diff.mean())
    se_diff   = float(diff.std(ddof=1) / np.sqrt(n))
    ci_lo     = mean_diff - 1.96 * se_diff
    ci_hi     = mean_diff + 1.96 * se_diff

    result = {
        'metric':          metric_name,
        'n_runs':          n,
        'mean_proposed':   float(arr_p.mean()),
        'std_proposed':    float(arr_p.std(ddof=1)),
        'mean_baseline':   float(arr_b.mean()),
        'std_baseline':    float(arr_b.std(ddof=1)),
        'mean_difference': mean_diff,
        'ci_95':           [ci_lo, ci_hi],
        't_statistic':     float(t_stat),
        't_pvalue':        float(t_pval),
        'w_statistic':     float(w_stat),
        'w_pvalue':        float(w_pval),
        'cohens_d':        float(cohens_d),
        'significant':     significant,
        'primary_test':    'paired_t' if use_parametric else 'wilcoxon',
        'primary_pvalue':  float(primary_pval),
        'conclusion': (
            f"Proposed {'significantly' if significant else 'NOT significantly'} "
            f"{'better' if mean_diff < 0 else 'worse'} than baseline "
            f"(p={primary_pval:.4f}, α={alpha})"
        ),
    }
    return result


def compare_all_baselines(
    proposed_runs: Dict[str, List[float]],
    baseline_runs: Dict[str, Dict[str, List[float]]],
    metrics: Optional[List[str]] = None,
) -> Dict:

    metrics = metrics or list(proposed_runs.keys())
    results = {}

    for baseline_name, baseline_data in baseline_runs.items():
        results[baseline_name] = {}
        for metric in metrics:
            if metric not in proposed_runs or metric not in baseline_data:
                continue
            test = run_statistical_tests(
                proposed_results=proposed_runs[metric],
                baseline_results=baseline_data[metric],
                metric_name=metric,
            )
            results[baseline_name][metric] = test
            print(
                f"  vs {baseline_name:15s} | {metric:30s} | "
                f"p={test['primary_pvalue']:.4f} | "
                f"{'✓ sig' if test['significant'] else '✗ ns':6s} | "
                f"{test['conclusion']}"
            )

    return results