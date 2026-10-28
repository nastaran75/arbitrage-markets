"""
Conventions. A market or benchmark has P problems and M systems (models). Arrays are indexed in this order:
    solve_prob[problem, model, budget]   probability that a model solves a problem when given that budget on it
    expected_spend[problem, budget]      money actually spent on that budget, since attempts stop at the first success
    successes[problem, model], attempts[problem, model], attempt_price[problem, model]
An allocation is one budget per model, applied to every problem.
"""

import os, re, json, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    MultipleLocator,
    NullLocator,
    PercentFormatter,
)
from scipy.special import gammaln, betaln, logsumexp, digamma
from scipy.optimize import minimize

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# ---------------------------------------------------------------------------------------------------------------
# Look of the figures. They are drawn 5.5 inches wide, which is what these point sizes are chosen for.
# ---------------------------------------------------------------------------------------------------------------
FS = {
    "title": 9.0,
    "label": 8.5,
    "tick": 7.0,
    "legend": 7.0,
    "annot": 8.0,
    "dense": 6.5,
}
INK, GRID, AXIS = "#0b0b0b", "#e1e0d9", "#8f8e89"  # text, grid lines, axis lines
MODEL_COLOR, MODEL_ALPHA, MODEL_LW = "#2a78d6", 0.38, 1.0  # single models
ARB_COLOR, ARB_ALPHA, ARB_LW = "#eb6834", 0.65, 2.0  # the arbitrageur
LINE_ALPHA = 0.80


def dollar_label(value, _pos=None):
    """Tick labels for cost figures"""
    if value >= 1000:
        return f"${value / 1000:g}k"
    if value >= 1:
        return f"${value:,.0f}"
    if value >= 1e-3:
        return f"${value:g}"
    return rf"$\${{10}}^{{{int(round(np.log10(value)))}}}$"


def style_axes(ax, ylabel=None, cost_on_x=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, ls="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.5, labelsize=FS["tick"], colors=AXIS, labelcolor=INK)
    if cost_on_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(dollar_label))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS["label"])


# ---------------------------------------------------------------------------------------------------------------
# Data: one folder per benchmark, one JSONL per system, one row per problem {id, attempts: [0/1, ...], mean_cost}.
# ---------------------------------------------------------------------------------------------------------------
def load_results(base_dir):
    """{system: [rows]} for every JSONL in the folder."""
    results = {}
    for file in sorted(os.listdir(base_dir)):
        if file.endswith(".jsonl"):
            results[file[: -len(".jsonl")]] = [json.loads(line) for line in open(os.path.join(base_dir, file))]
    return results


def discover_models(base_dir):
    """The systems in a folder, by file name."""
    return sorted(f[: -len(".jsonl")] for f in os.listdir(base_dir) if f.endswith(".jsonl"))


def shared_problems(results, models):
    """Ids of the problems every listed system has rows for, sorted."""
    return sorted(set.intersection(*[{row["id"] for row in results[m]} for m in models]))


def attempt_prices(results, models):
    """(attempt_price[problem, model], problem ids) on the shared problems."""
    by_id = {m: {row["id"]: row for row in results[m]} for m in models}
    ids = shared_problems(results, models)
    attempt_price = np.array([[by_id[m][pid]["mean_cost"] for m in models] for pid in ids], dtype=float)
    return attempt_price, ids


def counts(results, models, ids):
    """(successes[problem, model], attempts[problem, model]) on the given problems."""
    by_id = {m: {row["id"]: row for row in results[m]} for m in models}
    successes = np.zeros((len(ids), len(models)))
    attempts = np.zeros((len(ids), len(models)))
    for j, m in enumerate(models):
        for i, pid in enumerate(ids):
            outcomes = by_id[m][pid]["attempts"]
            successes[i, j], attempts[i, j] = int(sum(outcomes)), len(outcomes)
    return successes, attempts


# ---------------------------------------------------------------------------------------------------------------
# pass@k estimators. Each takes one model's successes[problem] and attempts[problem] and the attempts k to
# evaluate at, and returns passk[problem, k]. k is a vector shared by all problems or a matrix with one row per
# problem; see attempts_per_problem.
# ---------------------------------------------------------------------------------------------------------------
LOG_CONCENTRATION_MAX = np.log(1e6)  # Beta(a, b) with a + b ~ 1e6 is already a point mass for any k of interest
LOG_CONCENTRATION_MIN = np.log(1e-4)
HYPERPRIOR_POW = 2.5  # p(a, b) ∝ (a + b)^(-HYPERPRIOR_POW), Gelman et al. (BDA3 §5.3). With mostly {0, 1} counts the
# likelihood is flat along a/(a+b) = const; the hyperprior breaks the tie toward a small concentration instead of
# the point-mass end, where every problem gets the same curve. 0 is the plain maximum-likelihood fit.
SPIKE_OFF_LOGIT = -40.0  # the plain Beta-Binomial fit shares the zero-inflated objective with the spike pinned at ~0


def attempts_per_problem(k, num_problems):
    """The attempts k at which to evaluate pass@k, as one row per problem."""
    k = np.asarray(k, dtype=float)
    return np.broadcast_to(k[None, :], (num_problems, k.size)) if k.ndim == 1 else k


def log_beta_binomial(successes, attempts, a, b):
    return (
        gammaln(attempts + 1)
        - gammaln(successes + 1)
        - gammaln(attempts - successes + 1)
        + gammaln(successes + a)
        + gammaln(attempts - successes + b)
        - gammaln(attempts + a + b)
        - (gammaln(a) + gammaln(b) - gammaln(a + b))
    )


def zero_inflated_objective(theta, successes, attempts, hyperprior_pow):
    """Negative log-posterior of the zero-inflated Beta-Binomial and its gradient, in theta = (log a, log b, logit
    of the spike weight)."""
    a, b = np.exp(theta[0]), np.exp(theta[1])
    spike = 1.0 / (1.0 + np.exp(-theta[2]))
    log_slab = np.log1p(-spike) + log_beta_binomial(successes, attempts, a, b)
    log_spike = np.where(successes == 0, np.log(spike), -np.inf)
    log_total = logsumexp(np.stack([log_spike, log_slab]), axis=0)
    w_slab = np.exp(log_slab - log_total)  # posterior weight of the slab, per problem
    w_spike = np.where(successes == 0, np.exp(log_spike - log_total), 0.0)
    d_a = digamma(successes + a) - digamma(attempts + a + b) - digamma(a) + digamma(a + b)
    d_b = digamma(attempts - successes + b) - digamma(attempts + a + b) - digamma(b) + digamma(a + b)
    grad = np.array(
        [
            -np.sum(w_slab * d_a) * a,
            -np.sum(w_slab * d_b) * b,
            -np.sum(w_spike * (1.0 - spike) - w_slab * spike),
        ]
    )
    nll = -np.sum(log_total)
    if hyperprior_pow:
        nll += hyperprior_pow * np.log(a + b)
        grad[0] += hyperprior_pow * a / (a + b)
        grad[1] += hyperprior_pow * b / (a + b)
    return nll, grad


OPTIONS = {"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8}


def fit_zero_inflated_beta(successes, attempts, hyperprior_pow=HYPERPRIOR_POW):
    """Our prior: type-II MAP of the zero-inflated Beta-Binomial, (a, b, spike weight), by bounded L-BFGS-B from
    three starts."""
    bounds = [(LOG_CONCENTRATION_MIN, LOG_CONCENTRATION_MAX)] * 2 + [(-12.0, 8.0)]
    best = None
    for start in ([-1, -1, -1], [0, 0, 0], [-2, 0, -0.5]):
        r = minimize(
            zero_inflated_objective,
            start,
            args=(successes, attempts, hyperprior_pow),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=OPTIONS,
        )
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x[0]), np.exp(best.x[1]), 1.0 / (1.0 + np.exp(-best.x[2]))


def fit_beta(successes, attempts, hyperprior_pow=HYPERPRIOR_POW):
    """Type-II MAP of the plain Beta-Binomial, (a, b): the zero-inflated fit with the spike pinned at zero."""

    def objective(theta2):
        nll, grad = zero_inflated_objective(
            np.array([theta2[0], theta2[1], SPIKE_OFF_LOGIT]),
            successes,
            attempts,
            hyperprior_pow,
        )
        return nll, grad[:2]

    bounds = [(LOG_CONCENTRATION_MIN, LOG_CONCENTRATION_MAX)] * 2
    best = None
    for start in ([-1, -1], [0, 0], [-2, 0]):
        r = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=OPTIONS,
        )
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x[0]), np.exp(best.x[1])


def plugin_passk(successes, attempts, k):
    """The plug-in: 1 - (1 - c/n)^k at any k, with c/n capped just below one."""
    rate = np.minimum(successes / attempts, 1.0 - 1.0 / (2.0 * attempts))
    return 1.0 - np.exp(np.log1p(-rate)[:, None] * attempts_per_problem(k, successes.size))


def chen_passk(successes, attempts, k):
    """Unbiased pass@k of Chen et al. (2021), 1 - C(n-c, k) / C(n, k) at integer k, frozen past k = n. Between
    integers the failure probability is log-linear (constant hazard within an attempt); the attempt that makes
    success certain is interpolated linearly."""
    num_problems = successes.size
    n_max = int(attempts.max())
    j = np.arange(n_max)[None, :]  # the j-th attempt
    numer = attempts[:, None] - successes[:, None] - j
    denom = attempts[:, None] - j
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(
            (numer > 0) & (denom > 0),
            np.log(numer / np.where(denom > 0, denom, 1)),
            -np.inf,
        )
    step = np.where(j < attempts[:, None], step, 0.0)
    log_fail = np.concatenate([np.zeros((num_problems, 1)), np.cumsum(step, axis=1)], axis=1)  # at k = 0..n_max
    fail = np.exp(log_fail)
    kk = np.minimum(attempts_per_problem(k, num_problems), attempts[:, None])
    k_floor = np.floor(kk).astype(int)
    k_ceil = np.minimum(k_floor + 1, n_max)
    fail_floor, fail_ceil = np.take_along_axis(fail, k_floor, axis=1), np.take_along_axis(fail, k_ceil, axis=1)
    frac = kk - k_floor
    linear = fail_floor + frac * (fail_ceil - fail_floor)
    log_floor = np.take_along_axis(log_fail, k_floor, axis=1)
    log_ceil = np.where(fail_ceil > 0, np.take_along_axis(log_fail, k_ceil, axis=1), 0.0)
    loglinear = np.exp((1.0 - frac) * log_floor + frac * log_ceil)
    return 1.0 - np.where(fail_ceil > 0, loglinear, linear)


def zero_inflated_beta_passk(successes, attempts, k, a, b, spike):
    """Ours: each problem's posterior predictive under the fitted zero-inflated Beta prior."""
    kk = attempts_per_problem(k, successes.size)
    successes, attempts = successes[:, None], attempts[:, None]
    fail = np.exp(
        betaln(a + successes, b + attempts - successes + kk) - betaln(a + successes, b + attempts - successes)
    )
    passk = 1.0 - fail
    unsolved = successes[:, 0] == 0
    if np.any(unsolved):
        p_zero_under_slab = np.exp(betaln(a, b + attempts[unsolved, 0]) - betaln(a, b))
        spike_weight = spike / (spike + (1.0 - spike) * p_zero_under_slab)  # posterior weight of the spike
        passk[unsolved] = (1.0 - spike_weight)[:, None] * passk[unsolved]
    return np.clip(passk, 0.0, 1.0)


def chen_then_beta_passk(successes, attempts, k, a, b):
    """Chen's estimator within the recorded attempts, continued past them by the Beta posterior predictive."""
    kk = attempts_per_problem(k, successes.size)
    within = chen_passk(successes, attempts, kk)
    extra = np.maximum(kk - attempts[:, None], 0.0)
    fail_extra = np.exp(
        betaln(a + successes[:, None], b + attempts[:, None] - successes[:, None] + extra)
        - betaln(a + successes[:, None], b + attempts[:, None] - successes[:, None])
    )
    unsolved = (successes == 0)[:, None].astype(float)  # Chen's failure at k = n is one iff nothing was solved
    return np.where(kk <= attempts[:, None], within, 1.0 - unsolved * fail_extra)


def market_passk(results, models, ids, attempt_price, budget, estimator):
    """solve_prob[problem, model, budget] on a market: 'chen' (frozen at the recorded attempts) or 'chen_bb'
    (continued past them by a Beta prior fit per system without the hyperprior)."""
    successes, attempts = counts(results, models, ids)
    solve_prob = np.empty((len(ids), len(models), budget.size))
    for j in range(len(models)):
        k = budget[None, :] / attempt_price[:, j, None]  # attempts affordable on each problem
        if estimator == "chen":
            solve_prob[:, j, :] = chen_passk(successes[:, j], attempts[:, j], k)
        elif estimator == "chen_bb":
            a, b = fit_beta(successes[:, j], attempts[:, j], hyperprior_pow=0.0)
            solve_prob[:, j, :] = chen_then_beta_passk(successes[:, j], attempts[:, j], k, a, b)
        else:
            raise ValueError(estimator)
    return solve_prob


# ---------------------------------------------------------------------------------------------------------------
# Spend and the arbitrage solver, on tabulated curves solve_prob[problem, model, budget].
# ---------------------------------------------------------------------------------------------------------------
def interpolate_at_budget(curves, budget, x):
    """Read every problem's curve at one budget x that need not lie on the grid."""
    if x <= budget[0]:
        return curves[:, 0] * (x / budget[0] if budget[0] > 0 else 0.0)
    if x >= budget[-1]:
        return curves[:, -1]
    hi = int(np.searchsorted(budget, x))
    lo = hi - 1
    frac = (x - budget[lo]) / (budget[hi] - budget[lo])
    return curves[:, lo] + frac * (curves[:, hi] - curves[:, lo])


def expected_spend_curve(solve_prob, budget, spend_cap=None):
    """Money actually spent when committing each budget to one model on each problem, int_0^b (1 - U) dt: attempts
    stop at the first success. `spend_cap[problem]`: no spend past that budget (frozen curves).
    """
    fail = np.concatenate([np.ones((solve_prob.shape[0], 1)), 1.0 - solve_prob], axis=1)
    widths = np.diff(np.concatenate([[0.0], budget]))
    spend = np.cumsum(0.5 * (fail[:, 1:] + fail[:, :-1]) * widths[None, :], axis=1)
    if spend_cap is None:
        return spend
    capped_budget = np.minimum(budget[None, :], np.asarray(spend_cap, dtype=float)[:, None])
    return np.stack([np.interp(capped_budget[p], budget, spend[p]) for p in range(solve_prob.shape[0])])


def expected_spend_curves(solve_prob, budget, spend_cap=None):
    """[expected_spend_curve of each model]; `spend_cap[problem, model]` caps each cell."""
    return [
        expected_spend_curve(solve_prob[:, m, :], budget, None if spend_cap is None else spend_cap[:, m])
        for m in range(solve_prob.shape[1])
    ]


def single_models(solve_prob, budget, spend_cap=None):
    """Each model on its own: (solve rate[model, budget], expected spend[model, budget]), averaged over problems."""
    return solve_prob.mean(0), np.array([s.mean(0) for s in expected_spend_curves(solve_prob, budget, spend_cap)])


def pooled_solve_rate(solve_prob, budget, alloc):
    """Solve rate of an allocation: a problem is solved if any model solves it."""
    fail = np.stack(
        [1.0 - interpolate_at_budget(solve_prob[:, m, :], budget, alloc[m]) for m in range(solve_prob.shape[1])],
        1,
    )
    return float(np.mean(1.0 - np.prod(fail, axis=1)))


def cell_outcomes(solve_prob, expected_spend, budget, alloc):
    """Under an allocation, per problem and model: (solve probability, expected spend if the model is reached)."""
    num_models = solve_prob.shape[1]
    solve = np.stack(
        [interpolate_at_budget(solve_prob[:, m, :], budget, alloc[m]) for m in range(num_models)],
        1,
    )
    spend = np.stack(
        [interpolate_at_budget(expected_spend[m], budget, alloc[m]) for m in range(num_models)],
        1,
    )
    return solve, spend


def spend_in_order(solve_prob, expected_spend, budget, alloc, order):
    """Expected spend of an allocation deployed in a fixed model order: a model is paid only if every model
    before it failed."""
    solve, spend = cell_outcomes(solve_prob, expected_spend, budget, alloc)
    order = np.asarray(order)
    log_fail = np.log(np.clip(1.0 - solve, 1e-12, 1.0))[:, order]
    prob_reached = np.exp(np.cumsum(log_fail, axis=1) - log_fail)
    return float((prob_reached * spend[:, order]).sum(1).mean())


def greedy_order(solve_prob, expected_spend, budget, alloc):
    """Greedy deployment order: next the funded model with the most expected successes per expected dollar on the
    problems still unsolved. Unfunded models follow in index order."""
    solve, spend = cell_outcomes(solve_prob, expected_spend, budget, alloc)
    num_models = solve_prob.shape[1]
    funded = [m for m in range(num_models) if alloc[m] > 0]
    unsolved = np.ones(solve_prob.shape[0])  # probability each problem is still unsolved
    order = []
    while funded:
        best, best_value = None, -np.inf
        for m in funded:
            cost = float((unsolved * spend[:, m]).mean())
            value = float((unsolved * solve[:, m]).mean()) / cost if cost > 0 else -np.inf
            if value > best_value:
                best, best_value = m, value
        if best is None:
            order += funded
            break
        order.append(best)
        funded.remove(best)
        unsolved = unsolved * (1.0 - solve[:, best])
    return np.array(order + [m for m in range(num_models) if alloc[m] <= 0])


def greedy_spend(solve_prob, expected_spend, budget, alloc):
    """Expected spend of an allocation under its greedy order."""
    return spend_in_order(
        solve_prob,
        expected_spend,
        budget,
        alloc,
        greedy_order(solve_prob, expected_spend, budget, alloc),
    )


def coordinate_ascent(
    solve_prob,
    slope,
    budget,
    shadow_price,
    alloc0,
    sweeps=40,
    budget_cap=None,
    exact=False,
):
    """Cyclic coordinate ascent of the pooled solve rate minus shadow_price * total budget, one model at a time.
    The default step assumes a decreasing marginal return (the plug-in's curves) and interpolates where it equals
    the shadow price; `exact=True` takes the argmax over the budget grid, needed for the non-concave Chen curves.
    """
    num_problems, num_models = solve_prob.shape[:2]
    alloc = alloc0.copy()
    log_fail = np.log(
        np.clip(
            1.0
            - np.stack(
                [interpolate_at_budget(solve_prob[:, m, :], budget, alloc[m]) for m in range(num_models)],
                1,
            ),
            1e-12,
            1.0,
        )
    )
    for _ in range(sweeps):
        largest_move = 0.0
        total = log_fail.sum(1)
        for m in range(num_models):
            others_fail = np.exp(total - log_fail[:, m])
            if exact:
                gain = (others_fail @ solve_prob[:, m, :]) / num_problems - shadow_price * budget
                if budget_cap is not None:
                    gain = np.where(budget <= budget_cap[m], gain, -np.inf)
                b = int(np.argmax(gain))
                x = float(budget[b]) if gain[b] > 0.0 else 0.0
            else:
                marginal = (others_fail[:, None] * slope[:, m, :]).mean(0)
                if marginal[0] <= shadow_price:
                    x = 0.0
                elif marginal[-1] >= shadow_price:
                    x = float(budget[-1])
                else:
                    hi = int(np.searchsorted(-marginal, -shadow_price))
                    lo = hi - 1
                    drop = marginal[lo] - marginal[hi]
                    frac = (marginal[lo] - shadow_price) / drop if drop != 0 else 0.0
                    x = float(budget[lo] + frac * (budget[hi] - budget[lo]))
                if budget_cap is not None:
                    x = min(x, float(budget_cap[m]))
            largest_move = max(largest_move, abs(x - alloc[m]))
            alloc[m] = x
            new_log_fail = np.log(
                np.clip(
                    1.0 - interpolate_at_budget(solve_prob[:, m, :], budget, x),
                    1e-12,
                    1.0,
                )
            )
            total += new_log_fail - log_fail[:, m]
            log_fail[:, m] = new_log_fail
        if largest_move < 1e-10:
            break
    return alloc


def shadow_price_grid(solve_prob, budget, num=70, num_head=0, exact=False):
    """Shadow prices, high to low, from the largest mean marginal utility per dollar; `num_head` extra points just
    below it make the frontier continuous down to zero budget."""
    slope = np.gradient(solve_prob, budget, axis=2)
    num_models = solve_prob.shape[1]
    highest = float(max(slope[:, m, 0].mean() for m in range(num_models)))
    if exact:
        highest = max(highest, float(max(slope[:, m, :].mean(0).max() for m in range(num_models))))
    head = highest * (1.0 - np.logspace(-4, np.log10(0.8), num_head)) if num_head > 0 else np.empty(0)
    return np.concatenate(
        [
            head,
            np.logspace(
                np.log10(highest * (0.2 if num_head > 0 else 1.0)),
                np.log10(highest * 1e-6),
                num,
            ),
        ]
    )


def arbitrage_frontier(
    solve_prob,
    budget,
    num=70,
    budget_cap=None,
    exact=False,
    num_head=0,
    shadow_prices=None,
    warm_starts=None,
    sweeps=40,
    expected_spend=None,
):
    """The arbitrage frontier: for each shadow price, the allocation maximizing solve rate - price * spend.
    Returns (solve rates, expected spends under the greedy order, allocations), one per shadow price.
    `budget_cap[model]` bounds each model's budget; `expected_spend` is the spend table to price with (uncapped by
    default); `warm_starts[price]` starts each price from a given allocation instead of the previous price's.
    """
    slope = np.gradient(solve_prob, budget, axis=2)
    num_models = solve_prob.shape[1]
    if expected_spend is None:
        expected_spend = expected_spend_curves(solve_prob, budget)
    if shadow_prices is None:
        shadow_prices = shadow_price_grid(solve_prob, budget, num=num, num_head=num_head, exact=exact)
    solve, spend, allocs = (
        np.empty(shadow_prices.size),
        np.empty(shadow_prices.size),
        np.empty((shadow_prices.size, num_models)),
    )
    alloc = np.zeros(num_models)
    for k, shadow_price in enumerate(shadow_prices):
        alloc0 = alloc if warm_starts is None else np.asarray(warm_starts[k], dtype=float)
        alloc = coordinate_ascent(
            solve_prob,
            slope,
            budget,
            shadow_price,
            alloc0,
            budget_cap=budget_cap,
            exact=exact,
            sweeps=sweeps,
        )
        solve[k], spend[k], allocs[k] = (
            pooled_solve_rate(solve_prob, budget, alloc),
            greedy_spend(solve_prob, expected_spend, budget, alloc),
            alloc,
        )
    return solve, spend, allocs


def cost_at_solve_rate(solve, cost, targets):
    """Cost at which a (solve rate, cost) curve reaches each target; NaN outside its range."""
    order = np.argsort(solve)
    solve, cost = solve[order], cost[order]
    out = np.interp(targets, solve, cost, left=np.nan, right=np.nan)
    out[targets > solve[-1] + 1e-12] = np.nan
    out[targets < solve[0] - 1e-12] = np.nan
    return out


# ---------------------------------------------------------------------------------------------------------------
# The benchmarks.
# ---------------------------------------------------------------------------------------------------------------
def _benchmark(folder, title, xlabel, budget_min, budget_max, models=None):
    return dict(
        base_dir=os.path.join(DATA, folder),
        models=models,
        title=title,
        xlabel=xlabel,
        bmin=budget_min,
        bmax=budget_max,
    )


DATASETS = {
    # markets: dollar-priced systems
    "livecodebench_priced": _benchmark(
        "livecodebench_priced",
        "LiveCodeBench (24 systems)",
        "LiveCodeBench solve rate",
        1e-7,
        2.0,
    ),
    "terminal_bench2_priced": _benchmark(
        "terminal_bench2_priced",
        "Terminal-Bench 2.0 (28 systems)",
        "Terminal-Bench solve rate",
        3e-4,
        30.0,
    ),
    "terminal_bench4_priced": _benchmark(
        "terminal_bench4_priced",
        "Terminal-Bench 4.0",
        "Terminal-Bench 4.0 solve rate",
        1e-2,
        500.0,
    ),
    "deepswe_priced": _benchmark("deepswe_priced", "DeepSWE (70 systems)", "DeepSWE solve rate", 1e-2, 100.0),
    # estimation benchmarks: cost is a compute proxy (FLOPs on Monkey Business, parameters x tokens on IRSL)
    "monkey_math": _benchmark(
        "monkey_math",
        "MATH (Monkey Business)",
        "MATH solve rate",
        30.0,
        5e7,
        models=[
            "Pythia-1B",
            "Pythia-1.4B",
            "Pythia-2.8B",
            "Pythia-6.9B",
            "Pythia-12B",
            "Gemma-2B",
            "Gemma-7B",
            "Llama-3-8B",
            "Llama-3-8B-Instruct",
            "Llama-3-70B-Instruct",
        ],
    ),
    "monkey_codecontests": _benchmark(
        "monkey_codecontests",
        "CodeContests (Monkey Business)",
        "CodeContests solve rate",
        500.0,
        5e8,
        models=[
            "Gemma-2B",
            "Gemma-7B",
            "Llama-3-8B",
            "Llama-3-8B-Instruct",
            "Llama-3-70B-Instruct",
        ],
    ),
    "irsl_aime2024": _benchmark("irsl_aime2024", "AIME 2024 (IRSL)", "AIME 2024 solve rate", 1e3, 1e8),
    "irsl_aime2025": _benchmark("irsl_aime2025", "AIME 2025 (IRSL)", "AIME 2025 solve rate", 1e3, 1e8),
    "irsl_gmmlu": _benchmark("irsl_gmmlu", "Global-MMLU-Lite (IRSL)", "Global-MMLU-Lite solve rate", 1e3, 1e8),
    "irsl_mmlupro": _benchmark("irsl_mmlupro", "MMLU-Pro (IRSL)", "MMLU-Pro solve rate", 1e3, 1e8),
}


def display_title(benchmark):
    """The title without the system count."""
    return re.sub(r"\s*\(\d+ systems\)", "", benchmark["title"])


# ---------------------------------------------------------------------------------------------------------------
# Markets (Figures 1 to 4): curves on a budget grid, the figure record, cross-validation helpers.
# ---------------------------------------------------------------------------------------------------------------
POINTS_PER_DECADE = 50
NUM_SHADOW_PRICES, NUM_HEAD_PRICES = 70, 16


def _market_tables(benchmark, estimator, extra_decades=4):
    """(models, ids, attempt_price, solve_prob, budget, spend_cap) on a budget grid extended `extra_decades` below
    bmin, so every model is defined down to ~0% solve rate. The frozen 'chen' curves spend nothing past the
    recorded attempts, which is what spend_cap[problem, model] says."""
    models = benchmark["models"] or discover_models(benchmark["base_dir"])
    results = load_results(benchmark["base_dir"])
    attempt_price, ids = attempt_prices(results, models)
    lo, hi = np.log10(benchmark["bmin"]) - extra_decades, np.log10(benchmark["bmax"])
    budget = np.logspace(lo, hi, int((hi - lo) * POINTS_PER_DECADE))
    solve_prob = market_passk(results, models, ids, attempt_price, budget, estimator)
    _, attempts = counts(results, models, ids)
    spend_cap = attempts * attempt_price if estimator == "chen" else None
    return models, ids, attempt_price, solve_prob, budget, spend_cap


def market_curves(benchmark, estimator="chen"):
    """(models, ids, solve_prob[problem, model, budget], expected_spend[model, problem, budget], budget grid)."""
    models, ids, _, solve_prob, budget, spend_cap = _market_tables(benchmark, estimator)
    return (
        models,
        ids,
        solve_prob,
        np.stack(expected_spend_curves(solve_prob, budget, spend_cap)),
        budget,
    )


def in_sample_costs(benchmark, estimator="chen", num=70):
    """In-sample cost curves of a market on a solve-rate grid: (grid, model_cost[model, grid], arbitrageur cost,
    best single model cost, floor), costs summed over the benchmark."""
    models, ids, _, solve_prob, budget, spend_cap = _market_tables(benchmark, estimator)
    num_problems = len(ids)
    model_solve, model_spend = single_models(solve_prob, budget, spend_cap)
    arb_solve, arb_spend, _ = arbitrage_frontier(
        solve_prob,
        budget,
        num=num,
        exact=True,
        num_head=NUM_HEAD_PRICES,
        expected_spend=expected_spend_curves(solve_prob, budget, spend_cap),
    )
    funded = arb_spend > 0
    arb_solve, arb_spend = arb_solve[funded], arb_spend[funded]
    grid = np.linspace(max(0.0, min(model_solve.min(), arb_solve.min())), float(arb_solve.max()), 300)
    model_cost = np.array([cost_at_solve_rate(model_solve[j], model_spend[j], grid) for j in range(len(models))])
    with np.errstate(all="ignore"):
        market_cost = np.nanmin(model_cost, axis=0)
    arb_cost = np.fmin(
        cost_at_solve_rate(arb_solve, arb_spend, grid), market_cost
    )  # a single model is a feasible allocation
    return (
        grid,
        model_cost * num_problems,
        arb_cost * num_problems,
        market_cost * num_problems,
        benchmark["bmin"] * num_problems,
    )


def market_record(benchmark, costs):
    """The figure record of a market, from in_sample_costs."""
    grid, model_cost, arb_cost, market_cost, floor = costs
    title = display_title(benchmark)
    return dict(
        title=title,
        legend=re.sub(r"\s+\d+(\.\d+)?$", "", title),
        xlabel=benchmark["xlabel"],
        floor=float(floor),
        solve_rate=grid * 100.0,
        model_cost=model_cost,
        arb_cost=arb_cost,
        market_cost=market_cost,
    )


def folds(num_problems, num_folds, seed=0):
    """Test folds over the problems (num_folds = 0: leave-one-out)."""
    perm = np.random.default_rng(seed).permutation(num_problems)
    num_folds = num_problems if num_folds == 0 else num_folds
    return [np.sort(perm[i::num_folds]) for i in range(num_folds)]


def fit_frontier(solve_prob, expected_spend, budget, shadow_prices, warm_starts=None, sweeps=40):
    """Frontier on the given problems: (solve rates, spends, allocations, greedy orders), one per shadow price."""
    solve, spend, allocs = arbitrage_frontier(
        solve_prob,
        budget,
        exact=True,
        shadow_prices=shadow_prices,
        warm_starts=warm_starts,
        sweeps=sweeps,
        expected_spend=list(expected_spend),
    )
    return (
        solve,
        spend,
        allocs,
        [greedy_order(solve_prob, list(expected_spend), budget, alloc) for alloc in allocs],
    )


def evaluate_allocation(solve_prob, expected_spend, budget, alloc, order):
    """Per problem: solve probability and expected spend of an allocation deployed in `order`."""
    solve, spend = cell_outcomes(solve_prob, list(expected_spend), budget, alloc)
    log_fail = np.log(np.clip(1.0 - solve, 1e-12, 1.0))
    log_fail_ordered, spend_ordered = log_fail[:, order], spend[:, order]
    return 1.0 - np.exp(log_fail.sum(1)), (
        np.exp(np.cumsum(log_fail_ordered, 1) - log_fail_ordered) * spend_ordered
    ).sum(1)


def lower_envelope(solve, spend):
    """Lower-left envelope of pooled (solve rate, spend) points, sorted by solve rate: the points no other point
    beats on both."""
    order = np.argsort(solve)
    solve, spend = solve[order], spend[order]
    keep = np.zeros(solve.size, dtype=bool)
    cheapest = np.inf
    for i in range(solve.size - 1, -1, -1):
        if spend[i] < cheapest:
            keep[i], cheapest = True, spend[i]
    return solve[keep], spend[keep]


def envelope_indices(solve, spend):
    """Indices of the lower-left envelope of pooled points, sorted by spend."""
    keep, cheapest = [], np.inf
    for i in np.argsort(solve)[::-1]:
        if spend[i] < cheapest:
            keep.append(i)
            cheapest = spend[i]
    return np.array(sorted(keep, key=lambda i: spend[i]))


def along_envelope(x, spend_kept, values_kept):
    """values interpolated along the kept envelope points at spend x, clamped at the ends."""
    x = float(np.clip(x, spend_kept[0], spend_kept[-1]))
    hi = int(np.searchsorted(spend_kept, x))
    if hi == 0:
        return values_kept[0]
    lo = hi - 1
    frac = (x - spend_kept[lo]) / (spend_kept[hi] - spend_kept[lo])
    return (1 - frac) * values_kept[lo] + frac * values_kept[hi]


def best_single_model(model_solve, shadow_price, budget):
    """The market baseline under a shadow price: the (model, budget index) maximizing solve rate - price * budget."""
    value = model_solve - shadow_price * budget[None, :]
    m, j = np.unravel_index(int(np.argmax(value)), value.shape)
    return int(m), int(j)


def _budget_series(lo, hi, reaches, max_rows):
    """Budgets on the 1-2-5 series within [lo, hi] where `reaches(b)` holds, thinned to max_rows keeping the ends."""
    series = [
        m * 10.0**e for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1) for m in (1, 2, 5)
    ]
    budgets = [b for b in series if lo <= b <= hi and reaches(b)]
    if len(budgets) > max_rows:
        budgets = [budgets[int(round(i))] for i in np.linspace(0, len(budgets) - 1, max_rows)]
    return budgets


def summary_cv(record, max_rows=7):
    """Cross-validated solve rate of the arbitrageur and of the best single model at a few budgets per task."""
    num_problems = float(record["n_problems"])
    arb_solve, arb_spend = lower_envelope(
        np.asarray(record["cv_arb_solve"], float),
        np.asarray(record["cv_arb_spend"], float) / num_problems,
    )
    best_solve, best_spend = lower_envelope(
        np.asarray(record["cv_market_solve"], float),
        np.asarray(record["cv_market_spend"], float) / num_problems,
    )

    def solve_at(budget, solve, spend):  # flat past the last point, undefined below the first
        return np.nan if budget < spend[0] else float(np.interp(budget, spend, solve))

    budgets = _budget_series(
        arb_spend[0],
        max(arb_spend[-1], best_spend[-1]),
        lambda b: max(solve_at(b, arb_solve, arb_spend), solve_at(b, best_solve, best_spend)) >= 0.1,
        max_rows,
    )
    print(f"\n== {record['title']} ({record['cv']}, {record['n_problems']} problems)")
    print("  budget per task ($) | arbitrageur solve rate | best model solve rate")
    for b in budgets:
        print(
            f"  {b:19.4g} | {100 * solve_at(b, arb_solve, arb_spend):21.1f}% | {100 * solve_at(b, best_solve, best_spend):20.1f}%"
        )


def summary_revenue(split, record, min_share=0.05, max_rows=5):
    """Who the arbitrageur pays: at a few budgets per task, the held-out solve rate and the share of the spend that
    reaches each system (systems below `min_share` pooled into 'others'). `split`: the entry run_cv keeps in
    REVENUE_SPLITS."""
    models = [str(m) for m in split["models"]]
    solve, spend, revenue = (
        split["solve"].mean(1),
        split["spend"].mean(1),
        split["rev"].sum(1),
    )
    kept = envelope_indices(solve, spend)
    solve, spend, revenue = solve[kept], spend[kept], revenue[kept]
    budgets = _budget_series(spend[0], spend[-1], lambda b: along_envelope(b, spend, solve) >= 0.1, max_rows)
    print(f"\n== {record['title']} ({record['cv']}, {record['n_problems']} problems)")
    print("  budget per task ($) | solve rate | share of the arbitrageur's spend")
    for b in budgets:
        money = along_envelope(b, spend, revenue)
        share = money / max(money.sum(), 1e-12)
        top = [(share[j], models[j]) for j in np.argsort(-share) if share[j] >= min_share]
        others = 1.0 - sum(s for s, _ in top)
        text = ", ".join(f"{short_name(m)} {100 * s:.0f}%" for s, m in top) + (
            f", others {100 * others:.0f}%" if others >= 0.005 else ""
        )
        print(f"  {b:19.4g} | {100 * along_envelope(b, spend, solve):9.1f}% | {text}")


# ---------------------------------------------------------------------------------------------------------------
# System names and providers.
# ---------------------------------------------------------------------------------------------------------------
PROVIDERS = [  # (key, legend label, colour)
    ("openai", "OpenAI (GPT)", "#0F9D7A"),
    ("anthropic", "Anthropic (Claude)", "#D97757"),
    ("google", "Google (Gemini)", "#3B78E7"),
    ("other", "Other", "#7B3FA0"),
]
OTHER_PROVIDER_NAMES = (
    "minimax",
    "deepseek",
    "kimi",
    "glm",
    "qwen",
    "dscoder",
    "llama",
    "mistral",
    "codestral",
    "grok",
    "muse",
)
EFFORTS = ("low", "medium", "high", "xhigh", "max")  # DeepSWE reasoning-effort suffixes


def provider(name):
    """Provider of the model half of an Agent__Model name."""
    model = (name.split("__", 1)[1] if "__" in name else name).lower()
    if "claude" in model or "fable" in model or "opus" in model or "sonnet" in model:
        return "anthropic"
    if "gemini" in model:
        return "google"
    if "gpt" in model or "codex" in model:
        return "openai"
    if any(k in model for k in OTHER_PROVIDER_NAMES):
        return "other"
    raise ValueError(f"unknown provider for {name!r}")


def model_family(name):
    """A DeepSWE config name without its reasoning-effort suffix."""
    for effort in EFFORTS:
        if name.endswith("_" + effort):
            return name[: -len(effort) - 1]
    return name


def short_deepswe_name(name):
    family, effort = model_family(name), ""
    for e in EFFORTS:
        if name.endswith("_" + e):
            effort = e
    label = "-".join(w.capitalize() if w.isalpha() else w for w in family.split("_"))
    label = (
        label.replace("Claude-", "")
        .replace("Gpt", "GPT")
        .replace("Glm", "GLM")
        .replace("Deepseek", "DeepSeek")
        .replace("-Preview", "")
    )
    return f"{label} ({effort})" if effort else label


def short_name(name):
    """Display name of a system."""
    if "__" not in name and name == name.lower():  # DeepSWE config
        return short_deepswe_name(name)
    agent, _, model = name.replace(" (N=10)", "").partition("__")
    model = (
        model.replace("-Preview", "")
        .replace("gpt-5.3-codex", "GPT-5.3-Codex")
        .replace("Claude-Opus-4-7", "Claude-Opus-4.7")
    )
    agent = agent.replace("IndusAGICodingAgent", "IndusAGI").replace("Gemini_CLI", "Gemini-CLI")
    return f"{agent} · {model}" if model else agent


# ---------------------------------------------------------------------------------------------------------------
# Figure 1: the cost of reaching a target solve rate, per system.
# ---------------------------------------------------------------------------------------------------------------
COST_BAR_SETTINGS = {  # budget grid (log10 range of $ per task) and the label of the 'other' provider group
    "terminal_bench2_priced": dict(
        grid=(-4.0, np.log10(10.0)),
        other_label="Other (MiniMax / DeepSeek / Kimi / GLM)",
    ),
    "deepswe_priced": dict(
        grid=(-2.0, np.log10(200.0)),
        one_per_model=True,
        other_label="Other (DeepSeek / GLM / Kimi / Grok / Qwen / Muse)",
    ),
    "terminal_bench4_priced": dict(grid=(-2.0, np.log10(500.0)), other_label="Other (GLM / Grok / Gemini)"),
    "livecodebench_priced": dict(
        grid=(-7.0, np.log10(2.0)),
        other_label="Other (Qwen / DeepSeek / Llama / Mistral)",
    ),
}
TARGETS_BY_KEY = {
    "terminal_bench2_priced": [0.40, 0.70],
    "livecodebench_priced": [0.30, 0.50],
    "deepswe_priced": [0.50, 0.80],
    "terminal_bench4_priced": [0.30, 0.60],
}
OUTLIER_MULTIPLE = 20  # a system costing more than this multiple of the panel median is shown as NA
SORT_TARGET = 0  # the target whose cost fixes the x order
NA_COLOR = "0.55"


def single_model_curves(key):
    """Each system's (solve rate, expected spend per task) curve, from Chen's estimator continued past the recorded
    attempts by a Beta prior. Returns ({system: solve}, {system: spend}, models, number of tasks).
    """
    base_dir = DATASETS[key]["base_dir"]
    models = discover_models(base_dir)
    results = load_results(base_dir)
    attempt_price, ids = attempt_prices(results, models)
    budget = np.logspace(*COST_BAR_SETTINGS[key]["grid"], 600)
    solve_prob = market_passk(results, models, ids, attempt_price, budget, "chen_bb")
    solve, spend = {}, {}
    for j, m in enumerate(models):
        s, c = solve_prob[:, j, :].mean(0), expected_spend_curve(solve_prob[:, j, :], budget).mean(0)
        order = np.argsort(s)
        solve[m], spend[m] = s[order], c[order]
    return solve, spend, models, len(ids)


def cost_to_reach(solve, spend, target):
    """Expected $ per task at which a (solve rate, spend) curve reaches the target; NaN if it never does."""
    return np.nan if target > solve[-1] + 1e-9 else float(np.interp(target, solve, spend))


def cost_bars(key):
    settings = COST_BAR_SETTINGS[key]
    targets = TARGETS_BY_KEY[key]
    providers = [(k, settings["other_label"] if k == "other" else label, face) for k, label, face in PROVIDERS]
    solve, spend, models, num_tasks = single_model_curves(key)
    total = {
        t: {m: cost_to_reach(solve[m], spend[m], t) * num_tasks for m in models} for t in targets
    }  # over all tasks

    for t in targets:  # an outlier would set the y scale and flatten every other bar: shown as NA instead
        costs = np.array([v for v in total[t].values() if np.isfinite(v)])
        if costs.size:
            cap = OUTLIER_MULTIPLE * float(np.median(costs))
            for m, v in total[t].items():
                if np.isfinite(v) and v > cap:
                    print(
                        f"capped to NA at {int(t*100)}%: {short_name(m)} (${v:,.0f} > {OUTLIER_MULTIPLE}x median ${np.median(costs):,.0f})"
                    )
                    total[t][m] = np.nan

    if settings.get(
        "one_per_model"
    ):  # one reasoning-effort config per model: the one reaching the hardest target cheapest
        keep = {}
        for m in models:

            def rank(mm):
                costs = [total[t][mm] for t in targets]
                hardest = costs[-1]
                return (
                    np.isnan(hardest),
                    np.inf if np.isnan(hardest) else hardest,
                    np.inf if np.isnan(costs[SORT_TARGET]) else costs[SORT_TARGET],
                    mm,
                )

            family = model_family(m)
            if family not in keep or rank(m) < rank(keep[family]):
                keep[family] = m
        kept = sorted(keep.values())
        print(f"dedupe: {len(models)} configs -> {len(kept)} (one per model)")
        models = kept
    dropped = [m for m in models if all(np.isnan(total[t][m]) for t in targets)]
    models = [m for m in models if m not in dropped]
    if dropped:
        print("dropped (NA at every target):", ", ".join(short_name(m) for m in dropped))

    sort_target = targets[SORT_TARGET]
    other_targets = [t for t in targets if t != sort_target]

    def sort_key(m):
        c = total[sort_target][m]
        if not np.isnan(c):
            return (0, c, m)
        reached = [total[t][m] for t in other_targets if not np.isnan(total[t][m])]
        return (1, min(reached) if reached else np.inf, m)

    order = sorted(models, key=sort_key)
    xs = np.arange(len(order))
    providers_of = np.array([provider(m) for m in order])

    fig, axes = plt.subplots(len(targets), 1, sharex=True, figsize=(5.5, 1.10 * len(targets) + 1.24))
    for panel_idx, (ax, t) in enumerate(zip(axes, targets)):
        values = np.array([total[t][m] for m in order])
        reached = ~np.isnan(values)
        ymax = np.nanmax(values) * 1.08
        middle = reached & (xs > 0.25 * len(order)) & (xs < 0.75 * len(order))  # headroom for the centred target label
        label_y = 0.82 if panel_idx == 0 else 0.99
        floor = 0.88 if panel_idx == 0 else 0.86
        if middle.any():
            ymax = max(ymax, float(np.nanmax(values[middle])) / (label_y * floor))
        for provider_key, _, face in providers:
            sel = reached & (providers_of == provider_key)
            if sel.any():
                ax.bar(
                    xs[sel],
                    values[sel],
                    color=face,
                    width=0.78,
                    zorder=3,
                    edgecolor="white",
                    linewidth=0.5,
                )
        for x in xs[~reached]:
            ax.text(
                x,
                ymax * 0.015,
                "NA",
                ha="center",
                va="bottom",
                fontsize=FS["dense"],
                rotation=90,
                color=NA_COLOR,
                fontweight="bold",
            )
        ax.text(
            0.5,
            label_y,
            f"Target: {int(t*100)}% solve rate",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FS["annot"],
            fontweight="bold",
        )
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Total cost (USD)", fontsize=FS["label"])
        ax.tick_params(axis="y", labelsize=FS["tick"])
        ax.grid(axis="y", color=GRID, lw=0.5, ls="-")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(
        [short_name(m) for m in order],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=FS["dense"],
    )
    handles = [Patch(facecolor=face, edgecolor="white", linewidth=0.5, label=label) for _, label, face in providers]
    fig.tight_layout(h_pad=0.6)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.96),
        bbox_transform=axes[0].transAxes,
        fontsize=FS["legend"],
        frameon=False,
        ncol=2,
        handlelength=1.6,
        columnspacing=1.2,
        labelspacing=0.2,
        borderpad=0.0,
    )
    print(f"{len(order)} systems x {len(targets)} targets, {num_tasks} tasks")


# ---------------------------------------------------------------------------------------------------------------
# Figure 2: cross-validated frontiers.
# ---------------------------------------------------------------------------------------------------------------
LEFT_EDGE = {"LiveCodeBench": 0.01}  # left edge of a cost panel in $, where the default (the cheapest point) is too far


def draw_cost_panel(ax, record):
    """Cost against solve rate: every single model as a haze, the cross-validated arbitrageur on top."""
    solve_rate, floor = record["solve_rate"], record["floor"]
    model_cost = np.where(record["model_cost"] >= floor, record["model_cost"], np.nan)  # clipped at the budget floor
    arb_cost = np.where(record["arb_cost"] >= floor, record["arb_cost"], np.nan)
    for row in model_cost:
        ax.plot(
            row,
            solve_rate,
            color=MODEL_COLOR,
            alpha=MODEL_ALPHA,
            lw=MODEL_LW,
            zorder=5,
            solid_capstyle="round",
        )
    arb_cost_cv = np.asarray(record["arb_cost_cv"], dtype=float)
    arb_cost_cv = np.where(arb_cost_cv >= floor, arb_cost_cv, np.nan)
    ax.plot(
        arb_cost_cv,
        solve_rate,
        color=ARB_COLOR,
        alpha=ARB_ALPHA,
        lw=ARB_LW,
        zorder=3,
        solid_capstyle="round",
    )
    ax.set_title(record["title"], fontsize=FS["label"], pad=4)
    ax.set_xlabel("Total cost ($)", fontsize=FS["label"])
    style_axes(ax, ylabel="Solve rate", cost_on_x=True)
    costs = np.concatenate([model_cost.ravel(), arb_cost])
    costs = costs[np.isfinite(costs)]
    left = LEFT_EDGE.get(record.get("legend"))
    ax.set_xlim(float(costs.min()) if left is None else float(left), float(costs.max()))
    ax.set_ylim(0.0, float(solve_rate.max()))
    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR,
            alpha=min(1.0, MODEL_ALPHA * 2.2),
            lw=1.4,
            label="Models",
        ),
        Line2D(
            [0],
            [0],
            color=ARB_COLOR,
            alpha=ARB_ALPHA,
            lw=ARB_LW,
            label="Arbitrageur (ours)",
        ),
    ]


def draw_frontiers(records, figsize=(5.5, 1.95)):
    """One cost panel per market record, with a shared legend above."""
    fig, axs = plt.subplots(1, len(records), figsize=figsize)
    axs = np.atleast_1d(axs)
    handles = []
    for ax, record in zip(axs, records):
        handles = draw_cost_panel(ax, record)
    for ax in axs[1:]:
        ax.set_ylabel("")
    fig.tight_layout(w_pad=0.4)
    x0, x1 = axs[0].get_position().x0, axs[-1].get_position().x1
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=((x0 + x1) / 2, axs[0].get_position().y1 + 0.09),
        ncol=len(handles),
        fontsize=FS["legend"],
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
        labelcolor=INK,
    )


# ---------------------------------------------------------------------------------------------------------------
# Figure 3: revenue shares.
# ---------------------------------------------------------------------------------------------------------------
# The arbitrageur's cross-validated spend split per system, one entry per market, filled by the notebook's run_cv:
# dict(models, lams, solve[price, problem], spend[price, problem], rev[price, problem, system]).
REVENUE_SPLITS = {}
REVENUE_MARKETS = [
    "livecodebench_priced",
    "terminal_bench2_priced",
    "terminal_bench4_priced",
    "deepswe_priced",
]
REVENUE_TITLES = (
    "LiveCodeBench",
    "Terminal-Bench 2.0",
    "Terminal-Bench 4.0",
    "DeepSWE",
)
REVENUE_CMAP = LinearSegmentedColormap.from_list("rev", ["#ffffff", "#fbd0bd", ARB_COLOR, "#7a2d10"])
ROW_MIN_SHARE = 0.01  # a system must peak above this share to get a row
MONOPOLY_SHARE = 0.95  # a bin is a monopoly when one system holds more than this share
LEFT_MARGIN_DECADES = 0.15  # the axis starts this far before the first non-monopoly bin
ROW_HEIGHT_IN = 0.115  # heatmap row height, inches
BIN_WIDTH_PP = {
    "terminal_bench2_priced": 2.0,
    "deepswe_priced": 1.5,
    "terminal_bench4_priced": 3.0,
    "livecodebench_priced": 0.5,
}


def mean_attempt_price(key, model):
    """Mean price of one attempt of a system, averaged over tasks."""
    path = os.path.join(DATASETS[key]["base_dir"], f"{model}.jsonl")
    return float(np.mean([json.loads(line)["mean_cost"] for line in open(path)]))


def revenue_bins(key):
    """(system labels, attempt prices, budget bin centres, share[bin, system]) of a market's revenue split: the pooled
    held-out points on their envelope, binned by solve rate, each bin's spend split by system.
    """
    split = REVENUE_SPLITS[key]
    models = split["models"]
    solve_rate = split["solve"].mean(1) * 100.0
    spend = split["spend"].mean(1)
    revenue = split["rev"].sum(1)  # [price, system]
    kept = envelope_indices(solve_rate / 100.0, spend)
    solve_rate, revenue, spend = solve_rate[kept], revenue[kept], spend[kept]
    share = revenue / np.clip(revenue.sum(1, keepdims=True), 1e-12, None)
    reshuffle = np.abs(np.diff(share, axis=0)).sum(1) * 100.0
    spike = np.nonzero((solve_rate[1:] >= solve_rate.max() - 3.0) & (reshuffle > 20.0))[
        0
    ]  # the frontier's terminal spike
    if spike.size:
        solve_rate, revenue, spend = (a[: spike[0] + 1] for a in (solve_rate, revenue, spend))
    bin_width = BIN_WIDTH_PP.get(key, 2.0)
    edges = np.arange(np.floor(solve_rate.min()), solve_rate.max() + bin_width, bin_width)
    bin_of = np.clip(np.digitize(solve_rate, edges) - 1, 0, len(edges) - 2)
    centres, revenue_per_bin = [], []
    for b in range(len(edges) - 1):
        in_bin = bin_of == b
        if not in_bin.any():
            continue
        weight = revenue[in_bin].sum(1)
        centres.append(float((spend[in_bin] * weight).sum() / weight.sum()))  # money-weighted mean spend of the bin
        revenue_per_bin.append(revenue[in_bin].sum(0))
    revenue_per_bin = np.stack(revenue_per_bin)  # [bin, system]
    share = revenue_per_bin / np.clip(revenue_per_bin.sum(1, keepdims=True), 1e-12, None)
    prices = np.array([mean_attempt_price(key, m) for m in models])
    return [short_name(m) for m in models], prices, np.array(centres), share


def draw_revenue_heatmap(key, ax):
    labels, prices, centres, share = revenue_bins(key)
    rows = np.nonzero(share.max(0) > ROW_MIN_SHARE)[0]
    rows = rows[np.argsort(prices[rows])]  # cheapest at the top
    cells = share[:, rows].T * 100  # [row, bin]
    edges = np.sqrt(centres[1:] * centres[:-1])
    edges = np.concatenate([[centres[0] ** 2 / edges[0]], edges, [centres[-1] ** 2 / edges[-1]]])
    ys = np.arange(len(rows) + 1)
    mesh = ax.pcolormesh(
        edges,
        ys,
        cells[::-1],
        cmap=REVENUE_CMAP,
        vmin=0,
        vmax=100,
        edgecolors="white",
        linewidth=0.4,
    )
    ax.set_xscale("log")
    dominant = cells.max(0)  # drop the ultra-low budgets where one system holds essentially all the spend
    first = int(np.argmax(dominant < 100 * MONOPOLY_SHARE)) if (dominant < 100 * MONOPOLY_SHARE).any() else 0
    ax.set_xlim(max(edges[first] / 10**LEFT_MARGIN_DECADES, edges[0]), edges[-1])
    ax.set_yticks(ys[:-1] + 0.5)
    ax.set_yticklabels([labels[j] for j in rows][::-1], fontsize=FS["dense"])
    ax.tick_params(axis="x", labelsize=FS["tick"], length=2.5)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def draw_revenue_shares():
    """The four markets as a 2 x 2 grid of revenue-share heatmaps."""
    fig_w = 5.5
    num_rows = [int((revenue_bins(key)[3].max(0) > ROW_MIN_SHARE).sum()) for key in REVENUE_MARKETS]
    decor_in = 0.62  # title, x axis and label per grid row
    h1, h2 = (
        max(num_rows[0], num_rows[1]) * ROW_HEIGHT_IN + decor_in,
        max(num_rows[2], num_rows[3]) * ROW_HEIGHT_IN + decor_in,
    )
    fig = plt.figure(figsize=(fig_w, h1 + h2))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.155,
        right=0.93,
        bottom=0.0,
        top=1.0,
        wspace=1.05,
        hspace=0.0,
        height_ratios=[h1, h2],
    )
    axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    for ax, key, title in zip(axs, REVENUE_MARKETS, REVENUE_TITLES):
        mesh = draw_revenue_heatmap(key, ax)
        ax.set_title(title, fontsize=FS["title"], pad=4)
        ax.set_xlabel("Budget per task ($)", fontsize=FS["label"])
        width_in = (ax.get_position().x1 - ax.get_position().x0) * fig_w
        ax.set_box_aspect(len(ax.get_yticks()) * ROW_HEIGHT_IN / width_in)
        ax.set_anchor("N")
    fig.canvas.draw()
    top, bottom = max(ax.get_position().y1 for ax in axs), min(ax.get_position().y0 for ax in axs)
    colorbar = fig.colorbar(mesh, cax=fig.add_axes([gs.right + 0.03, bottom, 0.012, top - bottom]))
    colorbar.set_label("Revenue share (%)", fontsize=FS["label"])
    colorbar.ax.tick_params(labelsize=FS["tick"])


# ---------------------------------------------------------------------------------------------------------------
# Figure 4: deployment order.
# ---------------------------------------------------------------------------------------------------------------
DP_MAX_FUNDED = 16  # exact best fixed order by subset DP up to 2^16 states; local search beyond
ORDER_NAMES = ["greedy", "cheapest", "best", "oracle"]
ORDER_FIGURE_MARKETS = [
    "livecodebench_priced",
    "terminal_bench2_priced",
    "terminal_bench4_priced",
]
ORDER_LINES = {  # order -> (label, colour, linestyle)
    "greedy": ("Greedy (ours)", ARB_COLOR, "-"),
    "cheapest": ("Cheapest first", MODEL_COLOR, (0, (4, 2))),
}


def oracle_per_problem(solve_prob, expected_spend, budget, alloc):
    """Per-problem expected spend if each problem could be served in its own best order, an upper bound no
    deployable order can reach."""
    solve, spend = cell_outcomes(solve_prob, list(expected_spend), budget, alloc)
    value = np.where(spend > 0, solve / np.where(spend > 0, spend, 1.0), -1.0)
    order = np.argsort(-value, axis=1)
    log_fail = np.log(np.clip(1.0 - solve, 1e-12, 1.0))
    log_fail, spend = np.take_along_axis(log_fail, order, axis=1), np.take_along_axis(spend, order, axis=1)
    return (np.exp(np.cumsum(log_fail, axis=1) - log_fail) * spend).sum(1)


def _spend_of_permutation(log_fail, spend, perm):
    lf, sp = log_fail[:, perm], spend[:, perm]
    return float((np.exp(np.cumsum(lf, 1) - lf) * sp).sum(1).mean())


def best_order_by_local_search(log_fail, spend, start):
    """Descent over pairwise swaps and single-element insertions from `start`, a permutation of range(n).
    log_fail, spend: [problem, model] for the funded models. Returns (order, expected spend).
    """
    order, current, n = (
        list(start),
        _spend_of_permutation(log_fail, spend, start),
        len(start),
    )
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = order.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                value = _spend_of_permutation(log_fail, spend, candidate)
                if value < current - 1e-15:
                    order, current, improved = candidate, value, True
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                candidate = order.copy()
                candidate.insert(j, candidate.pop(i))
                value = _spend_of_permutation(log_fail, spend, candidate)
                if value < current - 1e-15:
                    order, current, improved = candidate, value, True
    return order, current


def best_order_by_dp(fail, spend):
    """Exact minimum-expected-spend order by dynamic programming over subsets: the probability of reaching a model
    depends only on the set queried before it. fail, spend: [problem, model] for the funded models.
    Returns (order, expected spend)."""
    num_problems, n = fail.shape
    num_subsets = 1 << n
    reach = np.empty((num_subsets, num_problems))  # probability every model in the subset failed, per problem
    reach[0] = 1.0
    for subset in range(1, num_subsets):
        lowest = subset & -subset
        reach[subset] = reach[subset ^ lowest] * fail[:, lowest.bit_length() - 1]
    step = reach @ spend / num_problems  # [subset, model]: expected spend of querying the model after the subset
    cost = np.full(num_subsets, np.inf)
    cost[0] = 0.0
    last = np.full(num_subsets, -1, dtype=int)
    for subset in range(1, num_subsets):
        for m in range(n):
            if subset >> m & 1:
                value = cost[subset ^ (1 << m)] + step[subset ^ (1 << m), m]
                if value < cost[subset]:
                    cost[subset], last[subset] = value, m
    order, subset = [], num_subsets - 1
    while subset:
        order.append(last[subset])
        subset ^= 1 << last[subset]
    return order[::-1], float(cost[num_subsets - 1])


def summary_order(record):
    """Expected spend of the greedy and the cheapest-first order relative to the best fixed order, and of the best
    fixed order relative to the per-problem oracle, cross-validated and in-sample, at a few solve rates.
    """
    solve_rate = record["solve_cv"] * 100
    greedy, cheapest, best, oracle = (np.asarray(record[f"spend_{name}_cv"]) for name in ORDER_NAMES)
    greedy_in, cheapest_in, best_in, oracle_in = (np.asarray(record[f"spend_{name}_in"]) for name in ORDER_NAMES)
    print(f"\n== {record['title']} ({record['cv']}, {record['n_problems']} problems, {record['n_models']} models)")
    print(
        "  solve | #funded | CV spend over best fixed order: greedy cheapest | best over oracle | in-sample: greedy cheapest | best over oracle"
    )
    for target in [30, 40, 50, 60, 70, 80, 90, 95, 98]:
        if target < solve_rate.min() or target > solve_rate.max():
            continue
        k = int(np.argmin(np.abs(solve_rate - target)))
        print(
            f"  {solve_rate[k]:5.1f}% |  {record['n_funded'][k]:2d}     |   {greedy[k]/best[k]:6.3f}   {cheapest[k]/best[k]:6.3f}         |   {best[k]/oracle[k]:6.3f}    "
            f"|   {greedy_in[k]/best_in[k]:6.3f}   {cheapest_in[k]/best_in[k]:6.3f}     |   {best_in[k]/oracle_in[k]:6.3f}"
        )
    ok = best > 0
    for label, g, c, b in [
        ("CV", greedy, cheapest, best),
        ("in-sample", greedy_in, cheapest_in, best_in),
    ]:
        print(
            f"  {label} greedy over best, across lambda: min {np.min(g[ok]/b[ok]):.3f} median {np.median(g[ok]/b[ok]):.3f} max {np.max(g[ok]/b[ok]):.3f};"
            f"  cheapest over best: min {np.min(c[ok]/b[ok]):.3f} median {np.median(c[ok]/b[ok]):.3f} max {np.max(c[ok]/b[ok]):.3f}"
        )


def draw_order_panel(ax, record, ylabel):
    """In-sample spend of the greedy and the cheapest-first order relative to the best fixed order, against solve rate."""
    solve_rate = record["solve_in"] * 100.0
    best = record["spend_best_in"]
    sel = np.isfinite(solve_rate) & (best > 0)
    order = np.argsort(solve_rate[sel])
    ylo, yhi = 0.0, 0.0
    for name, (label, color, ls) in ORDER_LINES.items():
        excess = (record[f"spend_{name}_in"] / best - 1.0) * 100.0
        ax.plot(
            solve_rate[sel][order],
            excess[sel][order],
            color=color,
            ls=ls,
            lw=ARB_LW,
            alpha=LINE_ALPHA if name == "greedy" else 0.9,
            zorder=5 if name == "greedy" else 4,
            solid_capstyle="round",
            dash_capstyle="round",
            label=label,
        )
        ylo, yhi = min(ylo, float(np.nanmin(excess[sel]))), max(yhi, float(np.nanmax(excess[sel])))
    ax.axhline(0.0, color=AXIS, lw=0.6, zorder=1)
    ax.set_title(record["title"], fontsize=FS["title"], pad=6)
    ax.set_xlabel("Solve rate", fontsize=FS["label"])
    style_axes(ax, ylabel=ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    several_funded = sel & (np.asarray(record["n_funded"]) > 1)  # x starts where more than one model is funded
    x0, x1 = (float(np.nanmin(solve_rate[several_funded])) if several_funded.any() else 0.0), float(
        np.nanmax(solve_rate[sel])
    )
    ax.set_xlim(x0, x1)
    if x1 - x0 <= 50:
        ax.xaxis.set_major_locator(MultipleLocator(10))
    pad = 0.06 * (yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + pad)
    legend = ax.legend(
        fontsize=FS["legend"],
        loc="upper left",
        handlelength=2.2,
        handletextpad=0.6,
        borderaxespad=0.3,
    )
    for text in legend.get_texts():
        text.set_color(INK)


def draw_order(records, figsize=(5.5, 1.95)):
    fig, axs = plt.subplots(1, len(records), figsize=figsize, squeeze=False)
    for i, (ax, record) in enumerate(zip(axs.ravel(), records)):
        draw_order_panel(ax, record, ylabel="Cost relative to\nbest fixed order" if i == 0 else None)
        lo, hi = ax.get_ylim()
        if hi - lo < 0.5:  # both orders optimal everywhere: keep a readable axis
            ax.set_ylim(-0.05, 1.0)
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    fig.tight_layout(w_pad=0.4)


# ---------------------------------------------------------------------------------------------------------------
# Estimation benchmarks (Figure 5 and the estimation-error figure): the calibration protocol.
# ---------------------------------------------------------------------------------------------------------------
CALIBRATION_SIZES = (2, 4, 8, 16)  # calibration attempts per cell
NUM_REPLICATES = 10  # replicates per calibration size
NUM_BUDGETS = 64
SEED = 0
TARGETS = {
    "monkey_math": (0.5, 0.7, 0.8, 0.9, 0.95),
    "monkey_codecontests": (0.1, 0.15, 0.2, 0.25, 0.3, 0.35),
    "irsl_aime2024": (0.4, 0.5, 0.6, 0.7, 0.8),
    "irsl_aime2025": (0.4, 0.5, 0.6, 0.7, 0.8),
    "irsl_gmmlu": (0.6, 0.7, 0.8, 0.9, 0.95),
    "irsl_mmlupro": (0.6, 0.7, 0.8, 0.9, 0.95),
}
DISPLAY_NAMES = {
    "monkey_math": "MATH",
    "monkey_codecontests": "CodeContests",
    "irsl_aime2024": "AIME 2024",
    "irsl_aime2025": "AIME 2025",
    "irsl_gmmlu": "Global-MMLU-Lite",
    "irsl_mmlupro": "MMLU-Pro",
}
_LOADED = {}


def load_benchmark(key):
    """(benchmark, models, attempt_price[problem, model], successes, attempts, budget grid), loaded once per benchmark."""
    if key not in _LOADED:
        benchmark = DATASETS[key]
        models = benchmark["models"] or discover_models(benchmark["base_dir"])
        results = load_results(benchmark["base_dir"])
        attempt_price, ids = attempt_prices(results, models)
        successes, attempts = counts(results, models, ids)
        budget = np.logspace(
            np.log10(benchmark["bmin"] * 0.5),
            np.log10(benchmark["bmax"] * 20),
            NUM_BUDGETS,
        )
        _LOADED[key] = (benchmark, models, attempt_price, successes, attempts, budget)
    return _LOADED[key]


def budget_caps(max_attempts, attempt_price):
    """[model] budget beyond which every problem's curve is flat: max_attempts at the model's highest price."""
    return np.asarray(max_attempts, float) * attempt_price.max(0)


def freeze_cells(solve_prob, budget, attempt_price, attempt_cap):
    """Flatten each cell's curve beyond its own attempt cap attempt_cap[problem, model] (in place)."""
    num_problems, num_models, num_budgets = solve_prob.shape
    for j in range(num_models):
        cap_budget = attempt_cap[:, j] * attempt_price[:, j]
        idx = np.clip(np.searchsorted(budget, cap_budget), 1, num_budgets - 1)
        b0, b1 = budget[idx - 1], budget[idx]
        frac = np.clip((cap_budget - b0) / (b1 - b0), 0.0, 1.0)
        rows = np.arange(num_problems)
        at_cap = (1 - frac) * solve_prob[rows, j, idx - 1] + frac * solve_prob[rows, j, idx]
        solve_prob[:, j, :] = np.where(budget[None, :] > cap_budget[:, None], at_cap[:, None], solve_prob[:, j, :])
    return solve_prob


def curves_from_counts(successes, attempts, attempt_price, budget, estimator, max_attempts, attempt_cap):
    """solve_prob[problem, model, budget] from counts: 'geom' (the plug-in), 'zibb' (ours) or 'chen' (the oracle),
    at most max_attempts[model] attempts, each cell flat beyond its own cap attempt_cap[problem, model].
    """
    num_problems, num_models = successes.shape
    max_attempts = np.broadcast_to(np.asarray(max_attempts, float), (num_models,))
    solve_prob = np.empty((num_problems, num_models, budget.size))
    for j in range(num_models):
        k = np.minimum(budget[None, :] / attempt_price[:, j, None], max_attempts[j])  # attempts affordable
        if estimator == "geom":
            solve_prob[:, j, :] = plugin_passk(successes[:, j], attempts[:, j], k)
        elif estimator == "zibb":
            a, b, spike = fit_zero_inflated_beta(successes[:, j], attempts[:, j])
            solve_prob[:, j, :] = zero_inflated_beta_passk(successes[:, j], attempts[:, j], k, a, b, spike)
        elif estimator == "chen":
            solve_prob[:, j, :] = chen_passk(successes[:, j], attempts[:, j], k)
        else:
            raise ValueError(estimator)
    return freeze_cells(solve_prob, budget, attempt_price, attempt_cap)


def split_attempts(successes, attempts, num_calibration, rng):
    """Draw `num_calibration` attempts without replacement from each cell: (calibration successes, calibration
    attempts, held-out successes, held-out attempts)."""
    calib_attempts = np.minimum(attempts, num_calibration).astype(int)
    calib_successes = rng.hypergeometric(
        successes.astype(int), (attempts - successes).astype(int), calib_attempts
    ).astype(float)
    calib_attempts = calib_attempts.astype(float)
    return (
        calib_successes,
        calib_attempts,
        successes - calib_successes,
        attempts - calib_attempts,
    )


def score_allocations(solve_prob, expected_spend, budget, allocs):
    """(solve rate, expected spend) of each allocation on the given curves."""
    solve = np.array([pooled_solve_rate(solve_prob, budget, alloc) for alloc in allocs])
    spend = np.array([greedy_spend(solve_prob, expected_spend, budget, alloc) for alloc in allocs])
    return solve, spend


def oracle_frontier(key):
    """The arbitrageur planning on the oracle curves, Chen's estimator on all recorded attempts: (solve rates,
    spends) along its frontier, and the number of problems."""
    _, models, attempt_price, successes, attempts, budget = load_benchmark(key)
    solve_prob = curves_from_counts(successes, attempts, attempt_price, budget, "chen", attempts.max(0), attempts)
    solve, spend, _ = arbitrage_frontier(
        solve_prob,
        budget,
        num=70,
        budget_cap=budget_caps(attempts.max(0), attempt_price),
    )
    return solve, spend, solve_prob.shape[0]


# ---------------------------------------------------------------------------------------------------------------
# Figure 5: arbitrage on estimated curves.
# ---------------------------------------------------------------------------------------------------------------
ESTIMATOR_NAMES = {"geom": "Plug-in", "zibb": "Beta-binomial"}
BAND = (25, 75)  # percentile band over the replicates
GRID_POINTS = 300  # solve-rate grid points
CALIBRATION_PANELS = [
    "monkey_math",
    "monkey_codecontests",
    "irsl_aime2024",
    "irsl_aime2025",
    "irsl_gmmlu",
    "irsl_mmlupro",
]
CALIBRATION_COLUMNS = 3  # panels per row
CALIBRATION_SIZES_DRAWN = [2, 4, 8, 16]
CALIBRATION_RECORDS = {}  # (benchmark, calibration size) -> record, filled by summarize_calibration
CALIBRATION_LINES = {
    "geom": ("Arbitrageur using plug-in", MODEL_COLOR, "o"),
    "zibb": ("Arbitrageur using EB pass@$k$ (ours)", ARB_COLOR, "s"),
}
ORACLE_LS = (0, (4, 2))


def envelope_cost(solve, spend, grid):
    """Cheapest cost reaching each solve rate of `grid` (lower envelope, interpolated in log cost); NaN past reach."""
    keep = spend > 0
    solve, spend = np.asarray(solve)[keep], np.asarray(spend)[keep]
    order = np.argsort(spend)
    solve, spend = solve[order], spend[order]
    best_solve = np.maximum.accumulate(solve)
    keep = np.concatenate([[True], np.diff(best_solve) > 0])
    best_solve, spend = best_solve[keep], spend[keep]
    out = np.exp(np.interp(grid, best_solve, np.log(spend), left=np.nan, right=np.nan))
    out[grid > best_solve[-1]] = np.nan
    return out


def band_stats(values, fill):
    """Median and quartiles over the replicates (axis 0); a replicate that does not reach counts as `fill`."""
    values = np.where(np.isfinite(values), values, fill)
    median = np.median(values, 0)
    lo, hi = np.percentile(values, BAND, axis=0)
    return [np.where(np.isfinite(x), x, np.nan) for x in (median, lo, hi)]


def realized_cost_ratio(replicates, estimator, grid, oracle_cost, num_problems):
    """Realized cost of the plans made on one estimator's curves, as a multiple of the oracle's, on `grid`:
    median and quartiles over the replicates."""
    realized = np.array([envelope_cost(*r[estimator]["realized"], grid) * num_problems for r in replicates])
    with np.errstate(all="ignore"):
        median, lo, hi = band_stats(realized / oracle_cost[None, :], np.inf)
    return dict(realized_ratio_med=median, realized_ratio_lo=lo, realized_ratio_hi=hi)


def solve_rate_at_budget(cost, grid, budget):
    """Solve rate a cost-to-reach-solve-rate curve delivers at a deployment budget."""
    ok = np.isfinite(cost) & (cost > 0)
    cost, grid = cost[ok], grid[ok]
    order = np.argsort(cost)
    return float(np.interp(budget, cost[order], grid[order]))


def summarize_calibration(replicates_by_benchmark):
    """Summarize the calibration replicates of each benchmark at its target solve rates, per calibration size.
    `replicates_by_benchmark`: {benchmark: [calibrate_one results]}. Fills CALIBRATION_RECORDS.
    """
    for key, replicates in replicates_by_benchmark.items():
        oracle_solve, oracle_spend, num_problems = oracle_frontier(key)
        print(f"[{key}] {len(replicates)} replicates")
        grid = np.linspace(0.02, float(oracle_solve.max()), GRID_POINTS)
        oracle_cost = envelope_cost(oracle_solve, oracle_spend, grid) * num_problems
        targets = np.array(TARGETS[key], float)
        oracle_cost_at_targets = envelope_cost(oracle_solve, oracle_spend, targets) * num_problems
        for size in sorted(set(r["n_sub"] for r in replicates)):
            of_size = [r for r in replicates if r["n_sub"] == size]
            record = dict(
                legend=DISPLAY_NAMES[key],
                n_sub=size,
                solve_rate=grid * 100.0,
                oracle_cost=oracle_cost,
                est={},
                targets=dict(solve_rate=targets * 100.0, oracle_cost=oracle_cost_at_targets),
            )
            for estimator in ESTIMATOR_NAMES:
                if estimator in of_size[0]:
                    record["est"][estimator] = realized_cost_ratio(of_size, estimator, grid, oracle_cost, num_problems)
            CALIBRATION_RECORDS[(key, size)] = record
            # the quantity Figure 5 draws: at the deployment budget the oracle needs for each target, the solve rate
            # the plan made on each estimator's curves actually reaches (median over replicates)
            print(
                f"  n={size} attempts per cell: solve rate reached at the budget the oracle needs for each target (medians over replicates)"
            )
            print("     budget | oracle | " + " | ".join(f"{ESTIMATOR_NAMES[e]:>13s}" for e in record["est"]))
            for t, budget_at_target in zip(targets, oracle_cost_at_targets):
                reached = [
                    solve_rate_at_budget(
                        oracle_cost * record["est"][e]["realized_ratio_med"],
                        grid * 100.0,
                        float(budget_at_target),
                    )
                    for e in record["est"]
                ]
                print(
                    f"  {float(budget_at_target):9.3g} | {t*100:5.0f}% | " + " | ".join(f"{r:12.1f}%" for r in reached)
                )


def size_style(i):
    """(linestyle, alpha) for the i-th smallest calibration size: one solid line per size on a lightness ramp."""
    return "-", 0.30 + 0.70 * (i / (len(CALIBRATION_SIZES_DRAWN) - 1))


class StackedHandle:
    """Legend proxy for an orange estimator line stacked above its blue counterpart."""

    def __init__(self, linestyle, alpha):
        self.linestyle = linestyle
        self.alpha = alpha


class StackedHandler(HandlerBase):
    def create_artists(self, legend, handle, xdescent, ydescent, width, height, fontsize, trans):
        x = [xdescent, xdescent + width]
        lines = [
            Line2D(
                x,
                [ydescent + 0.85 * height] * 2,
                color=ARB_COLOR,
                ls=handle.linestyle,
                lw=ARB_LW,
                alpha=handle.alpha,
            ),
            Line2D(
                x,
                [ydescent + 0.15 * height] * 2,
                color=MODEL_COLOR,
                ls=handle.linestyle,
                lw=ARB_LW,
                alpha=handle.alpha,
            ),
        ]
        for line in lines:
            line.set_transform(trans)
        return lines


def draw_calibration():
    """Solve rate reached against the oracle's, at the budget the oracle needs for each target. The oracle is the
    diagonal; a plan that falls short drops below it. One panel per benchmark, CALIBRATION_COLUMNS per row."""
    num_rows = -(-len(CALIBRATION_PANELS) // CALIBRATION_COLUMNS)
    fig, grid_axs = plt.subplots(num_rows, CALIBRATION_COLUMNS, figsize=(5.5, 1.75 * num_rows), squeeze=False)
    axs = grid_axs.ravel()
    for ax in axs[len(CALIBRATION_PANELS) :]:
        ax.set_visible(False)
    for ax, key in zip(axs, CALIBRATION_PANELS):
        records = {size: CALIBRATION_RECORDS[(key, size)] for size in CALIBRATION_SIZES_DRAWN}
        first = records[CALIBRATION_SIZES_DRAWN[0]]
        targets = np.array(first["targets"]["solve_rate"], float)
        budgets_at_targets = first["targets"]["oracle_cost"]
        pad = 0.06 * (targets.max() - targets.min())
        lo, hi = targets.min() - pad, targets.max() + pad
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.9, ls=ORACLE_LS, zorder=3)  # the oracle
        for estimator, (label, color, marker) in CALIBRATION_LINES.items():
            for i, size in enumerate(CALIBRATION_SIZES_DRAWN):
                ls, alpha = size_style(i)
                record = records[size]
                grid, oracle_cost = np.array(record["solve_rate"], float), np.array(record["oracle_cost"], float)
                reached = (
                    {}
                )  # a higher cost ratio reaches a lower solve rate at the same budget: hi is the band's bottom
                for tag in ("med", "lo", "hi"):
                    ratio = np.asarray(record["est"][estimator][f"realized_ratio_{tag}"], float)
                    reached[tag] = np.array(
                        [solve_rate_at_budget(oracle_cost * ratio, grid, b) for b in budgets_at_targets]
                    )
                ax.errorbar(
                    targets,
                    reached["med"],
                    yerr=[
                        reached["med"] - reached["hi"],
                        reached["lo"] - reached["med"],
                    ],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.7,
                    capsize=1.2,
                    capthick=0.7,
                    alpha=alpha * 0.9,
                    zorder=4,
                )
                ax.plot(
                    targets,
                    reached["med"],
                    color=color,
                    ls=ls,
                    marker=marker,
                    ms=3.0,
                    lw=ARB_LW,
                    alpha=alpha,
                    zorder=5,
                    solid_capstyle="round",
                )
        ax.set_title(first["legend"], fontsize=FS["annot"], pad=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, lw=0.5, ls="-", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=0.5, labelsize=FS["tick"], colors=AXIS, labelcolor=INK)
    for row in grid_axs:
        row[0].set_ylabel("Solve rate", fontsize=FS["label"])
    method_handles = [Line2D([], [], color=INK, lw=0.9, ls=ORACLE_LS, label="Arbitrageur using oracle")]
    for estimator, (label, color, marker) in CALIBRATION_LINES.items():
        method_handles.append(Line2D([], [], color=color, marker=marker, ms=3.0, lw=ARB_LW, label=label))
    size_handles = [StackedHandle(*size_style(i)) for i in range(len(CALIBRATION_SIZES_DRAWN))]
    size_labels = [f"$n={size}$" for size in CALIBRATION_SIZES_DRAWN]
    fig.tight_layout(w_pad=0.4, h_pad=1.4)
    p0, p1 = grid_axs[0][0].get_position(), grid_axs[0][-1].get_position()  # the legends go above the top row
    bottom = grid_axs[-1][0].get_position()  # the shared x label goes below the bottom row
    center = (p0.x0 + p1.x1) / 2
    fig.text(
        center,
        bottom.y0 - 0.16 / num_rows,
        "Oracle solve rate",
        ha="center",
        va="top",
        fontsize=FS["label"],
        color=INK,
    )
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(center, p0.y1 + 0.15 / num_rows),
        ncol=len(method_handles),
        fontsize=FS["legend"],
        frameon=False,
        handlelength=1.6,
        columnspacing=0.8,
        handletextpad=0.5,
        labelcolor=INK,
    )
    fig.legend(
        handles=size_handles,
        labels=size_labels,
        loc="lower center",
        bbox_to_anchor=(center, p0.y1 + 0.05 / num_rows),
        ncol=len(size_handles),
        fontsize=FS["legend"],
        frameon=False,
        handlelength=1.6,
        columnspacing=1.8,
        handletextpad=0.5,
        labelcolor=INK,
        markerfirst=True,
        handler_map={StackedHandle: StackedHandler()},
    )


# ---------------------------------------------------------------------------------------------------------------
# Estimation error: pass@k RMSE of the plug-in and ours against the oracle, over calibration draws.
# ---------------------------------------------------------------------------------------------------------------
GAP_BENCHMARKS = [
    ("monkey_math", "MATH"),
    ("monkey_codecontests", "CodeContests"),
    ("irsl_aime2024", "AIME 2024"),
    ("irsl_aime2025", "AIME 2025"),
]
GAP_BENCHMARKS_APPENDIX = [
    ("irsl_gmmlu", "Global-MMLU-Lite"),
    ("irsl_mmlupro", "MMLU-Pro"),
]
GAP_CALIBRATION_SIZE = 8
GAP_DRAWS = 100
GAP_K_GRID = [
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
]  # the curves are scored on this grid of k
GAP_BINS = 16
GAP_SERIES = {"ours": ("Plug-in $-$ ours", ARB_COLOR)}


def estimate_passk_draws(key, num_calibration, num_draws):
    """(oracle pass@k [problem, model, k], {estimator: [draw, problem, model, k]}) over the seeded calibration draws."""
    _, models, _, successes, attempts, _ = load_benchmark(key)
    num_models = len(models)
    k = np.array(GAP_K_GRID, float)
    oracle_passk = np.stack(
        [chen_passk(successes[:, j], attempts[:, j], k) for j in range(num_models)],
        axis=1,
    )
    estimates = {"plug": [], "ours": []}
    for draw in range(num_draws):
        rng = np.random.default_rng(SEED + 1000 * num_calibration + draw)
        calib_successes, calib_attempts, _, _ = split_attempts(successes, attempts, num_calibration, rng)
        plugin, ours = np.empty_like(oracle_passk), np.empty_like(oracle_passk)
        for j in range(num_models):
            a, b, spike = fit_zero_inflated_beta(calib_successes[:, j], calib_attempts[:, j])
            plugin[:, j] = plugin_passk(calib_successes[:, j], calib_attempts[:, j], k)
            ours[:, j] = zero_inflated_beta_passk(calib_successes[:, j], calib_attempts[:, j], k, a, b, spike)
        estimates["plug"].append(plugin)
        estimates["ours"].append(ours)
        if draw % 20 == 19:
            print(f"[{key}] draw {draw + 1} of {num_draws}", flush=True)
    return oracle_passk, {name: np.array(x) for name, x in estimates.items()}


def draw_gap_panel(ax, gap_by_series, name):
    """Histogram of the paired RMSE gap over draws, with the share of draws above zero."""
    all_gaps = np.concatenate(list(gap_by_series.values()))
    lo, hi = all_gaps.min(), all_gaps.max()
    pad = 0.10 * (hi - lo) if hi > lo else 1e-3
    width = (hi - lo + 2 * pad) / GAP_BINS  # bin edges anchored at zero so no bar straddles the reference line
    edges = np.arange(np.floor((lo - pad) / width), np.ceil((hi + pad) / width) + 1) * width
    x0, x1 = min(edges[0], 0.0), max(edges[-1], 0.0)
    zero_frac = np.clip((0.0 - x0) / (x1 - x0), 0.0, 1.0)  # the zero line in axes fraction
    text_x, text_ha = (zero_frac - 0.03, "right") if zero_frac > 0.25 else (zero_frac + 0.06, "left")
    for i, (series, (label, color)) in enumerate(GAP_SERIES.items()):
        gaps = gap_by_series[series]
        ax.hist(
            gaps,
            bins=edges,
            color=color,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
        ax.text(
            text_x,
            0.96 - 0.13 * i,
            f"{np.mean(gaps > 0) * 100:.0f}% > 0",
            transform=ax.transAxes,
            ha=text_ha,
            va="top",
            fontsize=FS["annot"],
            color=color,
        )
    ax.axvline(0.0, color="#b02a2e", lw=0.9, ls=(0, (3, 2)), zorder=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=False))
    ax.set_title(name, fontsize=FS["annot"], pad=3)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.5, labelsize=FS["tick"], colors=AXIS, labelcolor=INK)
