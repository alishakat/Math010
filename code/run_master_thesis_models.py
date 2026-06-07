from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Run the modelling tables used in the master thesis."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_dir / "config" / "model_config.json",
        help="JSON file containing model settings and control-variable lists.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data",
        help="Folder containing the clean thesis CSV inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "tables",
        help="Folder where model-output CSV tables are written.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=None,
        help="Number of school-cluster bootstrap replications. Overrides the config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed used for the school-cluster bootstrap. Overrides the config file.",
    )
    parser.add_argument(
        "--pre-periods",
        type=int,
        nargs="+",
        default=None,
        help="Event-time periods included in the joint pretrend test. Overrides the config file.",
    )
    parser.add_argument(
        "--post-periods",
        type=int,
        nargs="+",
        default=None,
        help="Event-time periods shown in the event-study table. Overrides the config file.",
    )
    parser.add_argument(
        "--exclude-years",
        type=int,
        nargs="*",
        default=None,
        help="Years excluded in the COVID-period sensitivity row. Overrides the config file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def setting(args: argparse.Namespace, config: dict[str, object], name: str):
    value = getattr(args, name)
    if value is not None:
        return value
    return config[name]


def normal_pvalue(estimate: float, std_error: float) -> float:
    if pd.isna(std_error) or std_error <= 0:
        return float("nan")
    return float(math.erfc(abs(estimate / std_error) / math.sqrt(2.0)))


def confidence_interval(estimate: float, std_error: float) -> tuple[float, float]:
    if pd.isna(std_error):
        return float("nan"), float("nan")
    return estimate - 1.96 * std_error, estimate + 1.96 * std_error


def make_model_sample(panel: pd.DataFrame, outcome_col: str, count_col: str | None = None) -> pd.DataFrame:
    columns = ["school_id", "year", "first_treat_year", "post", "event_time", outcome_col]
    if count_col is not None:
        columns.append(count_col)

    sample = panel[columns].copy()
    sample = sample.loc[sample[outcome_col].notna()].copy()
    sample["school_id"] = sample["school_id"].astype(str)
    sample["year"] = sample["year"].astype(int)
    sample["first_treat_year"] = sample["first_treat_year"].astype(int)
    sample["post"] = sample["post"].astype(int)
    sample["event_time"] = sample["event_time"].round().astype(int)
    return sample.sort_values(["school_id", "year"]).reset_index(drop=True)


def estimate_delta_att(
    treated_delta: pd.Series,
    control_delta: pd.Series,
    treated_base: pd.Series,
    control_base: pd.Series,
) -> float:
    """
    Estimate the cohort-year ATT as the difference between:

    1. the mean outcome change among treated schools; and
    2. the mean outcome change among not-yet-treated schools.

    The treated_base and control_base arguments are retained so that the
    function remains compatible with the existing calls. They are used only
    to ensure that both the baseline-year and comparison-year outcomes exist.
    """
    treated = pd.DataFrame(
        {
            "delta": treated_delta,
            "baseline": treated_base,
        }
    ).dropna()

    controls = pd.DataFrame(
        {
            "delta": control_delta,
            "baseline": control_base,
        }
    ).dropna()

    if treated.empty:
        raise ValueError("No complete treated-school changes are available.")

    if controls.empty:
        raise ValueError("No complete not-yet-treated-school changes are available.")

    return float(treated["delta"].mean() - controls["delta"].mean())


def estimate_delta_att_weighted(
    treated_delta: pd.Series,
    control_delta: pd.Series,
    treated_base: pd.Series,
    control_base: pd.Series,
    treated_weights: pd.Series,
    control_weights: pd.Series,
) -> float:
    """
    Estimate a pupil-count-weighted cohort-year ATT.

    The ATT is the weighted mean outcome change among treated schools minus
    the weighted mean outcome change among not-yet-treated schools.
    """
    treated = pd.DataFrame(
        {
            "delta": treated_delta,
            "baseline": treated_base,
            "weight": treated_weights,
        }
    ).dropna()

    controls = pd.DataFrame(
        {
            "delta": control_delta,
            "baseline": control_base,
            "weight": control_weights,
        }
    ).dropna()

    treated = treated.loc[treated["weight"] > 0].copy()
    controls = controls.loc[controls["weight"] > 0].copy()

    if treated.empty:
        raise ValueError("No weighted treated-school changes are available.")

    if controls.empty:
        raise ValueError(
            "No weighted not-yet-treated-school changes are available."
        )

    treated_mean = np.average(
        treated["delta"],
        weights=treated["weight"],
    )

    control_mean = np.average(
        controls["delta"],
        weights=controls["weight"],
    )

    return float(treated_mean - control_mean)


def cohort_time_att(sample: pd.DataFrame, outcome_col: str, weight_col: str | None = None) -> pd.DataFrame:
    records = []
    cohorts = sorted(sample["first_treat_year"].dropna().astype(int).unique())

    for cohort in cohorts:
        baseline_year = cohort - 1
        treated_schools = sample.loc[sample["first_treat_year"] == cohort, "school_id"].unique()

        for year in sorted(sample["year"].unique()):
            if year == baseline_year:
                continue

            control_schools = sample.loc[sample["first_treat_year"] > year, "school_id"].unique()
            if len(control_schools) == 0:
                continue

            columns = ["school_id", "year", outcome_col]
            if weight_col is not None:
                columns.append(weight_col)

            treated = sample.loc[
                sample["school_id"].isin(treated_schools) & sample["year"].isin([baseline_year, year]),
                columns,
            ]
            controls = sample.loc[
                sample["school_id"].isin(control_schools) & sample["year"].isin([baseline_year, year]),
                columns,
            ]

            treated_y = treated.pivot(index="school_id", columns="year", values=outcome_col)
            control_y = controls.pivot(index="school_id", columns="year", values=outcome_col)

            needed = [baseline_year, year]
            if any(col not in treated_y.columns for col in needed):
                continue
            if any(col not in control_y.columns for col in needed):
                continue

            treated_complete = treated_y[needed].notna().all(axis=1)
            control_complete = control_y[needed].notna().all(axis=1)

            if weight_col is None:
                if not treated_complete.any() or not control_complete.any():
                    continue
                treated_delta = treated_y.loc[treated_complete, year] - treated_y.loc[treated_complete, baseline_year]
                control_delta = control_y.loc[control_complete, year] - control_y.loc[control_complete, baseline_year]
                treated_base = treated_y.loc[treated_complete, baseline_year]
                control_base = control_y.loc[control_complete, baseline_year]
                att = estimate_delta_att(treated_delta, control_delta, treated_base, control_base)
                treated_weight_sum = float(treated_complete.sum())
            else:
                treated_w = treated.pivot(index="school_id", columns="year", values=weight_col)
                control_w = controls.pivot(index="school_id", columns="year", values=weight_col)

                if any(col not in treated_w.columns for col in needed):
                    continue
                if any(col not in control_w.columns for col in needed):
                    continue

                treated_complete = treated_complete & treated_w[needed].notna().all(axis=1)
                control_complete = control_complete & control_w[needed].notna().all(axis=1)
                if not treated_complete.any() or not control_complete.any():
                    continue

                treated_delta = treated_y.loc[treated_complete, year] - treated_y.loc[treated_complete, baseline_year]
                control_delta = control_y.loc[control_complete, year] - control_y.loc[control_complete, baseline_year]
                treated_base = treated_y.loc[treated_complete, baseline_year]
                control_base = control_y.loc[control_complete, baseline_year]
                treated_weights = (treated_w.loc[treated_complete, year] + treated_w.loc[treated_complete, baseline_year]) / 2
                control_weights = (control_w.loc[control_complete, year] + control_w.loc[control_complete, baseline_year]) / 2
                att = estimate_delta_att_weighted(
                    treated_delta,
                    control_delta,
                    treated_base,
                    control_base,
                    treated_weights,
                    control_weights,
                )
                treated_weight_sum = float(treated_weights.sum())

            records.append(
                {
                    "cohort": int(cohort),
                    "year": int(year),
                    "event_time": int(year - cohort),
                    "att_gt": float(att),
                    "n_treated_schools": int(treated_complete.sum()),
                    "n_control_schools": int(control_complete.sum()),
                    "treated_weight_sum": treated_weight_sum,
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError(f"No supported ATT cells for {outcome_col}.")
    return result.sort_values(["cohort", "year"]).reset_index(drop=True)


def aggregate_post_att(att_gt: pd.DataFrame, weight_col: str = "n_treated_schools") -> float:
    post = att_gt.loc[att_gt["event_time"] >= 0].copy()
    if post.empty:
        raise ValueError("No post-treatment ATT cells are supported.")
    return float(np.average(post["att_gt"], weights=post[weight_col]))


def aggregate_by_event_time(att_gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_time, group in att_gt.groupby("event_time", sort=True):
        rows.append(
            {
                "event_time": int(event_time),
                "att": float(np.average(group["att_gt"], weights=group["n_treated_schools"])),
                "treated_observations": int(group["n_treated_schools"].sum()),
                "not_yet_treated_observations": int(group["n_control_schools"].sum()),
                "schools_contributing": int(group["n_treated_schools"].sum()),
                "cohorts_represented": int(group["cohort"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)


def bootstrap_school_sample(sample: pd.DataFrame, rng: np.random.Generator, rep: int) -> pd.DataFrame:
    schools = sample["school_id"].drop_duplicates().to_numpy()
    sampled = rng.choice(schools, size=len(schools), replace=True)

    pieces = []
    for position, school_id in enumerate(sampled):
        piece = sample.loc[sample["school_id"] == school_id].copy()
        piece["school_id"] = f"{school_id}_boot_{rep}_{position}"
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def bootstrap_overall_att(
    sample: pd.DataFrame,
    outcome_col: str,
    reps: int,
    seed: int,
    weight_col: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for rep in range(reps):
        boot_sample = bootstrap_school_sample(sample, rng, rep)
        try:
            att_gt = cohort_time_att(boot_sample, outcome_col, weight_col)
            if weight_col is None:
                estimate = aggregate_post_att(att_gt)
            else:
                estimate = aggregate_post_att(att_gt, "treated_weight_sum")
            rows.append({"rep": rep, "estimate": float(estimate)})
        except Exception:
            continue

    return pd.DataFrame(rows)


def bootstrap_event_att(sample: pd.DataFrame, outcome_col: str, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for rep in range(reps):
        boot_sample = bootstrap_school_sample(sample, rng, rep)
        try:
            event_table = aggregate_by_event_time(cohort_time_att(boot_sample, outcome_col))
        except Exception:
            continue
        for row in event_table.itertuples(index=False):
            rows.append({"rep": rep, "event_time": int(row.event_time), "estimate": float(row.att)})

    return pd.DataFrame(rows)


def estimate_overall_att(
    sample: pd.DataFrame,
    outcome_col: str,
    reps: int,
    seed: int,
    weight_col: str | None = None,
) -> dict[str, float | pd.DataFrame]:
    att_gt = cohort_time_att(sample, outcome_col, weight_col)
    weight_name = "treated_weight_sum" if weight_col is not None else "n_treated_schools"
    estimate = aggregate_post_att(att_gt, weight_name)

    boot = bootstrap_overall_att(sample, outcome_col, reps, seed, weight_col)
    std_error = float(boot["estimate"].std(ddof=1)) if len(boot) > 1 else float("nan")
    ci_low, ci_high = confidence_interval(estimate, std_error)

    return {
        "estimate": float(estimate),
        "std_error": std_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": normal_pvalue(estimate, std_error),
        "att_gt": att_gt,
        "bootstrap": boot,
        "successful_bootstrap_reps": int(len(boot)),
    }


def estimate_twfe(sample: pd.DataFrame, outcome_col: str) -> dict[str, float]:
    formula = f"{outcome_col} ~ post + C(school_id) + C(year)"
    model = smf.ols(formula, data=sample).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["school_id"]},
    )
    ci_low, ci_high = model.conf_int().loc["post"]
    return {
        "estimate": float(model.params["post"]),
        "std_error": float(model.bse["post"]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(model.pvalues["post"]),
    }


def build_control_sample(panel: pd.DataFrame, outcome_col: str, controls: list[str]) -> pd.DataFrame:
    columns = ["school_id", "year", "first_treat_year", "post", "event_time", outcome_col] + controls
    sample = panel[columns].copy()
    sample = sample.loc[sample[outcome_col].notna()].dropna(subset=controls).copy()
    sample["school_id"] = sample["school_id"].astype(str)
    sample["year"] = sample["year"].astype(int)
    sample["first_treat_year"] = sample["first_treat_year"].astype(int)
    sample["post"] = sample["post"].astype(int)
    sample["event_time"] = sample["event_time"].round().astype(int)
    return sample.sort_values(["school_id", "year"]).reset_index(drop=True)


def residualize_outcome(
    sample: pd.DataFrame,
    outcome_col: str,
    controls: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Residualise the outcome on the selected covariates,
    school fixed effects, and year fixed effects.
    """
    work = sample.copy()

    used_controls = [
        col
        for col in controls
        if work[col].nunique(dropna=True) > 1
    ]

    formula_terms = used_controls + [
        "C(school_id)",
        "C(year)",
    ]

    formula = f"{outcome_col} ~ " + " + ".join(formula_terms)

    model = smf.ols(
        formula=formula,
        data=work,
    ).fit()

    work["_residual_outcome"] = model.resid

    return work, used_controls

def bootstrap_controlled_att(
    sample: pd.DataFrame,
    outcome_col: str,
    controls: list[str],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for rep in range(reps):
        boot_sample = bootstrap_school_sample(sample, rng, rep)
        try:
            residualized, _ = residualize_outcome(boot_sample, outcome_col, controls)
            att_gt = cohort_time_att(residualized, "_residual_outcome")
            rows.append({"rep": rep, "estimate": aggregate_post_att(att_gt)})
        except Exception:
            continue

    return pd.DataFrame(rows)


def estimate_controlled_att(
    panel: pd.DataFrame,
    outcome_col: str,
    controls: list[str],
    reps: int,
    seed: int,
) -> dict[str, float | int]:
    sample = build_control_sample(panel, outcome_col, controls)
    residualized, used_controls = residualize_outcome(sample, outcome_col, controls)

    att_gt = cohort_time_att(residualized, "_residual_outcome")
    estimate = aggregate_post_att(att_gt)

    boot = bootstrap_controlled_att(sample, outcome_col, used_controls, reps, seed)
    std_error = float(boot["estimate"].std(ddof=1)) if len(boot) > 1 else float("nan")
    ci_low, ci_high = confidence_interval(estimate, std_error)

    return {
        "observations": int(len(sample)),
        "schools": int(sample["school_id"].nunique()),
        "estimate": float(estimate),
        "std_error": std_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": normal_pvalue(estimate, std_error),
        "successful_bootstrap_reps": int(len(boot)),
    }


def cohort_specific_table(samples: dict[str, tuple[pd.DataFrame, str]], reps: int, seed: int) -> pd.DataFrame:
    rows = []

    for outcome_name, (sample, outcome_col) in samples.items():
        att_gt = cohort_time_att(sample, outcome_col)
        post_att = att_gt.loc[att_gt["event_time"] >= 0].copy()

        rng = np.random.default_rng(seed)
        boot_rows = []
        for rep in range(reps):
            boot_sample = bootstrap_school_sample(sample, rng, rep)
            try:
                boot_att = cohort_time_att(boot_sample, outcome_col)
                boot_post = boot_att.loc[boot_att["event_time"] >= 0].copy()
            except Exception:
                continue
            for cohort, group in boot_post.groupby("cohort"):
                estimate = np.average(group["att_gt"], weights=group["n_treated_schools"])
                boot_rows.append({"rep": rep, "cohort": int(cohort), "estimate": float(estimate)})
        boot = pd.DataFrame(boot_rows)

        for cohort in sorted(sample["first_treat_year"].dropna().astype(int).unique()):
            cohort_post = post_att.loc[post_att["cohort"] == cohort].copy()
            n_schools = int(sample.loc[sample["first_treat_year"] == cohort, "school_id"].nunique())

            if cohort_post.empty:
                rows.append(
                    {
                        "Outcome": outcome_name,
                        "Cohort": int(cohort),
                        "Obs.": int(sample.loc[sample["first_treat_year"] == cohort].shape[0]),
                        "Schools": n_schools,
                        "Estimate": np.nan,
                        "SE": np.nan,
                        "CI lower": np.nan,
                        "CI upper": np.nan,
                        "Support warning": "No post-treatment controls",
                    }
                )
                continue

            estimate = float(np.average(cohort_post["att_gt"], weights=cohort_post["n_treated_schools"]))
            draws = boot.loc[boot["cohort"] == cohort, "estimate"] if not boot.empty else pd.Series(dtype=float)
            std_error = float(draws.std(ddof=1)) if len(draws) > 1 else float("nan")
            ci_low, ci_high = confidence_interval(estimate, std_error)

            warning = "Supported"
            if len(cohort_post) == 1 or cohort_post["n_control_schools"].sum() <= 6:
                warning = "Very limited support"

            rows.append(
                {
                    "Outcome": outcome_name,
                    "Cohort": int(cohort),
                    "Obs.": int(cohort_post["n_treated_schools"].sum() + cohort_post["n_control_schools"].sum()),
                    "Schools": n_schools,
                    "Estimate": estimate,
                    "SE": std_error,
                    "CI lower": ci_low,
                    "CI upper": ci_high,
                    "Support warning": warning,
                }
            )

    return pd.DataFrame(rows)


def event_study_table(
    samples: dict[str, tuple[pd.DataFrame, str]],
    reps: int,
    seed: int,
    pre_periods: list[int],
    post_periods: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    joint_rows = []

    for outcome_name, (sample, outcome_col) in samples.items():
        att_gt = cohort_time_att(sample, outcome_col)
        event = aggregate_by_event_time(att_gt)
        boot = bootstrap_event_att(sample, outcome_col, reps, seed)

        if boot.empty:
            boot_stats = pd.DataFrame(columns=["event_time", "std_error", "successful_bootstrap_reps"])
        else:
            boot_stats = (
                boot.groupby("event_time")["estimate"]
                .agg(std_error=lambda s: s.std(ddof=1), successful_bootstrap_reps="size")
                .reset_index()
            )
        event = event.merge(boot_stats, on="event_time", how="left")
        event["ci_low"] = event["att"] - 1.96 * event["std_error"]
        event["ci_high"] = event["att"] + 1.96 * event["std_error"]
        event["p_value"] = [
            normal_pvalue(estimate, se) for estimate, se in zip(event["att"], event["std_error"])
        ]

        shown_periods = pre_periods + post_periods
        for event_time in shown_periods:
            row = event.loc[event["event_time"] == event_time]
            if row.empty:
                event_rows.append(
                    {
                        "Outcome": outcome_name,
                        "event_time": int(event_time),
                        "ATT": np.nan,
                        "Bootstrap SE": np.nan,
                        "CI lower": np.nan,
                        "CI upper": np.nan,
                        "p-value": np.nan,
                        "schools_contributing": 0,
                        "cohorts_represented": 0,
                        "successful_bootstrap_reps": 0,
                        "support_note": "No supported ATT cell",
                    }
                )
            else:
                row = row.iloc[0]
                note = ""
                if row["cohorts_represented"] <= 1:
                    note = "Single-cohort support"
                event_rows.append(
                    {
                        "Outcome": outcome_name,
                        "event_time": int(event_time),
                        "ATT": float(row["att"]),
                        "Bootstrap SE": float(row["std_error"]),
                        "CI lower": float(row["ci_low"]),
                        "CI upper": float(row["ci_high"]),
                        "p-value": float(row["p_value"]),
                        "schools_contributing": int(row["schools_contributing"]),
                        "cohorts_represented": int(row["cohorts_represented"]),
                        "successful_bootstrap_reps": int(row["successful_bootstrap_reps"]),
                        "support_note": note,
                    }
                )

        pre = event.loc[event["event_time"].isin(pre_periods)].set_index("event_time").sort_index()
        boot_pre = boot.loc[boot["event_time"].isin(pre_periods)].pivot(
            index="rep", columns="event_time", values="estimate"
        )
        boot_pre = boot_pre.reindex(columns=pre_periods).dropna()

        if len(pre) == len(pre_periods) and len(boot_pre) > len(pre_periods):
            coef = pre.reindex(pre_periods)["att"].to_numpy()
            covariance = np.cov(boot_pre.to_numpy(), rowvar=False, ddof=1)
            covariance_inv = np.linalg.pinv(covariance)
            wald = float(coef.T @ covariance_inv @ coef)
            f_stat = wald / len(pre_periods)
            p_value = float(stats.f.sf(f_stat, len(pre_periods), max(len(boot_pre) - len(pre_periods), 1)))

            largest_tau = int(pre["att"].abs().idxmax())
            largest_coef = float(pre.loc[largest_tau, "att"])
            largest_draws = boot_pre[largest_tau]
            centered_draws = largest_draws - largest_draws.mean()
            coef_p = float((np.abs(centered_draws) >= abs(largest_coef)).mean())
            largest_text = f"{largest_coef:.6f} (tau={largest_tau})"
        else:
            f_stat = float("nan")
            p_value = float("nan")
            largest_text = ""
            coef_p = float("nan")

        joint_rows.append(
            {
                "Outcome": f"{outcome_name}' mathematics grade",
                "Pre periods tested": len(pre_periods),
                "Joint F-statistic": f_stat,
                "Joint p-value": p_value,
                "Largest |pre-coef|": largest_text,
                "p-value of that coefficient": coef_p,
            }
        )

    return pd.DataFrame(event_rows), pd.DataFrame(joint_rows)


def baseline_support_table(samples: dict[str, tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    rows = []
    for outcome_name, (sample, _) in samples.items():
        rows.append(
            {
                "Outcome": f"{outcome_name}' mathematics grade",
                "Treated obs.": int(sample["post"].sum()),
                "Not-yet-treated obs.": int((sample["post"] == 0).sum()),
                "Cohorts": int(sample["first_treat_year"].nunique()),
                "Event-time min": int(sample["event_time"].min()),
                "Event-time max": int(sample["event_time"].max()),
            }
        )
    return pd.DataFrame(rows)


def treatment_support_by_year(panel: pd.DataFrame, samples: dict[str, tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    rows = []
    panels = {"full panel": panel[["school_id", "year", "post"]].copy()}
    for outcome_name, (sample, _) in samples.items():
        panels[f"{outcome_name.lower()} model-ready sample"] = sample[["school_id", "year", "post"]].copy()

    for sample_name, frame in panels.items():
        for year, group in frame.groupby("year", sort=True):
            rows.append(
                {
                    "sample": sample_name,
                    "year": int(year),
                    "number of untreated/not-yet-treated observations": int((group["post"] == 0).sum()),
                    "number of treated observations": int((group["post"] == 1).sum()),
                    "total observations": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def run_baseline_tables(samples: dict[str, tuple[pd.DataFrame, str]], reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for outcome_name, (sample, outcome_col) in samples.items():
        result = estimate_overall_att(sample, outcome_col, reps, seed)
        rows.append(
            {
                "outcome": f"{outcome_name}' mathematics grade",
                "estimator": "Cohort-aware ATT using not-yet-treated controls",
                "number of schools": int(sample["school_id"].nunique()),
                "number of observations": int(len(sample)),
                "ATT / coefficient": result["estimate"],
                "standard error": result["std_error"],
                "confidence interval lower bound": result["ci_low"],
                "confidence interval upper bound": result["ci_high"],
                "p-value, if available": result["p_value"],
                "successful bootstrap replications": result["successful_bootstrap_reps"],
            }
        )
    return pd.DataFrame(rows)


def controlled_table(
    panel: pd.DataFrame,
    controls: list[str],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for outcome_name, outcome_col in [("Boys", "boy_grade"), ("Girls", "girl_grade")]:
        result = estimate_controlled_att(panel, outcome_col, controls, reps, seed)
        rows.append(
            {
                "Outcome": outcome_name,
                "Obs.": result["observations"],
                "Schools": result["schools"],
                "Estimate": result["estimate"],
                "SE": result["std_error"],
                "95% CI lower": result["ci_low"],
                "95% CI upper": result["ci_high"],
                "p-value": result["p_value"],
                "successful bootstrap replications": result["successful_bootstrap_reps"],
            }
        )
    return pd.DataFrame(rows)


def twfe_table(samples: dict[str, tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    rows = []
    for outcome_name, (sample, outcome_col) in samples.items():
        result = estimate_twfe(sample, outcome_col)
        rows.append(
            {
                "Outcome": outcome_name,
                "Obs.": int(len(sample)),
                "Schools": int(sample["school_id"].nunique()),
                "beta treated x post": result["estimate"],
                "Cluster-robust SE": result["std_error"],
                "95% CI lower": result["ci_low"],
                "95% CI upper": result["ci_high"],
                "p-value": result["p_value"],
            }
        )
    return pd.DataFrame(rows)


def weighted_and_exclusion_sensitivity(
    samples: dict[str, tuple[pd.DataFrame, str]],
    count_columns: dict[str, str],
    reps: int,
    seed: int,
    exclude_years: list[int],
) -> pd.DataFrame:
    rows = []

    for outcome_name, (sample, outcome_col) in samples.items():
        result = estimate_overall_att(sample, outcome_col, reps, seed)
        rows.append(
            {
                "Outcome": f"{outcome_name}' mathematics grade",
                "Specification": "Unweighted baseline",
                "Obs.": int(len(sample)),
                "Estimate": result["estimate"],
                "SE": result["std_error"],
                "95% CI lower": result["ci_low"],
                "95% CI upper": result["ci_high"],
                "p": result["p_value"],
                "successful bootstrap replications": result["successful_bootstrap_reps"],
            }
        )

        weighted = estimate_overall_att(sample, outcome_col, reps, seed, count_columns[outcome_name])
        rows.append(
            {
                "Outcome": f"{outcome_name}' mathematics grade",
                "Specification": "Pupil-count weighted",
                "Obs.": int(len(sample)),
                "Estimate": weighted["estimate"],
                "SE": weighted["std_error"],
                "95% CI lower": weighted["ci_low"],
                "95% CI upper": weighted["ci_high"],
                "p": weighted["p_value"],
                "successful bootstrap replications": weighted["successful_bootstrap_reps"],
            }
        )

        restricted = sample.loc[~sample["year"].isin(exclude_years)].copy()
        restricted_result = estimate_overall_att(restricted, outcome_col, reps, seed)
        rows.append(
            {
                "Outcome": f"{outcome_name}' mathematics grade",
                "Specification": "Excluding years: " + ", ".join(str(year) for year in exclude_years),
                "Obs.": int(len(restricted)),
                "Estimate": restricted_result["estimate"],
                "SE": restricted_result["std_error"],
                "95% CI lower": restricted_result["ci_low"],
                "95% CI upper": restricted_result["ci_high"],
                "p": restricted_result["p_value"],
                "successful bootstrap replications": restricted_result["successful_bootstrap_reps"],
            }
        )

    return pd.DataFrame(rows)


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_panel = pd.read_csv(data_dir / "final_school_year_panel.csv")
    municipality_panel = pd.read_csv(data_dir / "final_school_year_panel_with_municipality_covariates.csv")
    teacher_panel = pd.read_csv(data_dir / "final_school_year_panel_with_teacher_covariates.csv")

    for frame in [final_panel, municipality_panel, teacher_panel]:
        frame["school_id"] = frame["school_id"].astype(str)
        frame["year"] = frame["year"].astype(int)
        frame["first_treat_year"] = frame["first_treat_year"].astype(int)
        frame["post"] = frame["post"].astype(int)
        frame["event_time"] = frame["event_time"].round().astype(int)

    return final_panel, municipality_panel, teacher_panel


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_reps = int(setting(args, config, "bootstrap_reps"))
    seed = int(setting(args, config, "seed"))
    pre_periods = [int(value) for value in setting(args, config, "pre_periods")]
    post_periods = [int(value) for value in setting(args, config, "post_periods")]
    exclude_years = [int(value) for value in setting(args, config, "exclude_years")]
    municipality_controls = [str(value) for value in config["municipality_controls"]]
    teacher_controls = [str(value) for value in config["teacher_controls"]]

    final_panel, municipality_panel, teacher_panel = load_inputs(args.data_dir)

    samples = {
        "Boys": (make_model_sample(final_panel, "boy_grade", "boy_count"), "boy_grade"),
        "Girls": (make_model_sample(final_panel, "girl_grade", "girl_count"), "girl_grade"),
    }
    count_columns = {"Boys": "boy_count", "Girls": "girl_count"}

    baseline = run_baseline_tables(samples, bootstrap_reps, seed)
    baseline.to_csv(args.output_dir / "table_7_5_baseline_att_estimates_by_outcome.csv", index=False)

    support = treatment_support_by_year(final_panel, samples)
    support.to_csv(args.output_dir / "table_7_5_treatment_support_for_baseline_estimation.csv", index=False)

    baseline_support_table(samples).to_csv(args.output_dir / "table_a6_1_baseline_support.csv", index=False)

    cohort_specific = cohort_specific_table(samples, bootstrap_reps, seed)
    cohort_specific.to_csv(args.output_dir / "table_a6_2_cohort_specific_att.csv", index=False)

    event_table, joint_pretrend = event_study_table(
        samples,
        bootstrap_reps,
        seed,
        pre_periods,
        post_periods,
    )
    event_table.to_csv(args.output_dir / "table_a6_3_event_study_full.csv", index=False)
    joint_pretrend.to_csv(args.output_dir / "table_6_2_joint_pretrends_test.csv", index=False)

    controlled_table(municipality_panel, municipality_controls, bootstrap_reps, seed).to_csv(
        args.output_dir / "table_a7_1_municipality_controlled_att.csv",
        index=False,
    )
    controlled_table(teacher_panel, teacher_controls, bootstrap_reps, seed).to_csv(
        args.output_dir / "table_a7_2_teacher_controlled_att.csv",
        index=False,
    )
    controlled_table(
        teacher_panel,
        municipality_controls + teacher_controls,
        bootstrap_reps,
        seed,
    ).to_csv(args.output_dir / "table_a7_3_combined_controlled_att.csv", index=False)

    twfe_table(samples).to_csv(args.output_dir / "table_a7_4_twfe_benchmark.csv", index=False)

    weighted_and_exclusion_sensitivity(
        samples,
        count_columns,
        bootstrap_reps,
        seed,
        exclude_years,
    ).to_csv(args.output_dir / "table_a7_5_weighted_and_covid_sensitivity.csv", index=False)

    print(f"Finished. Model tables were written to: {args.output_dir}")


if __name__ == "__main__":
    main()
