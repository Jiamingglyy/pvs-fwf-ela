"""Core analyses for 5.0 T migraine PVS–FWF–ELA."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

GROUPS = ["HC", "EM_LF", "EM_HF", "CM"]
X, M, Y = "pvs_frontal_solidity", "fwf_frontal_matched", "ela_rigidity"
SEED = 20261116


def coefficients(arr):
    x, m, y = arr[:,0], arr[:,1], arr[:,2]
    cov = arr[:,3:]
    xm = np.column_stack([cov,x])
    a = np.linalg.lstsq(xm,m,rcond=None)[0][-1]
    direct,b = np.linalg.lstsq(np.column_stack([xm,m]),y,rcond=None)[0][-2:]
    total = np.linalg.lstsq(xm,y,rcond=None)[0][-1]
    assert np.isclose(total, direct+a*b)
    return np.array([a,b,a*b,direct,total])


def design(d):
    return np.column_stack([d[[X, M, Y]], np.ones(len(d)),
                            *[d.group.eq(g).astype(float) for g in GROUPS[1:] if d.group.eq(g).any()],
                            d[["age", "sex"]]])


def path_boot(d, label):
    b = 5000
    arr = design(d)
    obs = coefficients(arr)
    strata = [np.flatnonzero(d.group.eq(g)) for g in GROUPS if d.group.eq(g).any()]
    rng = np.random.default_rng(SEED)
    draws = np.array([coefficients(arr[np.concatenate([rng.choice(s,len(s)) for s in strata])])
                      for _ in range(b)])
    for k,effect in enumerate(['a','b','indirect','direct','total']):
        lo,hi=np.quantile(draws[:,k],[.025,.975])
        extreme = min(np.sum(draws[:,k]<=0),np.sum(draws[:,k]>=0))
        yield dict(model=label,outcome=Y,effect=effect,n=len(d),estimate=obs[k],lo=lo,hi=hi,
                   p=min(1.,2*(extreme+1)/(b+1)),floor=bool(extreme==0),B=b)


def association(frame: pd.DataFrame, model_name: str, outcome: str,
                predictors: list[str], order: list[str],
                patient_only: bool = False) -> list[dict]:
    columns = [outcome, *predictors, "group", "age", "sex"]
    data = frame[columns].copy()
    if patient_only:
        data = data[data["group"].astype(str) != "HC"]
        categories = order[1:]
        reference = "EM_LF"
    else:
        categories = order
        reference = "HC"
    for column in [outcome, *predictors, "age", "sex"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    data["group"] = pd.Categorical(data["group"], categories=categories, ordered=True)
    formula = f"{outcome} ~ " + " + ".join(
        predictors + [f"C(group, Treatment(reference='{reference}'))", "age", "sex"])
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    return [{
        "model": model_name,
        "sample": "patients_only" if patient_only else "all_groups",
        "outcome": outcome, "predictor": predictor, "n": int(model.nobs),
        "beta": float(model.params[predictor]), "HC3_se": float(model.bse[predictor]),
        "HC3_t": float(model.tvalues[predictor]), "p": float(model.pvalues[predictor]),
        "coefficient_scale": "raw original units; no z transformation",
    } for predictor in predictors]


def table_three(joint):
    specs = [
        ("frontal_PVS_to_FWF", M, [X], False),
        ("frontal_FWF_to_ELA", Y, [M], False),
        ("frontal_PVS_to_ELA", Y, [X], False),
        ("frontal_joint_model", Y, [X, M], False),
        ("wholebrain_PVS_to_FWF", "wb_FWF", ["wb_solidity"], False),
        ("wholebrain_joint_model", Y, ["wb_solidity", "wb_FWF"], False),
        ("frontal_joint_model", Y, [X, M], True),
    ]
    tests = [
        row
        for model, outcome, predictors, patient_only in specs
        for row in association(joint, model, outcome, predictors, GROUPS, patient_only)
    ]
    result = pd.DataFrame(tests)
    result["q_bh"] = multipletests(result.p, method="fdr_bh")[1]
    return result


def permutation_t(data: pd.DataFrame, outcome: str, tested: np.ndarray,
                  permutations: int, rng: np.random.Generator) -> tuple[float, float, float]:
    y = data[outcome].to_numpy(dtype=float)
    age = data["age"].to_numpy(dtype=float)
    sex = data["sex"].to_numpy(dtype=float)
    reduced = np.column_stack([np.ones(len(y)), age, sex])
    full = np.column_stack([reduced, tested])
    reduced_beta = np.linalg.lstsq(reduced, y, rcond=None)[0]
    fitted = reduced @ reduced_beta
    residuals = y - fitted
    inverse = np.linalg.pinv(full.T @ full)

    def coefficient_t(response: np.ndarray) -> tuple[float, float]:
        beta = inverse @ full.T @ response
        errors = response - full @ beta
        variance = float(errors @ errors / (len(response) - full.shape[1]))
        se = float(np.sqrt(max(inverse[-1, -1] * variance, 0)))
        return float(beta[-1]), float(beta[-1] / se)

    estimate, observed = coefficient_t(y)
    extreme = 0
    for _ in range(permutations):
        _, permuted = coefficient_t(fitted + rng.permutation(residuals))
        extreme += int(abs(permuted) >= abs(observed))
    return estimate, observed, (extreme + 1) / (permutations + 1)


PVS_FAMILY = ["wb_solidity", X, "pvs_strict_frontal_solidity"]
FWF_FAMILY = ["wb_FWF", M, "fwf_strict_frontal"]
ELA_FAMILY = [
    "unique_state_count", "major_dwell_mean", "basin_dwell_mean",
    "direct_transition_frequency", "indirect_transition_frequency", "transition_entropy",
    "mean_energy", "energy_sd", Y,
]
PAIRS = list(combinations(GROUPS, 2))


def load_table(path, outcomes):
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    table = pd.read_csv(path, sep=sep, dtype={"subject_id": "string"})
    if table.subject_id.isna().any() or not table.subject_id.is_unique:
        raise ValueError("A table contains missing or repeated participant keys.")
    if set(table.group) != set(GROUPS):
        raise ValueError("Expected all four diagnostic groups.")
    for col in ["age", "sex", *outcomes]:
        if not np.isfinite(table[col].to_numpy(float)).all():
            raise ValueError(f"Column {col} must contain complete numeric values.")
        table[col] = table[col].astype(float)
    table["severity"] = table.group.map({g: i for i, g in enumerate(GROUPS)}).astype(float)
    return table.set_index("subject_id")


def descriptive(table, outcome):
    summaries = table.groupby("group")[outcome].agg(n="size", mean="mean", sd="std").reindex(GROUPS)
    return summaries.reset_index().assign(outcome=outcome)


def diffusion_or_pvs(table, family, primary):
    """Pairwise HC3 contrasts from the single adjusted categorical model."""
    model = smf.ols(f"{primary} ~ C(group, Treatment(reference='HC')) + age + sex", table).fit(cov_type="HC3")
    names = model.params.index.tolist()
    rows = []
    for reference, comparison in PAIRS:
        weights = np.zeros(len(names))
        for level, sign in [(reference, -1), (comparison, 1)]:
            if level != "HC":
                key = next(k for k in names if k.endswith(f"[T.{level}]"))
                weights[names.index(key)] = sign
        result = model.t_test(weights)
        lo, hi = result.conf_int()[0]
        rows.append(dict(outcome=primary, contrast=f"{comparison}-{reference}", n=int(model.nobs),
                         estimate=result.effect.item(), lo=lo, hi=hi, p=result.pvalue.item()))
    comparisons = pd.DataFrame(rows)
    comparisons["q"] = multipletests(comparisons.p, method="fdr_bh")[1]

    trends = []
    for outcome in family:
        fit = smf.ols(f"{outcome} ~ severity + age + sex", table).fit(cov_type="HC3")
        lo, hi = fit.conf_int().loc["severity"]
        trends.append(dict(outcome=outcome, contrast="ordinal", n=int(fit.nobs),
                           estimate=fit.params["severity"], lo=lo, hi=hi, p=fit.pvalues["severity"]))
    trends = pd.DataFrame(trends)
    trends["q"] = multipletests(trends.p, method="fdr_bh")[1]
    return pd.concat([comparisons, trends.loc[trends.outcome.eq(primary)]], ignore_index=True)


def rigidity_groups(ela):
    """Rigidity from the designated ELA export, with age/sex Freedman–Lane tests."""
    rows = []
    for i, (reference, comparison) in enumerate(PAIRS):
        subset = ela.loc[ela.group.isin([reference, comparison])].copy()
        subset["tested"] = subset.group.eq(comparison).astype(float)
        rng = np.random.default_rng(20260728 + ELA_FAMILY.index(Y) * 100 + i)
        beta, _, p = permutation_t(subset, Y, subset.tested.to_numpy(), 10000, rng)
        fit = smf.ols(f"{Y} ~ tested + age + sex", subset).fit(cov_type="HC3")
        lo, hi = fit.conf_int().loc["tested"]
        rows.append(dict(outcome=Y, contrast=f"{comparison}-{reference}", n=len(subset),
                         estimate=beta, lo=lo, hi=hi, p=p))
    pairs = pd.DataFrame(rows)
    pairs["q"] = multipletests(pairs.p, method="fdr_bh")[1]

    # All nine ordinal p values are needed for the original BH family.
    trend = []
    x = ela.severity.to_numpy()
    for i, outcome in enumerate(ELA_FAMILY):
        rng = np.random.default_rng(20260728 + i * 100 + 50)
        beta, _, p = permutation_t(ela, outcome, x, 10000, rng)
        trend.append(dict(outcome=outcome, contrast="ordinal", n=len(ela), estimate=beta, p=p))
    trend = pd.DataFrame(trend)
    trend["q"] = multipletests(trend.p, method="fdr_bh")[1]
    fit = smf.ols(f"{Y} ~ severity + age + sex", ela).fit(cov_type="HC3")
    trend.loc[trend.outcome.eq(Y), ["lo", "hi"]] = fit.conf_int().loc["severity"].to_numpy()
    return pd.concat([pairs, trend.loc[trend.outcome.eq(Y)]], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["pvs", "fwf", "ela", "joint", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    pvs = load_table(args.pvs, PVS_FAMILY)
    fwf = load_table(args.fwf, FWF_FAMILY)
    ela = load_table(args.ela, ELA_FAMILY)
    joint = load_table(args.joint, [X, M, Y, "wb_solidity", "wb_FWF"])
    for source in (pvs, fwf, ela):
        if not joint.index.isin(source.index).all() or not joint[["group", "age", "sex"]].equals(
            source.loc[joint.index, ["group", "age", "sex"]]
        ):
            raise ValueError("Complete-case IDs or demographic covariates disagree across tables.")
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = pd.concat([descriptive(pvs, X), descriptive(fwf, M), descriptive(ela, Y)])
    group_models = pd.concat([
        diffusion_or_pvs(pvs, PVS_FAMILY, X),
        diffusion_or_pvs(fwf, FWF_FAMILY, M),
        rigidity_groups(ela),
    ])
    summaries.to_csv(args.out / "table2_descriptives.tsv", sep="\t", index=False)
    group_models.to_csv(args.out / "table2_group_models.tsv", sep="\t", index=False)
    table_three(joint).to_csv(args.out / "table3_hc3_bh.tsv", sep="\t", index=False)
    pd.DataFrame(path_boot(joint, "primary_frontal")).to_csv(
        args.out / "figure5d_primary_paths.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
