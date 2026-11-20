# 5.0 T migraine PVS–FWF–ELA

Core statistical analyses for a 5.0 T MRI study of frontal perivascular-space (PVS) solidity, NODDI isotropic fraction (FWF), and energy-landscape (ELA) rigidity across migraine groups.

## Run

Use Python 3.12 and install the analysis dependencies.

```sh
python -m pip install -r requirements.txt
python analysis.py --pvs data/pvs.tsv --fwf data/fwf.tsv --ela data/ela.tsv --joint data/joint.tsv --out summaries
```

## Inputs

Supply authorized CSV or TSV exports with one row per participant. All four tables use the same pseudonymous `subject_id` and include `group`, `age`, and numeric `sex`. Group values are `HC`, `EM_LF`, `EM_HF`, and `CM`. Preserve the archived row order for the seeded resampling.

| Input | Other required columns |
|---|---|
| PVS | `pvs_frontal_solidity`, `pvs_strict_frontal_solidity`, `wb_solidity` |
| FWF | `fwf_frontal_matched`, `fwf_strict_frontal`, `wb_FWF` |
| ELA | `ela_rigidity`, `unique_state_count`, `major_dwell_mean`, `basin_dwell_mean`, `direct_transition_frequency`, `indirect_transition_frequency`, `transition_entropy`, `mean_energy`, `energy_sd` |
| Joint | `pvs_frontal_solidity`, `fwf_frontal_matched`, `ela_rigidity`, `wb_solidity`, `wb_FWF` |

Use the designated corrected complete-case export for the joint analysis and the corresponding unimodal ELA export for group comparisons. The models use the rigidity scores supplied in those exports. Source-level imaging reconstruction and score provenance are outside the scope of these scripts.

## Outputs

| File | Analysis |
|---|---|
| `table2_descriptives.tsv` | Group sample sizes, means, and standard deviations for the three primary measures. |
| `table2_group_models.tsv` | Age- and sex-adjusted group contrasts and ordinal trends. PVS and FWF use HC3 regression. ELA rigidity uses 10,000 Freedman–Lane permutations and HC3 intervals. |
| `table3_hc3_bh.tsv` | Cross-modal HC3 regressions adjusting for diagnostic group, age, and sex. BH correction covers the full 10-test family. |
| `figure5d_primary_paths.tsv` | The primary frontal coefficient products and 95% percentile intervals from 5,000 group-stratified bootstrap draws (seed 20261116). |

The repository contains code and documentation. Participant-level inputs are managed separately under the study’s consent terms. All output tables contain aggregate results.
