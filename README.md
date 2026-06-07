# Master Thesis Modelling Package

This folder contains the clean data and Python program needed to reproduce the modelling tables used in the thesis.

The package starts from already prepared CSV files. It does not collect data from the web, parse Excel workbooks, create figures, edit Word documents, or write thesis text.

## Folder Contents

```text
master_thesis_submission/
  code/
    run_master_thesis_models.py
  config/
    model_config.json
  data/
    final_school_year_panel.csv
    final_school_year_panel_with_municipality_covariates.csv
    final_school_year_panel_with_teacher_covariates.csv
    municipality_covariates_merged.csv
    teacher_covariates_official_county_year.csv
  outputs/
    tables/
  requirements.txt
  submission_manifest.csv
```

## What the Program Does

The program estimates the thesis models from the clean school-year panel:

1. Baseline cohort-aware ATT using not-yet-treated schools as controls.
2. Treatment-support table for the baseline estimation sample.
3. Cohort-specific ATT estimates.
4. Event-study estimates.
5. Joint pretrend Wald test for the configured pre-treatment event periods.
6. Municipality-control robustness.
7. Teacher-control robustness.
8. Combined municipality and teacher-control robustness.
9. Two-way fixed-effects benchmark.
10. Pupil-count weighting and exclusion-year sensitivity.

All outputs are saved as CSV files in `outputs/tables/`.

## How to Run

From this folder:

```bash
python code/run_master_thesis_models.py
```

The default run uses the settings in `config/model_config.json`, including `999` school-cluster bootstrap replications.

For a quick test run:

```bash
python code/run_master_thesis_models.py --bootstrap-reps 20 --output-dir outputs/test_tables
```

For the thesis run:

```bash
python code/run_master_thesis_models.py --bootstrap-reps 999 --output-dir outputs/tables
```

## Python Requirements

Install the packages listed in `requirements.txt` if they are not already available:

```bash
pip install -r requirements.txt
```

The code was written to use relative paths. It should run after moving the whole `master_thesis_submission` folder to another machine, as long as the folder structure is kept the same.

## Model Settings

The file `config/model_config.json` contains the thesis modelling settings:

- number of bootstrap replications;
- random seed;
- event-time periods used for the pretrend test;
- event-time periods shown in the event-study table;
- year excluded in the COVID-period sensitivity row;
- municipality control variables;
- teacher control variables.

These settings can be changed in the config file. Some can also be overridden from the command line, for example:

```bash
python code/run_master_thesis_models.py --bootstrap-reps 499 --seed 12345
```

## Output Tables

The main output files are:

```text
table_6_2_joint_pretrends_test.csv
table_7_5_baseline_att_estimates_by_outcome.csv
table_7_5_treatment_support_for_baseline_estimation.csv
table_a6_1_baseline_support.csv
table_a6_2_cohort_specific_att.csv
table_a6_3_event_study_full.csv
table_a7_1_municipality_controlled_att.csv
table_a7_2_teacher_controlled_att.csv
table_a7_3_combined_controlled_att.csv
table_a7_4_twfe_benchmark.csv
table_a7_5_weighted_and_covid_sensitivity.csv
```

The bootstrap columns report how many bootstrap draws were successfully estimated. This is useful because some resampled school panels can lose the support needed for a cohort-aware ATT cell.

## Important Notes

The program uses the thesis identification strategy: cohort-aware difference-in-differences with not-yet-treated schools as controls. The robustness models with controls first residualise the outcome on the listed controls and then apply the same cohort-aware ATT estimator.

The controlled robustness tables are support-limited because the municipality and teacher covariates are not available for the full 2007-2021 panel. This is a data-support issue, not a plotting or formatting issue.

