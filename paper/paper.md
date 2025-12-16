---
title: 'bca-survival: A Python package for body composition-based survival analysis in medical imaging research'
tags:
  - Python
  - survival analysis
  - body composition
  - CT imaging
  - oncology
  - medical imaging
authors:
  - name: Eric Frodl
    orcid: 0009-0002-7657-0344
    corresponding: true
    affiliation: 1
  - name: Benedikt Wichtlhuber
    corresponding: false
    affiliation: 1
  - name: Matthias Neitzel
    orcid: 0009-0008-5172-3512
    corresponding: false
    affiliation: 1
  - name: Andreas Michael Bucher
    orcid: 0009-0003-5012-161X
    corresponding: false
    affiliation: 1
affiliations:
  - name: Goethe University Frankfurt, University Hospital, Department of Radiology and Nuclear Medicine, Germany
    index: 1
    ror: 03f6n9m15
date: 16 December 2025
bibliography: paper.bib
---

# Summary

Body composition analysis (BCA) derived from computed tomography (CT) imaging has emerged as an important prognostic factor in oncology, geriatrics, and critical care medicine. Parameters such as skeletal muscle mass, visceral adipose tissue, and subcutaneous fat have been associated with treatment outcomes, survival, and quality of life across diverse patient populations. However, integrating CT-derived body composition metrics with clinical survival data requires substantial data processing and statistical expertise, creating barriers for clinical researchers.

`bca-survival` is an open-source Python package that streamlines the analysis pipeline from CT-derived body composition measurements to survival analysis results. The package provides a unified interface for merging clinical data with body composition measurements, performing Cox proportional hazards regression (both univariate and multivariate), generating Kaplan-Meier survival curves, and exporting results in multiple formats. It is specifically designed to integrate with output from automated segmentation algorithms such as the Body and Organ Analysis (BOA) algorithm [@Haubold:2024] and TotalSegmentator [@Wasserthal:2023], both built on the nnU-Net framework [@Isensee:2021].

# Statement of need

The relationship between body composition and clinical outcomes is an active area of research in oncology, geriatrics, and critical care medicine [@Keyl:2024; @Caan:2018]. Sarcopenia (low muscle mass) and changes in adipose tissue distribution have been linked to chemotherapy toxicity, surgical complications, and overall survival in various cancer types [@Kazemi-Bajestani:2016; @Shachar:2016]. Modern deep learning algorithms can automatically segment body compartments from CT scans, producing detailed measurements of skeletal muscle, visceral adipose tissue, and subcutaneous adipose tissue at scale [@Wasserthal:2023; @Haubold:2024].

Despite the availability of automated segmentation tools, translating these measurements into meaningful survival analyses remains challenging. Researchers must address several interconnected tasks: data integration between imaging measurements and clinical databases, time-to-event calculation from diagnosis or treatment dates, statistical modeling with appropriate handling of covariates and confounders, and results dissemination in formats suitable for publications. While general-purpose survival analysis packages exist, such as `lifelines` [@Davidson-Pilon:2019] for Python and the `survival` package for R [@Therneau:2000], these tools require researchers to implement custom pipelines for the specific challenges of body composition research.

`bca-survival` addresses this gap by providing a domain-specific toolkit that handles the entire workflow from raw segmentation output to publication-ready results. The package reduces the technical barrier for clinicians and researchers investigating body composition as a prognostic factor, enabling larger-scale studies and facilitating reproducible research.

# Functionality

The core functionality centers around the `BCASurvivalAnalyzer` class, which accepts clinical and measurement dataframes and provides methods for comprehensive survival analysis:

```python
from bca_survival import BCASurvivalAnalyzer
import pandas as pd

# Load clinical and body composition data
clinical_data = pd.read_csv('clinical_data.csv')
bca_measurements = pd.read_csv('bca_measurements.csv')

# Initialize analyzer with patient matching
analyzer = BCASurvivalAnalyzer(
    df_main=clinical_data,
    df_measurements=bca_measurements,
    main_id_col='patient_id',
    measurement_id_col='id',
    start_date_col='diagnosis_date',
    event_date_col='death_date',
    event_col='event_status'
)

# Univariate Cox regression for body composition variables
results = analyzer.univariate_cox_regression([
    'l5::WL::imat::mean_ml',
    'l5::WL::tat::mean_ml',
    'l5::WL::muscle::mean_ml'
])

# Kaplan-Meier curves with configurable stratification
analyzer.kaplan_meier_plot(
    'l5::WL::muscle::mean_ml',
    split_strategy='median'
)

# Multivariate analysis with automatic multicollinearity handling
model = analyzer.multivariate_cox_regression([
    'l5::WL::muscle::mean_ml',
    'age',
    'tumor_stage'
])
```

Key features of the package include automatic patient matching between clinical and imaging datasets with handling of missing measurements, flexible stratification strategies for survival curves (median, mean, percentile, or fixed cutoffs), variance inflation factor (VIF)-based multicollinearity detection and variable removal, and support for correction variables in univariate analyses to adjust for confounders.

The package also includes command-line utilities that integrate with existing clinical workflows. The `boa-extract` tool extracts measurements from BOA algorithm output directories, converting hierarchical folder structures into analysis-ready tabular format. The `bca-merge` tool combines clinical Excel files with body composition measurements based on patient identifiers. The `survival-result-converter` tool exports analysis results to PDF, CSV, and TXT formats for reporting, and `pdf-report-extractor` provides secure PDF encryption for patient data distribution.

# Implementation

`bca-survival` is built on the established scientific Python ecosystem. Survival analysis functionality relies on `lifelines` [@Davidson-Pilon:2019], a well-validated survival analysis library implementing Cox proportional hazards models and Kaplan-Meier estimation. Data manipulation uses `pandas` [@McKinney:2010] and `numpy` [@Harris:2020], while visualizations are created with `matplotlib` [@Hunter:2007] and `seaborn` [@Waskom:2021]. Statistical modeling is supplemented by `statsmodels` [@Seabold:2010] for variance inflation factor calculations and `scikit-learn` [@Pedregosa:2011] for data standardization.

The package follows modern Python packaging standards using `pyproject.toml` configuration and `setuptools_scm` for version management from git tags. Comprehensive type hints support static analysis with `mypy`. The test suite uses `pytest` with coverage reporting, and code quality is enforced through `black`, `isort`, and `flake8`. Continuous integration via GitHub Actions runs tests across Python 3.9-3.11 on Linux, macOS, and Windows. Documentation is generated using Sphinx with autodoc support and is automatically deployed to GitHub Pages. The package is distributed via PyPI, enabling installation with `pip install bca-survival`.

# Research applications

`bca-survival` is designed to support research investigating the prognostic value of body composition in clinical outcomes. Target applications include studies of sarcopenia as a predictor of survival in cancer patients, longitudinal analysis of body composition changes during chemotherapy or immunotherapy, investigation of adipose tissue distribution as a marker of treatment toxicity, and multi-site studies requiring standardized, reproducible analysis pipelines.

The package has been developed to support body composition research at University Hospital Frankfurt and is intended for use by the broader medical imaging and oncology research communities. By lowering the technical barrier to survival analysis, `bca-survival` aims to enable more researchers to investigate the clinical significance of CT-derived body composition metrics.

# Acknowledgements

We acknowledge the developers of the Body and Organ Analysis (BOA) algorithm and TotalSegmentator for making automated body composition segmentation accessible to the research community. This project was funded by "NUM 3.0" (FKZ: 01KX2524)

# Author Contributions

**Eric Frodl**: Conceptualization, software development, documentation, testing, deployment, manuscript writing - original draft, review & editing.

**Benedikt Wichtlhuber**: Software development, testing, manuscript review .

**Matthias Neitzel**: Field testing, user feedback, bug fixing, documentation.

**Andreas Michael Bucher**: Domain expertise, requirements specification, medical/clinical validation, supervision, manuscript review.


# References
