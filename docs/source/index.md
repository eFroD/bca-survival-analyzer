# BCA Survival Analysis Package

A Python package for survival analysis with body composition assessment (BCA) data.

## Overview

This package provides tools for conducting survival analysis on medical data, with specialized support for the [BOA - Body and Organ Analysis](https://github.com/UMEssen/Body-and-Organ-Analysis). It includes implementations of Cox proportional hazards models, Kaplan-Meier survival curves, and utilities for processing BCA data.

## Features

- **Cox Proportional Hazards Regression**: Analyze the effect of multiple variables on survival time
- **Kaplan-Meier Survival Curves**: Visualize and compare survival functions
- **Body Composition Analysis**: Process and integrate BCA data from various sources
- **BOA Results Extraction**: Extract data from BOA analysis results
- **Data Preprocessing**: Comprehensive tools for cleaning and preparing survival data


## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: API Reference

api/modules

```

## Installation

```bash
pip install bca-survival
```

For development installation:

```bash
git clone https://github.com/eFroD/bca-survival-analyzer.git
cd bca-survival-analyzer
pip install -e ".[dev,docs]"
```

## Quickstart

### Preparing Body Composition Data
Suppose you have a folder of results obtained from the Body-and-Organ analysis. It will consist of one folder per series, each with a `json` file containing the body composition measurements.
In order to aggregate them into a single table, with one patient per row, you can use the `boa-extract` command-line tool:

```bash
boa-extract /path/to/boa-results /path/to/output
```
This will create two files: 
- **bca-measurements.xlsx**: A table with the body-composition values. Important columns include:
  - **StudyID**: the ID to match the patient with the clinical data table. Should be a pseudonym.
  - Body composition values of the following form: `bodypart::ALL/WL::compatiment::statistic`. Body Part follows the terminology of the original algorithm. It starts with the whole scan, over regions like `abdominal cavity` to single slice measurements at vertebrae level. The second value is either `ALL`, meaning limbs are included or `WL` which stands for _without limbs_. The third is either `muscle`, `bone`, `imat`, `tat`, `eat`,`sat`,`vat`, following the definitions of the [Body Composition Analysis](https://pubmed.ncbi.nlm.nih.gov/32945971/)
- **total-measurements.xlsx**: Contains a similar structure but with organ measurements obtained from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

If you have the table in place, you can continue the survival analysis. The tool will take care of the merging and a big part of preprocessing for you.

### Prepare your Patient Data
`bca_survival` comes with some useful tools for data preprocessing. Check the API doc for details but here are a few:
````python
from bca_survival.preprocessing import compute_ratios
from bca_survival.utils import calculate_age
import pandas as pd

# loading a patient table
patient_list = pd.read_excel("../data/patient_lists/patients.xlsx")

# loading the previously extracted bca table
bca_values = pd.read_excel("../data/bca-measurements.xlsx")

# we want to automatically add ratio columns to extract new values like 
# muscle to bone ratio, which is an interesting biomarker
bca_values = compute_ratios(bca_values)

# let's create a patient age column called "Age"
patient_list = calculate_age(df=patient_list, birth_date_col="date_of_birth",
                             current_date_col="date_of_diagnosis")
````

### Perform the Survival Analysis.
Continuing the example above, let's use the data to perform a survival analysis by fitting a Cox-Regression
```python
from bca_survival import BCASurvivalAnalyzer
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Initialize analyzer

analyzer = BCASurvivalAnalyzer(
    df_main=patient_list, # the patient list with clinical data
    df_measurements=bca_values, # the extracted bca values
    main_id_col="ID", # id column for patients
    measurement_id_col="StudyID", # id column for the bca table
    start_date_col="Date_Diagnosis", # start date for survival analysis
    event_date_col="event_date", # event date e.g. follow-up date
    event_col="event" # indication whether the event occurred (0 or 1) 
)


# Fit Cox model
# Each provided column will be analzed with a univariate analysis
results = analyzer.univariate_cox_regression(
    columns=['age', 'bmi', 'abdominal_cavity::WL::muscle/bone::sum_ml'] 
)

# Generate Kaplan-Meier curves, split higher and lower median.
analyzer.kaplan_meier_plot(column='abdominal_cavity::WL::muscle/bone::sum_ml', split_strategy='median', output_path='KM-Plots')
```



## Indices and tables

* {ref}`genindex`
* {ref}`modindex`


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/eFroD/bca-survival-analyzer/issues)
- **Documentation**: [https://eFroD.github.io/bca-survival-analyzer/](https://eFroD.github.io/bca-survival-analyzer/)
- **Repository**: [https://github.com/eFroD/bca-survival-analyzer](https://github.com/eFroD/bca-survival-analyzer)
