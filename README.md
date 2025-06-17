[![pipeline status](https://fdm-git.diz-ag.med.ovgu.de/ukf-racoon-gruppe/bca-survival-analyzer/badges/main/pipeline.svg)](https://fdm-git.diz-ag.med.ovgu.de/ukf-racoon-gruppe/bca-survival-analyzer/-/commits/main)
# Survival Analysis Package

A Python package for analyzing survival data with a focus on body composition assessment.

## Features

- **Survival Analysis**: Cox proportional hazards regression and Kaplan-Meier survival curves
- **Body Composition Analysis**: Tools for processing and analyzing BCA data
- **BOA Extractor**: Command-line tool for extracting measurements from BOA data
- **Data Preprocessing**: Utilities for cleaning and preparing survival data

## Installation

Check the [Package Registery](https://fdm-git.diz-ag.med.ovgu.de/ukf-racoon-gruppe/bca-survival-analyzer/-/packages) for installation details:

## Usage

### Basic Survival Analysis

```python
from survival_analysis.analyzer import BCASurvivalAnalyzer

# Load your data
df_main = pd.read_csv('clinical_data.csv')
df_measurements = pd.read_csv('bca_measurements.csv')

# Initialize the analyzer
analyzer = BCASurvivalAnalyzer(
    df_main, df_measurements,
    main_id_col='patient_id', measurement_id_col='id',
    start_date_col='diagnosis_date', event_date_col='event_date', event_col='event_status'
)

# Perform univariate analysis
columns = ['l5::WL::imat::mean_ml', 'l5::WL::tat::mean_ml', 'age', 'gender']
results = analyzer.univariate_cox_regression(columns)

# Generate Kaplan-Meier plot
analyzer.kaplan_meier_plot('l5::WL::imat::mean_ml', split_strategy='median')

# Perform multivariate analysis
model = analyzer.multivariate_cox_regression(columns)
```

### BOA Extractor

Extract measurements from BOA data using the command-line tool:

```bash
boa-extract /path/to/data /path/to/output
```

## Documentation

Refer to the documentation in the `docs/` directory for detailed information:

1. Install the package with documentation dependencies:
   ```bash
   pip install -e ".[docs]"
   ```

2. Build the documentation on Windows:
   ```bash
   cd docs
   make.bat html
   ```
   
   Or on Linux/macOS:
   ```bash
   cd docs
   make html
   ```

3. Open `docs/build/html/index.html` in your browser

## Development

Clone the repository and install in development mode:

```bash
git clone https://gitlab.com/your-group/survival-analysis.git
cd survival-analysis
pip install -e ".[dev]"
```

