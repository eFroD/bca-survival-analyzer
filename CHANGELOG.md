# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),

## [Unreleased]

### Added
- JOSS paper submission files (`paper/paper.md`, `paper/paper.bib`)
- CITATION.cff for proper citation metadata
- Codecov integration for test coverage reporting
- New test suite for CLI tools

### Changed
- Improved Windows compatibility with excel-file locking handling

## [0.2.6] - 2025-12-09

### Added
- PDF report encryption tool (`pdf-report-extractor`)
- Survival results converter with multi-format export (PDF, CSV, TXT)

### Fixed
- File handle cleanup on Windows to prevent PermissionError

## [0.2.4] - 2025-12-02

### Added
- BCA merger tool for combining clinical and measurement data
- Support for correction variables in univariate Cox regression

### Changed
- Improved date format handling in preprocessing functions
- Added more customization functionality for creating Plots
  - Add custom title
  - define the image quality
  - Custom names for stratification
- Numpy dependency is now >= 1.26.4

### Fixed
- Improved stability when dealing with non-numeric columns if using standardize

## [0.2.3] - 2025-05-02

### Added

- c-index, log_likelihood and AIC of the cox-models are now reported in the results

### Changed
- Added option to only output significant values from analyses 

## [0.2.2] - 2025-04-10

### Added
- Custom DPI settings for Kaplan-Meier Plots
- Custom title setting for plots
- Option to show the plot in a jupyter notebook 

### Changed
- Improved error reporting for split strategy in plotting

### Fixed
- Change the unit of the confidence interval to be in the correct scale.

## [0.2.1] - 2025-04-08

### Added
- A results converter, that converts the xlsx-results to pdf, txt and csv for easier sharing.

### Fixed
- working with copies at univariate regression to make sure that the variables remain the same.

## [0.2.0] - 2025-03-19

### Added
- A BCA merger cli that merges the measurement file with the patient list

## [0.1.2] - 2025-03-12

### Added
- Option for stratification depending on upper and lower quartile in kaplan-meier-analysis
- allow categorical data for Cox-Regression



## [0.1.1] - 2025-03-01

### Added
- Initial tag
- BOA extractor tool for processing Body and Organ Analysis output
- Basic Cox proportional hazards regression
- Univariate and multivariate survival analysis with lifelines integration
- Command-line interface for data extraction

