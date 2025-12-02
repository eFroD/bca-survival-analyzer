[![pipeline status](https://fdm-git.diz-ag.med.ovgu.de/ukf-racoon-gruppe/bca-survival-analyzer/badges/main/pipeline.svg)](https://fdm-git.diz-ag.med.ovgu.de/ukf-racoon-gruppe/bca-survival-analyzer/-/commits/main)
# Survival Analysis Package

A Python package for analyzing survival data with a focus on body composition assessment.

## Features

- **Survival Analysis**: Cox proportional hazards regression and Kaplan-Meier survival curves
- **Body Composition Analysis**: Tools for processing and analyzing BCA data
- **BOA Extractor**: Command-line tool for extracting measurements from BOA data
- **Data Preprocessing**: Utilities for cleaning and preparing survival data
- **CLI Tools**: Command-line utilities for data merging, format conversion, and PDF encryption

## Installation

Check the [Package Registry](https://fdm-git.diz-ag.med.ovgu.de/ukf-racoon-gruppe/bca-survival-analyzer/-/packages) for installation details.

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

## Command-Line Tools

The package includes several command-line tools for common data processing tasks:

### BOA Extractor

Extract measurements from BOA (Body Composition Assessment) data:
```bash
boa-extract /path/to/data /path/to/output
```

**Purpose**: Processes BOA data files and extracts relevant measurements for survival analysis.

**Arguments**:
- `data_path`: Path to the directory containing BOA data files
- `output_path`: Path where extracted measurements will be saved

---

### BCA Merger

Merge two Excel files based on ID columns:
```bash
bca-merge <first_file> <second_file> <id_column_name>
```

**Purpose**: Combines clinical data with body composition measurements by matching on ID columns.

**Arguments**:
- `first_file`: Path to the first Excel file (e.g., clinical data)
- `second_file`: Path to the second Excel file (e.g., BCA measurements)
- `id_column_name`: Name of the ID column in the first file to match with 'StudyID' in the second file

**Example**:
```bash
bca-merge clinical_data.xlsx bca_measurements.xlsx patient_id
```

**Output**: Creates a file named `{first_file}_merged.xlsx` with:
- All rows from both files (outer join)
- Matched records combined into single rows
- Date columns formatted as DD.MM.YYYY
- No duplicate StudyID columns

**Notes**:
- The second file must have a column named 'StudyID'
- Uses outer merge to preserve all data from both files
- Automatically removes duplicate ID columns

---

### Survival Result Converter

Convert Excel files to multiple formats (PDF, CSV, TXT):
```bash
survival-result-converter [directory]
```

**Purpose**: Batch converts Excel files to multiple formats for reporting and data sharing.

**Arguments**:
- `directory`: Directory to scan for Excel files (default: current directory)

**Example**:
```bash
# Convert all Excel files in current directory
survival-result-converter

# Convert Excel files in specific directory
survival-result-converter /path/to/results
```

**Output Structure**:
```
directory/
├── PDF/
│   ├── file1.pdf
│   └── file2.pdf
├── CSV/
│   ├── file1.csv
│   ├── file2_sheet1.csv
│   └── file2_sheet2.csv
└── TXT/
    ├── file1.txt
    └── file2.txt
```

**Features**:
- Recursively processes all `.xlsx` files in the directory tree
- Creates separate output folders (PDF, CSV, TXT)
- For multi-sheet Excel files:
  - PDF: All sheets in single file
  - CSV: Separate file per sheet
  - TXT: All sheets in single file with separators
- PDF generation supports two methods:
  - Windows: Uses COM automation for high-quality output
  - Cross-platform: Uses fpdf library with automatic column sizing

**PDF Features**:
- Landscape orientation for better table visibility
- Automatic column width adjustment
- Fits tables to page width
- Handles large tables (up to 1000 rows per sheet)
- Text wrapping for long content

---

### PDF Report Extractor

Encrypt and organize PDF files from a directory tree:
```bash
pdf-report-extractor <input_path> <output_path> <password>
```

**Purpose**: Finds PDF files in a directory structure, copies them with standardized names, and encrypts them for secure distribution.

**Arguments**:
- `input_path`: Root directory to search for PDF files
- `output_path`: Destination directory for encrypted PDFs
- `password`: Password to encrypt the PDFs with

**Example**:
```bash
pdf-report-extractor /data/patient_reports /encrypted_reports MySecureP@ss123
```

**Behavior**:
- Recursively searches for all `.pdf` files
- Copies files to destination with naming pattern: `encrypted_{parent_folder_name}.pdf`
- Encrypts each file using user password protection
- Requires `pdftk` to be installed

**Check pdftk Installation**:
```bash
pdf-report-extractor --check-pdftk
```

**Installing pdftk**:
- Ubuntu/Debian: `sudo apt-get install pdftk`
- macOS: `brew install pdftk-java`
- Windows: Download from [PDFtk website](https://www.pdflabs.com/tools/pdftk-the-pdf-toolkit/)

**Output Summary**:
```
Processing: /data/patient_reports/folder1/report.pdf
  -> /encrypted_reports/encrypted_folder1.pdf
  -> Encrypted successfully

Processing complete:
  - Files processed successfully: 15
  - Errors: 0
```

**Notes**:
- Original files remain unchanged
- If encryption fails, the unencrypted copy is removed from destination
- Parent folder name is used for output filename (one level up from the PDF)

---

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

## Requirements

### Core Dependencies
- pandas
- openpyxl (for Excel file handling)
- lifelines (for survival analysis)

### Optional Dependencies
- **For PDF conversion** (survival-result-converter):
  - Windows: pywin32
  - Cross-platform: fpdf, openpyxl
- **For PDF encryption** (pdf-report-extractor):
  - pdftk (external dependency)

## Common Workflows

### Workflow 1: Complete Data Processing Pipeline
```bash
# 1. Merge clinical and BCA data
bca-merge clinical.xlsx measurements.xlsx PatientID

# 2. Perform survival analysis (Python)
# ... (use BCASurvivalAnalyzer)

# 3. Convert results to multiple formats
survival-result-converter ./results

# 4. Encrypt PDF reports for distribution
pdf-report-extractor ./results/PDF ./encrypted_reports SecurePassword123
```

### Workflow 2: Quick Data Conversion
```bash
# Convert a directory of Excel results to PDF
survival-result-converter /path/to/results

# PDFs are created in /path/to/results/PDF/
```
