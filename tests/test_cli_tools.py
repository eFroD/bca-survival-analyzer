"""
Test suite for CLI tools in bca_survival.tools module.

This module tests the following CLI tools:
- results_converter: Converts Excel files to PDF, CSV, and TXT formats
- bca_merger: Merges two Excel files based on ID columns
- pdf_report_extractor: Encrypts PDF files using pdftk

Note: Some tests require optional dependencies (openpyxl, fpdf) and external tools (pdftk).
Tests for unavailable dependencies are skipped gracefully.
"""

import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Import the modules to test
from bca_survival.tools import bca_merger, results_converter


def robust_rmtree(path: str, max_retries: int = 5, delay: float = 0.5) -> None:
    """
    Remove a directory tree with retries for Windows file locking issues.

    On Windows, file handles may not be released immediately even after
    closing files. This function retries removal with delays.

    Args:
        path: Path to the directory to remove
        max_retries: Maximum number of retry attempts
        delay: Delay in seconds between retries
    """
    for attempt in range(max_retries):
        try:
            # Force garbage collection to release any file handles
            gc.collect()
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                # On final attempt, try to remove files individually
                # and ignore errors for locked files
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except PermissionError:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except (PermissionError, OSError):
                            pass
                try:
                    os.rmdir(path)
                except (PermissionError, OSError):
                    pass  # Best effort cleanup


class TestBCAMerger(unittest.TestCase):
    """Tests for the bca_merger module."""

    def setUp(self):
        """Set up temporary directory and test files."""
        self.test_dir = tempfile.mkdtemp()

        # Create first Excel file (clinical data)
        self.df1 = pd.DataFrame(
            {
                "patient_id": [1, 2, 3, 4, 5],
                "age": [45, 52, 38, 67, 55],
                "gender": ["M", "F", "M", "F", "M"],
                "diagnosis_date": pd.to_datetime(
                    ["2020-01-15", "2020-02-20", "2020-03-10", "2020-04-05", "2020-05-25"]
                ),
            }
        )
        self.first_file = os.path.join(self.test_dir, "clinical_data.xlsx")
        self.df1.to_excel(self.first_file, index=False, engine="openpyxl")

        # Create second Excel file (BCA measurements)
        self.df2 = pd.DataFrame(
            {
                "StudyID": [1, 2, 3, 6],  # Note: patient 4,5 missing; patient 6 extra
                "muscle_mass": [45.2, 38.1, 52.3, 41.0],
                "fat_mass": [22.1, 28.5, 18.3, 25.0],
            }
        )
        self.second_file = os.path.join(self.test_dir, "bca_measurements.xlsx")
        self.df2.to_excel(self.second_file, index=False, engine="openpyxl")

    def tearDown(self):
        """Clean up temporary directory."""
        robust_rmtree(self.test_dir)

    def test_merge_files_success(self):
        """Test successful merge of two Excel files."""
        # Change to test directory for output file
        original_dir = os.getcwd()
        os.chdir(self.test_dir)

        try:
            result = bca_merger.merge_files(self.first_file, self.second_file, "patient_id")

            self.assertTrue(result)

            # Check output file exists
            output_file = os.path.join(self.test_dir, "clinical_data_merged.xlsx")
            self.assertTrue(os.path.exists(output_file))

            # Read and verify merged file
            merged_df = pd.read_excel(output_file, engine="openpyxl")

            # Should have all patients from both files (outer merge)
            # Patients 1,2,3 from both, 4,5 from first only, 6 from second only
            self.assertEqual(len(merged_df), 6)

            # Check columns are present
            self.assertIn("patient_id", merged_df.columns)
            self.assertIn("age", merged_df.columns)
            self.assertIn("muscle_mass", merged_df.columns)
            self.assertIn("fat_mass", merged_df.columns)

            # StudyID should be removed (duplicate of patient_id)
            self.assertNotIn("StudyID", merged_df.columns)

        finally:
            os.chdir(original_dir)

    def test_merge_files_missing_id_column(self):
        """Test merge fails gracefully when ID column is missing."""
        with self.assertRaises(SystemExit):
            bca_merger.merge_files(self.first_file, self.second_file, "nonexistent_column")

    def test_merge_files_missing_study_id(self):
        """Test merge fails gracefully when StudyID column is missing."""
        # Create a file without StudyID column
        df_no_study_id = pd.DataFrame({"other_id": [1, 2, 3], "value": [10, 20, 30]})
        no_study_id_file = os.path.join(self.test_dir, "no_study_id.xlsx")
        df_no_study_id.to_excel(no_study_id_file, index=False, engine="openpyxl")

        with self.assertRaises(SystemExit):
            bca_merger.merge_files(self.first_file, no_study_id_file, "patient_id")

    def test_merge_preserves_data_types(self):
        """Test that merge preserves numeric data correctly."""
        original_dir = os.getcwd()
        os.chdir(self.test_dir)

        try:
            bca_merger.merge_files(self.first_file, self.second_file, "patient_id")

            output_file = os.path.join(self.test_dir, "clinical_data_merged.xlsx")
            merged_df = pd.read_excel(output_file, engine="openpyxl")

            # Check that numeric values are preserved
            patient_1_row = merged_df[merged_df["patient_id"] == 1].iloc[0]
            self.assertEqual(patient_1_row["age"], 45)
            self.assertAlmostEqual(patient_1_row["muscle_mass"], 45.2, places=1)

        finally:
            os.chdir(original_dir)

    def test_main_with_missing_file(self):
        """Test main function handles missing files."""
        with patch.object(sys, "argv", ["bca-merge", "nonexistent.xlsx", self.second_file, "id"]):
            with self.assertRaises(SystemExit):
                bca_merger.main()

    def test_main_with_wrong_arguments(self):
        """Test main function handles wrong number of arguments."""
        with patch.object(sys, "argv", ["bca-merge", "only_one_arg.xlsx"]):
            with self.assertRaises(SystemExit):
                bca_merger.main()


class TestResultsConverter(unittest.TestCase):
    """Tests for the results_converter module."""

    def setUp(self):
        """Set up temporary directory and test Excel files."""
        self.test_dir = tempfile.mkdtemp()

        # Create a simple Excel file with one sheet
        self.df_single = pd.DataFrame(
            {"name": ["Alice", "Bob", "Charlie"], "score": [85, 92, 78], "grade": ["B", "A", "C"]}
        )
        self.single_sheet_file = os.path.join(self.test_dir, "single_sheet.xlsx")
        self.df_single.to_excel(self.single_sheet_file, index=False, engine="openpyxl")

        # Create an Excel file with multiple sheets
        self.multi_sheet_file = os.path.join(self.test_dir, "multi_sheet.xlsx")
        with pd.ExcelWriter(self.multi_sheet_file, engine="openpyxl") as writer:
            self.df_single.to_excel(writer, sheet_name="Sheet1", index=False)
            pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}).to_excel(
                writer, sheet_name="Sheet2", index=False
            )

    def tearDown(self):
        """Clean up temporary directory."""
        # Force garbage collection before cleanup
        gc.collect()
        robust_rmtree(self.test_dir)

    def test_create_output_folders(self):
        """Test that output folders are created correctly."""
        results_converter.create_output_folders(self.test_dir)

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "PDF")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "CSV")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "TXT")))

    def test_create_output_folders_idempotent(self):
        """Test that creating folders twice doesn't cause errors."""
        results_converter.create_output_folders(self.test_dir)
        results_converter.create_output_folders(self.test_dir)  # Should not raise

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "CSV")))

    def test_convert_to_csv_single_sheet(self):
        """Test CSV conversion for single-sheet Excel file."""
        output_folder = os.path.join(self.test_dir, "CSV")
        os.makedirs(output_folder, exist_ok=True)

        results_converter.convert_to_csv(self.single_sheet_file, output_folder)

        # Check output file exists
        output_file = os.path.join(output_folder, "single_sheet.csv")
        self.assertTrue(os.path.exists(output_file))

        # Verify content
        df_result = pd.read_csv(output_file)
        self.assertEqual(len(df_result), 3)
        self.assertListEqual(list(df_result.columns), ["name", "score", "grade"])

    def test_convert_to_csv_multi_sheet(self):
        """Test CSV conversion for multi-sheet Excel file."""
        output_folder = os.path.join(self.test_dir, "CSV")
        os.makedirs(output_folder, exist_ok=True)

        results_converter.convert_to_csv(self.multi_sheet_file, output_folder)

        # Check output files exist (one per sheet)
        output_file1 = os.path.join(output_folder, "multi_sheet_Sheet1.csv")
        output_file2 = os.path.join(output_folder, "multi_sheet_Sheet2.csv")

        self.assertTrue(os.path.exists(output_file1))
        self.assertTrue(os.path.exists(output_file2))

    def test_convert_to_txt_single_sheet(self):
        """Test TXT conversion for single-sheet Excel file."""
        output_folder = os.path.join(self.test_dir, "TXT")
        os.makedirs(output_folder, exist_ok=True)

        results_converter.convert_to_txt(self.single_sheet_file, output_folder)

        # Check output file exists
        output_file = os.path.join(output_folder, "single_sheet.txt")
        self.assertTrue(os.path.exists(output_file))

        # Verify content contains data
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Alice", content)
            self.assertIn("Bob", content)
            self.assertIn("85", content)

    def test_convert_to_txt_multi_sheet(self):
        """Test TXT conversion for multi-sheet Excel file."""
        output_folder = os.path.join(self.test_dir, "TXT")
        os.makedirs(output_folder, exist_ok=True)

        results_converter.convert_to_txt(self.multi_sheet_file, output_folder)

        # Check output file exists
        output_file = os.path.join(output_folder, "multi_sheet.txt")
        self.assertTrue(os.path.exists(output_file))

        # Verify content contains sheet separators
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Sheet: Sheet1", content)
            self.assertIn("Sheet: Sheet2", content)
            self.assertIn("=" * 50, content)

    def test_convert_excel_file(self):
        """Test full conversion of Excel file to all formats."""
        results_converter.convert_excel_file(self.single_sheet_file)

        # Check that all output folders were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "CSV")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "TXT")))

        # Check that CSV and TXT files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "CSV", "single_sheet.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "TXT", "single_sheet.txt")))

    def test_process_directory(self):
        """Test processing entire directory for Excel files."""
        results_converter.process_directory(self.test_dir)

        # Check that CSV files were created for both Excel files
        csv_folder = os.path.join(self.test_dir, "CSV")
        self.assertTrue(os.path.exists(os.path.join(csv_folder, "single_sheet.csv")))

    def test_process_directory_no_excel_files(self):
        """Test processing directory with no Excel files."""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)

        # Should not raise an error
        results_converter.process_directory(empty_dir)

    def test_convert_to_csv_handles_error(self):
        """Test CSV conversion handles errors gracefully."""
        output_folder = os.path.join(self.test_dir, "CSV")
        os.makedirs(output_folder, exist_ok=True)

        # Try to convert non-existent file
        results_converter.convert_to_csv("nonexistent.xlsx", output_folder)
        # Should print error but not raise exception

    def test_convert_to_txt_handles_error(self):
        """Test TXT conversion handles errors gracefully."""
        output_folder = os.path.join(self.test_dir, "TXT")
        os.makedirs(output_folder, exist_ok=True)

        # Try to convert non-existent file
        results_converter.convert_to_txt("nonexistent.xlsx", output_folder)
        # Should print error but not raise exception

    def test_main_with_valid_directory(self):
        """Test main function with valid directory."""
        with patch.object(sys, "argv", ["survival-result-converter", self.test_dir]):
            results_converter.main()

        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "CSV")))

    def test_main_with_invalid_directory(self):
        """Test main function with invalid directory."""
        with patch.object(sys, "argv", ["survival-result-converter", "/nonexistent/path"]):
            with self.assertRaises(SystemExit):
                results_converter.main()

    def test_main_with_default_directory(self):
        """Test main function with default (current) directory."""
        original_dir = os.getcwd()
        os.chdir(self.test_dir)

        try:
            with patch.object(sys, "argv", ["survival-result-converter"]):
                results_converter.main()

            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, "CSV")))
        finally:
            os.chdir(original_dir)

    @unittest.skipIf(
        results_converter.PDF_CONVERTER is None, "No PDF converter available (fpdf or win32com)"
    )
    def test_convert_to_pdf_fpdf(self):
        """Test PDF conversion using fpdf (if available)."""
        if results_converter.PDF_CONVERTER != "fpdf":
            self.skipTest("fpdf converter not available")

        output_folder = os.path.join(self.test_dir, "PDF")
        os.makedirs(output_folder, exist_ok=True)

        results_converter.convert_to_pdf_fpdf(self.single_sheet_file, output_folder)

        # Check output file exists
        output_file = os.path.join(output_folder, "single_sheet.pdf")
        self.assertTrue(os.path.exists(output_file))

        # Check file is not empty
        self.assertGreater(os.path.getsize(output_file), 0)


class TestPDFReportExtractor(unittest.TestCase):
    """Tests for the pdf_report_extractor module."""

    def setUp(self):
        """Set up temporary directory and test PDF files."""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "input")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.input_dir)

        # Check if pdftk is available
        self.pdftk_available = self._check_pdftk()

    def _check_pdftk(self):
        """Check if pdftk is installed."""
        try:
            subprocess.run(
                ["pdftk", "--version"], capture_output=True, text=True, check=False, timeout=5
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _create_test_pdf(self, path):
        """Create a simple test PDF file using fpdf if available, otherwise a dummy file."""
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Test PDF Content", ln=1, align="C")
            pdf.output(path)
        except ImportError:
            # Create a minimal valid PDF manually
            # This is a minimal valid PDF structure
            with open(path, "wb") as f:
                f.write(b"%PDF-1.4\n")
                f.write(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
                f.write(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
                f.write(
                    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
                )
                f.write(b"xref\n0 4\n")
                f.write(b"0000000000 65535 f \n")
                f.write(b"0000000009 00000 n \n")
                f.write(b"0000000058 00000 n \n")
                f.write(b"0000000115 00000 n \n")
                f.write(b"trailer\n<< /Size 4 /Root 1 0 R >>\n")
                f.write(b"startxref\n198\n%%EOF\n")

    def tearDown(self):
        """Clean up temporary directory."""
        robust_rmtree(self.test_dir)

    def test_process_pdfs_creates_output_directory(self):
        """Test that process_pdfs creates the output directory."""
        from bca_survival.tools import pdf_report_extractor

        # Create a test PDF
        patient_dir = os.path.join(self.input_dir, "patient_001")
        os.makedirs(patient_dir)
        self._create_test_pdf(os.path.join(patient_dir, "report.pdf"))

        # Run process_pdfs (will fail encryption without pdftk, but should create directory)
        pdf_report_extractor.process_pdfs(self.input_dir, self.output_dir, "test_password")

        self.assertTrue(os.path.exists(self.output_dir))

    def test_process_pdfs_finds_nested_pdfs(self):
        """Test that process_pdfs finds PDFs in nested directories."""
        from bca_survival.tools import pdf_report_extractor

        # Create nested directory structure with PDFs
        patient1_dir = os.path.join(self.input_dir, "patient_001")
        patient2_dir = os.path.join(self.input_dir, "subdir", "patient_002")
        os.makedirs(patient1_dir)
        os.makedirs(patient2_dir)

        self._create_test_pdf(os.path.join(patient1_dir, "report.pdf"))
        self._create_test_pdf(os.path.join(patient2_dir, "scan.pdf"))

        # Mock the encrypt_pdf function to track calls
        with patch.object(pdf_report_extractor, "encrypt_pdf", return_value=True) as mock_encrypt:
            pdf_report_extractor.process_pdfs(self.input_dir, self.output_dir, "test_password")

            # Should have found 2 PDFs
            self.assertEqual(mock_encrypt.call_count, 2)

    def test_process_pdfs_uses_parent_folder_name(self):
        """Test that output files are named based on parent folder."""
        from bca_survival.tools import pdf_report_extractor

        # Create a PDF in a named folder
        patient_dir = os.path.join(self.input_dir, "patient_ABC123")
        os.makedirs(patient_dir)
        self._create_test_pdf(os.path.join(patient_dir, "report.pdf"))

        # Mock encrypt_pdf to track the output path
        with patch.object(pdf_report_extractor, "encrypt_pdf", return_value=True) as mock_encrypt:
            pdf_report_extractor.process_pdfs(self.input_dir, self.output_dir, "test_password")

            # Check that the output filename includes the parent folder name
            call_args = mock_encrypt.call_args
            output_path = call_args[0][1]  # Second positional argument is output_path
            self.assertIn("patient_ABC123", str(output_path))

    @unittest.skipUnless(
        shutil.which("pdftk") is not None, "pdftk not installed - skipping encryption test"
    )
    def test_encrypt_pdf_success(self):
        """Test successful PDF encryption with pdftk."""
        from bca_survival.tools import pdf_report_extractor

        # Create test PDF
        input_pdf = os.path.join(self.test_dir, "test_input.pdf")
        output_pdf = os.path.join(self.test_dir, "test_output.pdf")
        self._create_test_pdf(input_pdf)

        result = pdf_report_extractor.encrypt_pdf(Path(input_pdf), Path(output_pdf), "password123")

        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_pdf))

    def test_encrypt_pdf_no_pdftk(self):
        """Test encrypt_pdf handles missing pdftk gracefully."""
        from bca_survival.tools import pdf_report_extractor

        input_pdf = Path(self.test_dir) / "test_input.pdf"
        output_pdf = Path(self.test_dir) / "test_output.pdf"

        # Create a test PDF
        self._create_test_pdf(str(input_pdf))

        # Mock subprocess.run to simulate pdftk not found
        with patch("subprocess.run", side_effect=FileNotFoundError("pdftk not found")):
            result = pdf_report_extractor.encrypt_pdf(input_pdf, output_pdf, "password")

        self.assertFalse(result)

    def test_encrypt_pdf_cleans_up_temp_file(self):
        """Test that encrypt_pdf cleans up temporary files on failure."""
        from bca_survival.tools import pdf_report_extractor

        input_pdf = Path(self.test_dir) / "test_input.pdf"
        output_pdf = Path(self.test_dir) / "test_output.pdf"
        temp_pdf = Path(self.test_dir) / "_temp.pdf"

        self._create_test_pdf(str(input_pdf))

        # Mock subprocess to fail
        with patch("subprocess.run", side_effect=Exception("Mock error")):
            pdf_report_extractor.encrypt_pdf(input_pdf, output_pdf, "password")

        # Temp file should be cleaned up
        self.assertFalse(temp_pdf.exists())

    def test_main_check_pdftk_flag(self):
        """Test main function with --check-pdftk flag."""
        from bca_survival.tools import pdf_report_extractor

        with patch.object(
            sys,
            "argv",
            ["pdf-report-extractor", "--check-pdftk", "input", "output", "pass"],
        ):
            # Mock subprocess.run to simulate pdftk available
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="pdftk version 2.0", returncode=0)
                # Should not raise
                try:
                    pdf_report_extractor.main()
                except SystemExit:
                    pass  # Expected if pdftk check passes or fails

    def test_main_invalid_input_path(self):
        """Test main function with invalid input path."""
        from bca_survival.tools import pdf_report_extractor

        with patch.object(
            sys,
            "argv",
            ["pdf-report-extractor", "/nonexistent/path", self.output_dir, "password"],
        ):
            # Mock pdftk check to pass
            with patch("subprocess.run"):
                with self.assertRaises(SystemExit):
                    pdf_report_extractor.main()

    def test_process_pdfs_handles_copy_error(self):
        """Test process_pdfs handles file copy errors gracefully."""
        from bca_survival.tools import pdf_report_extractor

        # Create a PDF
        patient_dir = os.path.join(self.input_dir, "patient_001")
        os.makedirs(patient_dir)
        self._create_test_pdf(os.path.join(patient_dir, "report.pdf"))

        # Mock shutil.copy2 to raise an exception
        with patch("shutil.copy2", side_effect=PermissionError("Access denied")):
            # Should not raise, just print error
            pdf_report_extractor.process_pdfs(self.input_dir, self.output_dir, "password")

    def test_process_pdfs_empty_directory(self):
        """Test process_pdfs handles empty directory."""
        from bca_survival.tools import pdf_report_extractor

        # Run on empty input directory
        pdf_report_extractor.process_pdfs(self.input_dir, self.output_dir, "password")

        # Output directory should still be created
        self.assertTrue(os.path.exists(self.output_dir))


class TestResultsConverterLargeFiles(unittest.TestCase):
    """Tests for results_converter with larger/more complex Excel files."""

    def setUp(self):
        """Set up temporary directory and test files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        gc.collect()
        robust_rmtree(self.test_dir)

    def test_convert_large_dataframe(self):
        """Test conversion of Excel file with many rows."""
        # Create large dataframe
        np.random.seed(42)
        large_df = pd.DataFrame(
            {
                "id": range(500),
                "value1": np.random.randn(500),
                "value2": np.random.randn(500),
                "category": np.random.choice(["A", "B", "C"], 500),
            }
        )

        large_file = os.path.join(self.test_dir, "large_data.xlsx")
        large_df.to_excel(large_file, index=False, engine="openpyxl")

        # Convert to CSV
        csv_folder = os.path.join(self.test_dir, "CSV")
        os.makedirs(csv_folder)
        results_converter.convert_to_csv(large_file, csv_folder)

        # Verify
        output_file = os.path.join(csv_folder, "large_data.csv")
        self.assertTrue(os.path.exists(output_file))

        df_result = pd.read_csv(output_file)
        self.assertEqual(len(df_result), 500)

    def test_convert_wide_dataframe(self):
        """Test conversion of Excel file with many columns."""
        # Create wide dataframe
        wide_df = pd.DataFrame(
            {f"column_{i}": np.random.randn(10) for i in range(50)}  # 50 columns
        )

        wide_file = os.path.join(self.test_dir, "wide_data.xlsx")
        wide_df.to_excel(wide_file, index=False, engine="openpyxl")

        # Convert to TXT
        txt_folder = os.path.join(self.test_dir, "TXT")
        os.makedirs(txt_folder)
        results_converter.convert_to_txt(wide_file, txt_folder)

        # Verify
        output_file = os.path.join(txt_folder, "wide_data.txt")
        self.assertTrue(os.path.exists(output_file))

        with open(output_file, "r") as f:
            content = f.read()
            # Check some columns are present
            self.assertIn("column_0", content)
            self.assertIn("column_49", content)

    def test_convert_with_special_characters(self):
        """Test conversion handles special characters in data."""
        special_df = pd.DataFrame(
            {
                "name": ["Müller", "O'Brien", "José García", "北京"],
                "description": [
                    "Value with, comma",
                    'Value with "quotes"',
                    "Value with\nnewline",
                    "Normal value",
                ],
            }
        )

        special_file = os.path.join(self.test_dir, "special_chars.xlsx")
        special_df.to_excel(special_file, index=False, engine="openpyxl")

        # Convert to CSV
        csv_folder = os.path.join(self.test_dir, "CSV")
        os.makedirs(csv_folder)
        results_converter.convert_to_csv(special_file, csv_folder)

        # Verify the file was created and can be read back
        output_file = os.path.join(csv_folder, "special_chars.csv")
        self.assertTrue(os.path.exists(output_file))

        df_result = pd.read_csv(output_file)
        self.assertEqual(len(df_result), 4)


class TestBCAMergerEdgeCases(unittest.TestCase):
    """Additional edge case tests for bca_merger."""

    def setUp(self):
        """Set up temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        gc.collect()
        robust_rmtree(self.test_dir)

    def test_merge_with_duplicate_ids(self):
        """Test merge handles duplicate IDs correctly."""
        # Create first file with duplicate IDs
        df1 = pd.DataFrame(
            {"patient_id": [1, 1, 2, 3], "visit": [1, 2, 1, 1], "value1": [10, 20, 30, 40]}
        )
        first_file = os.path.join(self.test_dir, "duplicates.xlsx")
        df1.to_excel(first_file, index=False, engine="openpyxl")

        # Create second file
        df2 = pd.DataFrame({"StudyID": [1, 2, 3], "measurement": [100, 200, 300]})
        second_file = os.path.join(self.test_dir, "measurements.xlsx")
        df2.to_excel(second_file, index=False, engine="openpyxl")

        original_dir = os.getcwd()
        os.chdir(self.test_dir)

        try:
            result = bca_merger.merge_files(first_file, second_file, "patient_id")
            self.assertTrue(result)

            # Read merged file
            output_file = os.path.join(self.test_dir, "duplicates_merged.xlsx")
            merged_df = pd.read_excel(output_file, engine="openpyxl")

            # Should have all rows from first file merged with second
            # Patient 1 appears twice, so should be duplicated in merge
            self.assertGreaterEqual(len(merged_df), 4)
        finally:
            os.chdir(original_dir)

    def test_merge_with_empty_file(self):
        """Test merge handles empty files."""
        # Create empty first file (just headers)
        df1 = pd.DataFrame(columns=["patient_id", "value"])
        first_file = os.path.join(self.test_dir, "empty.xlsx")
        df1.to_excel(first_file, index=False, engine="openpyxl")

        # Create second file with data
        df2 = pd.DataFrame({"StudyID": [1, 2], "measurement": [100, 200]})
        second_file = os.path.join(self.test_dir, "measurements.xlsx")
        df2.to_excel(second_file, index=False, engine="openpyxl")

        original_dir = os.getcwd()
        os.chdir(self.test_dir)

        try:
            result = bca_merger.merge_files(first_file, second_file, "patient_id")
            self.assertTrue(result)
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    unittest.main()
