"""
Test suite for BOA Results extractor script.

This script tests the functionality of the BOA Results extractor by creating mock
directory structures and JSON files to mimic the real data environment.
"""

import json
import os
import shutil

# Import the module to be tested - adjust the import path as needed
import sys
import tempfile
import unittest

import pandas as pd

from tools.bca_totalseg_extraction import (
    process_bca_measurements,
    process_json_files,
    process_totalseg_measurements,
)

sys.path.append("tools")  # Add the parent directory to path if needed


class TestBOAExtractor(unittest.TestCase):
    """Tests for BOA Results extractor functions."""

    def setUp(self):
        """Set up temporary directory structure and mock data for tests."""
        # Create a temporary directory for our tests
        self.test_dir = tempfile.mkdtemp()

        # Create a mock directory structure
        # /tmp/test_dir/
        # ├── study1/
        # │   └── scan1/
        # │       ├── total-measurements.json
        # │       └── bca-measurements.json
        # └── study2/
        #     └── scan2/
        #         ├── total-measurements.json
        #         └── bca-measurements.json

        # Create study1 structure
        self.study1_path = os.path.join(self.test_dir, "study1")
        self.scan1_path = os.path.join(self.study1_path, "scan1")
        os.makedirs(self.scan1_path)

        # Create study2 structure
        self.study2_path = os.path.join(self.test_dir, "study2")
        self.scan2_path = os.path.join(self.study2_path, "scan2")
        os.makedirs(self.scan2_path)

        # Create mock total-measurements.json files
        self.total_meas1 = os.path.join(self.scan1_path, "total-measurements.json")
        total_data1 = {
            "segmentations": {
                "total": {
                    "liver": {"present": True, "volume": 1500.5, "hu_mean": 50.2},
                    "spleen": {"present": True, "volume": 250.3, "hu_mean": 45.1},
                    "kidney_left": {"present": False},
                }
            }
        }
        with open(self.total_meas1, "w") as f:
            json.dump(total_data1, f)

        self.total_meas2 = os.path.join(self.scan2_path, "total-measurements.json")
        total_data2 = {
            "segmentations": {
                "total": {
                    "liver": {"present": True, "volume": 1650.2, "hu_mean": 52.8},
                    "spleen": {"present": True, "volume": 270.5, "hu_mean": 46.3},
                    "kidney_left": {"present": True, "volume": 180.1, "hu_mean": 40.0},
                }
            }
        }
        with open(self.total_meas2, "w") as f:
            json.dump(total_data2, f)

        # Create mock bca-measurements.json files
        self.bca_meas1 = os.path.join(self.scan1_path, "bca-measurements.json")
        bca_data1 = {
            "aggregated": {
                "l3": {
                    "measurements": 0,
                    "worldLayerMeasurements": {
                        "imat": {"mean": 10.2, "sum": 150.5, "mean_hu": -90.1},
                        "tat": {"mean": 50.3, "sum": 750.8, "mean_hu": -110.2},
                    },
                },
                "l4": {
                    "measurements": 0,
                    "worldLayerMeasurements": {
                        "imat": {"mean": 12.5, "sum": 170.2, "mean_hu": -92.3},
                        "tat": {"mean": 55.6, "sum": 780.4, "mean_hu": -108.5},
                    },
                },
            }
        }
        with open(self.bca_meas1, "w") as f:
            json.dump(bca_data1, f)

        self.bca_meas2 = os.path.join(self.scan2_path, "bca-measurements.json")
        bca_data2 = {
            "aggregated": {
                "l3": {
                    "measurements": 0,
                    "worldLayerMeasurements": {
                        "imat": {"mean": 11.8, "sum": 160.3, "mean_hu": -91.5},
                        "tat": {"mean": 52.1, "sum": 770.6, "mean_hu": -112.0},
                    },
                },
                "t12": {
                    "measurements": 0,
                    "worldLayerMeasurements": {
                        "muscle": {"mean": 40.2, "sum": 600.5, "mean_hu": 45.6},
                        "bone": {"mean": 80.3, "sum": 400.1, "mean_hu": 250.7},
                    },
                },
            }
        }
        with open(self.bca_meas2, "w") as f:
            json.dump(bca_data2, f)

        # Create output directory
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir)

    def tearDown(self):
        """Clean up temporary files and directories."""
        shutil.rmtree(self.test_dir)

    def test_process_totalseg_measurements(self):
        """Test processing of total-measurements.json files."""
        # Test with the first file
        df1 = process_totalseg_measurements(self.total_meas1, self.scan1_path)

        # Check that the DataFrame has the expected structure
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(len(df1), 1)  # Should have one row
        self.assertEqual(df1.loc[0, "StudyID"], "study1")
        self.assertEqual(df1.loc[0, "liver::volume"], 1500.5)
        self.assertEqual(df1.loc[0, "liver::hu_mean"], 50.2)
        self.assertEqual(df1.loc[0, "spleen::volume"], 250.3)
        self.assertEqual(df1.loc[0, "spleen::hu_mean"], 45.1)

        # Test with the second file
        df2 = process_totalseg_measurements(self.total_meas2, self.scan2_path)

        # Check that the DataFrame has the expected structure
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertEqual(len(df2), 1)  # Should have one row
        self.assertEqual(df2.loc[0, "StudyID"], "study2")
        self.assertEqual(df2.loc[0, "liver::volume"], 1650.2)
        self.assertEqual(df2.loc[0, "kidney_left::volume"], 180.1)

    def test_process_bca_measurements(self):
        """Test processing of bca-measurements.json files."""
        # Test with the first file
        df1 = process_bca_measurements(self.scan1_path)

        # Check that the DataFrame has the expected structure
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(len(df1), 1)  # Should have one row
        self.assertEqual(df1.loc[0, "StudyID"], "study1")
        self.assertEqual(df1.loc[0, "l3::WL::imat::mean_ml"], 10.2)
        self.assertEqual(df1.loc[0, "l3::WL::imat::sum_ml"], 150.5)
        self.assertEqual(df1.loc[0, "l3::WL::imat::mean_hu"], -90.1)
        self.assertEqual(df1.loc[0, "l4::WL::tat::mean_ml"], 55.6)

        # Test with the second file
        df2 = process_bca_measurements(self.scan2_path)

        # Check that the DataFrame has the expected structure
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertEqual(len(df2), 1)  # Should have one row
        self.assertEqual(df2.loc[0, "StudyID"], "study2")
        self.assertEqual(df2.loc[0, "t12::WL::muscle::mean_ml"], 40.2)
        self.assertEqual(df2.loc[0, "t12::WL::bone::mean_hu"], 250.7)

        # Test with a non-existing file
        non_existent_path = os.path.join(self.test_dir, "non_existent")
        os.makedirs(non_existent_path, exist_ok=True)
        df3 = process_bca_measurements(non_existent_path)
        self.assertIsNone(df3)  # Should return None

    def test_process_json_files(self):
        """Test the overall processing of all JSON files in a directory structure."""
        total_df, bca_df = process_json_files(self.test_dir)

        # Check total measurements DataFrame
        self.assertIsInstance(total_df, pd.DataFrame)
        self.assertEqual(len(total_df), 2)  # Should have two rows, one for each study
        study_ids = sorted(total_df["StudyID"].unique())
        self.assertEqual(study_ids, ["study1", "study2"])

        # Check that specific measurements are present
        liver_volumes = total_df.set_index("StudyID")["liver::volume"].to_dict()
        self.assertAlmostEqual(liver_volumes["study1"], 1500.5)
        self.assertAlmostEqual(liver_volumes["study2"], 1650.2)

        # Check BCA measurements DataFrame
        self.assertIsInstance(bca_df, pd.DataFrame)
        self.assertEqual(len(bca_df), 2)  # Should have two rows, one for each study
        study_ids = sorted(bca_df["StudyID"].unique())
        self.assertEqual(study_ids, ["study1", "study2"])

        # Check that specific measurements are present
        # For study1
        study1_row = bca_df[bca_df["StudyID"] == "study1"].iloc[0]
        self.assertEqual(study1_row["l3::WL::imat::mean_ml"], 10.2)
        self.assertEqual(study1_row["l4::WL::tat::sum_ml"], 780.4)

        # For study2
        study2_row = bca_df[bca_df["StudyID"] == "study2"].iloc[0]
        self.assertEqual(study2_row["t12::WL::muscle::sum_ml"], 600.5)
        self.assertEqual(study2_row["l3::WL::tat::mean_hu"], -112.0)

    def test_empty_directory(self):
        """Test handling of an empty directory with no measurement files."""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)

        total_df, bca_df = process_json_files(empty_dir)

        # Both DataFrames should be empty
        self.assertTrue(total_df.empty)
        self.assertTrue(bca_df.empty)

    def test_partial_data(self):
        """Test handling of directories with only some types of measurement files."""
        partial_dir = os.path.join(self.test_dir, "partial")
        partial_scan_dir = os.path.join(partial_dir, "study3", "scan3")
        os.makedirs(partial_scan_dir)

        # Create only a total-measurements.json file, no bca-measurements.json
        total_data = {
            "segmentations": {
                "total": {"liver": {"present": True, "volume": 1700.1, "hu_mean": 53.5}}
            }
        }
        with open(os.path.join(partial_scan_dir, "total-measurements.json"), "w") as f:
            json.dump(total_data, f)

        total_df, bca_df = process_json_files(partial_dir)

        # total_df should have data, bca_df should be empty
        self.assertFalse(total_df.empty)
        self.assertTrue(bca_df.empty)
        self.assertEqual(total_df.loc[0, "StudyID"], "study3")
        self.assertEqual(total_df.loc[0, "liver::volume"], 1700.1)


if __name__ == "__main__":
    unittest.main()
