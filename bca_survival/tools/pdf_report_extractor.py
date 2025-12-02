#!/usr/bin/env python3
"""
PDF Encryption Script
Finds all PDF files in a directory tree, copies them to a destination folder
with names based on their parent directory, and encrypts them using pdftk.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Union


def encrypt_pdf(input_path: Path, output_path: Path, password: str) -> bool:
    """
    Encrypt a PDF file using pdftk.

    Args:
        input_path: Path to the input PDF file
        output_path: Path to save the encrypted PDF
        password: Password to encrypt the PDF with

    Returns:
        bool: True if successful, False otherwise
    """
    temp_path = output_path.parent / "_temp.pdf"

    try:
        # Run pdftk command to encrypt the PDF
        cmd = ["pdftk", str(input_path), "output", str(temp_path), "user_pw", password]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error encrypting {input_path}: {result.stderr}")
            return False

        # Move the temp file to the final location
        shutil.move(str(temp_path), str(output_path))
        return True

    except FileNotFoundError:
        print("Error: pdftk not found. Please install pdftk to use this script.")
        return False
    except Exception as e:
        print(f"Error encrypting {input_path}: {e}")
        return False
    finally:
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()


def process_pdfs(
    root_path: Union[Path, str], destination_path: Union[Path, str], password: str
) -> None:
    """
    Find all PDFs in the root path, copy and encrypt them to the destination.

    Args:
        root_path: Root directory to search for PDFs
        destination_path: Directory to save encrypted PDFs
        password: Password for encryption
    """
    root_path = Path(root_path)
    destination_path = Path(destination_path)

    # Create destination directory if it doesn't exist
    destination_path.mkdir(parents=True, exist_ok=True)

    # Counter for processed files
    processed = 0
    errors = 0

    # Find all PDF files recursively
    for pdf_file in root_path.rglob("*.pdf"):
        try:
            # Get the parent folder name (one level up from the PDF)
            parent_folder = pdf_file.parent.name

            # Create the new filename with "encrypted_" prefix
            new_name = f"encrypted_{parent_folder}.pdf"
            destination_file = destination_path / new_name

            print(f"Processing: {pdf_file}")
            print(f"  -> {destination_file}")

            # Copy the file to destination
            shutil.copy2(pdf_file, destination_file)

            # Encrypt the copied file
            if encrypt_pdf(destination_file, destination_file, password):
                processed += 1
                print("  -> Encrypted successfully")
            else:
                errors += 1
                print("  -> Encryption failed")
                # Remove the unencrypted copy
                if destination_file.exists():
                    destination_file.unlink()

        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            errors += 1

    print("\nProcessing complete:")
    print(f"  - Files processed successfully: {processed}")
    print(f"  - Errors: {errors}")


def main() -> None:
    """Main function to handle command line arguments and run the script."""
    parser = argparse.ArgumentParser(description="Copy and encrypt PDF files from a directory tree")
    parser.add_argument("input_path", help="Root path to search for PDF files")
    parser.add_argument("output_path", help="Destination path for encrypted PDFs")
    parser.add_argument("password", help="Password to encrypt PDFs with")
    parser.add_argument(
        "--check-pdftk", action="store_true", help="Check if pdftk is installed and exit"
    )

    args = parser.parse_args()

    # Check if pdftk is available
    if args.check_pdftk or True:  # Always check on first run
        try:
            result = subprocess.run(["pdftk", "--version"], capture_output=True, text=True)
            if args.check_pdftk:
                print("pdftk is installed and available")
                print(result.stdout)
                return
        except FileNotFoundError:
            print("Error: pdftk is not installed or not in PATH")
            print("Please install pdftk to use this script:")
            print("  - Ubuntu/Debian: sudo apt-get install pdftk")
            print("  - macOS: brew install pdftk-java")
            print("  - Windows: Download from https://www.pdflabs.com/tools/pdftk-the-pdf-toolkit/")
            sys.exit(1)

    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"Error: Input path does not exist: {args.input_path}")
        sys.exit(1)

    # Run the main processing
    process_pdfs(args.input_path, args.output_path, args.password)


if __name__ == "__main__":
    main()
