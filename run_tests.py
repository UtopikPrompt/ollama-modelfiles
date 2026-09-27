"""Run end-to-end tests for split model download functionality.

This script runs all tests in the tests package and reports results.

Usage:
    python3 -m tests
    python3 tests/test_split_model_download.py

To run with specific test class or method:
    python3 -m tests -v TestSplitModelDetection.test_split_model_filename_detection

To run with verbose output:
    python3 -m tests -v

To run with test discovery:
    python3 -m tests -v --discover
"""
import subprocess
import sys

def main():
    # Run pytest or unittest
    if len(sys.argv) > 1 and sys.argv[1] == "-m":
        # Run with pytest
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
    else:
        # Run with unittest
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "tests", "-v"],
            capture_output=True,
            text=True
        )
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print("="*60)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
