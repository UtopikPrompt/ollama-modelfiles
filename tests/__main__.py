"""Entry point for running tests with unittest."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    import unittest
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover(
        str(Path(__file__).parent),
        pattern="test_*.py",
        top_level_dir=str(Path(__file__).parent.parent)
    )
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)
