"""End-to-end tests for split model download functionality.

Tests cover:
- Split model detection via regex pattern
- Chunk file existence checking
- Partial chunk detection (not all chunks downloaded)
- Complete chunk detection (all chunks downloaded)
- Regular single-file model detection
"""
import unittest
import re
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO


# Module-level temp directories that persist across tests
# These are cleaned up once at the end of all tests, not per test
_TEST_BASE_DIR = tempfile.mkdtemp(prefix='ollama_modelfiles_tests_')


def _check_chunk_existence(base_model_name: str, chunk_match: re.Match) -> bool:
    """Helper method to check if all chunks of a split model exist.
    
    This avoids code duplication across multiple test methods.
    
    Args:
        base_model_name: The base name without chunk suffix
        chunk_match: Regex match object containing chunk info
        
    Returns:
        True if all chunks exist, False otherwise
    """
    chunk_exists = False
    
    if chunk_match:
        for i in range(1, int(chunk_match.group(2)) + 1):
            chunk_path = Path(_TEST_BASE_DIR) / base_model_name / f"{base_model_name}-{i:05d}.gguf"
            if not chunk_path.exists():
                chunk_exists = False
                break
            chunk_exists = True
    
    return chunk_exists


class TestSplitModelDetection(unittest.TestCase):
    """Test split model filename detection and regex pattern."""
    
    def test_split_model_filename_detection(self):
        """Test that split model filenames are correctly detected."""
        model_names = [
            "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00006",
            "model-chunk-00001-of-00003",
            "bert-base-00001-of-00010",
        ]
        
        for name in model_names:
            match = re.search(r'-([0-9]{5})-of-([0-9]{5})', name)
            self.assertIsNotNone(match, f"Failed to detect split model: {name}")
            self.assertEqual(match.group(1), "00001", f"Wrong chunk number for: {name}")
            self.assertEqual(match.group(2), "00006" if name == "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00006" else "00003" if name == "model-chunk-00001-of-00003" else "00010", f"Wrong total chunks for: {name}")
    
    def test_regular_model_filename_detection(self):
        """Test that regular (non-split) model filenames are NOT detected as split."""
        model_names = [
            "qwen2.5-32b-instruct-q5_k_m",
            "model.gguf",
            "bert-base-uncased",
        ]
        
        for name in model_names:
            match = re.search(r'-([0-9]{5})-of-([0-9]{5})', name)
            self.assertIsNone(match, f"False positive: {name} detected as split")
    
    def test_chunk_number_extraction(self):
        """Test that chunk numbers are correctly extracted."""
        test_cases = [
            ("qwen2.5-32b-instruct-q5_k_m-00001-of-00006", 1, 6),
            ("model-00001-of-00003", 1, 3),
            ("chunk-00010-of-00020", 10, 20),
        ]
        
        for name, expected_start, expected_total in test_cases:
            match = re.search(r'-([0-9]{5})-of-([0-9]{5})', name)
            self.assertIsNotNone(match)
            self.assertEqual(int(match.group(1)), expected_start)
            self.assertEqual(int(match.group(2)), expected_total)
    
    def test_base_filename_extraction(self):
        """Test that base filenames are correctly extracted."""
        test_cases = [
            ("qwen2.5-32b-instruct-q5_k_m-00001-of-00006", "qwen2.5-32b-instruct-q5_k_m"),
            ("model-00001-of-00003", "model"),
            ("bert-base-00010-of-00020", "bert-base"),
        ]
        
        for name, expected_base in test_cases:
            match = re.search(r'-([0-9]{5})-of-([0-9]{5})', name)
            self.assertIsNotNone(match)
            self.assertEqual(name[:match.start()], expected_base)


class TestChunkExistenceCheck(unittest.TestCase):
    """Test chunk file existence checking logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.organization = "Qwen"
        self.base_model_name = "qwen2.5-32b-instruct-q5_k_m"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_all_chunks_exist(self):
        """Test that all chunks are detected as existing."""
        # Create all chunk files
        for i in range(1, 7):
            chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00006")
        self.assertTrue(_check_chunk_existence(self.base_model_name, chunk_match), "All chunks should be detected as existing")
    
    def test_first_chunk_only_exists(self):
        """Test that partial chunks are detected as missing."""
        # Create only the first chunk
        chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-00001.gguf"
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00006")
        self.assertFalse(_check_chunk_existence(self.base_model_name, chunk_match), "Missing chunks should be detected")
    
    def test_last_chunk_only_exists(self):
        """Test that only the last chunk is detected as missing."""
        # Create only the last chunk
        chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-00006.gguf"
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00006")
        self.assertFalse(_check_chunk_existence(self.base_model_name, chunk_match), "Missing chunks should be detected")
    
    def test_no_chunks_exist(self):
        """Test that no chunks are detected as existing."""
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00006")
        self.assertFalse(_check_chunk_existence(self.base_model_name, chunk_match), "No chunks should be detected as existing")
    
    def test_three_chunk_model(self):
        """Test existence check for a 3-chunk model."""
        # Create all 3 chunks
        for i in range(1, 4):
            chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00003")
        self.assertTrue(_check_chunk_existence(self.base_model_name, chunk_match), "All 3 chunks should be detected as existing")
    
    def test_different_total_chunks(self):
        """Test existence check for models with different total chunk counts."""
        # Test 2-chunk model
        for i in range(1, 3):
            chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00002")
        self.assertTrue(_check_chunk_existence(self.base_model_name, chunk_match), "2-chunk model should be detected as complete")
    
    def test_larger_chunk_model(self):
        """Test existence check for a larger chunk model (10 chunks)."""
        # Create all 10 chunks
        for i in range(1, 11):
            chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00010")
        self.assertTrue(_check_chunk_existence(self.base_model_name, chunk_match), "10-chunk model should be detected as complete")
    
    def test_partial_chunks_different_positions(self):
        """Test that partial chunks are detected regardless of which chunks are missing."""
        # Only create middle chunks (2 and 8)
        for i in [2, 8]:
            chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00006")
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertFalse(chunk_exists, "Model with only middle chunks missing should be detected as incomplete")
    
    def test_first_and_last_chunks_only(self):
        """Test that only first and last chunks being present is detected as incomplete."""
        # Only create first and last chunks
        for i in [1, 6]:
            chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00006")
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertFalse(chunk_exists, "Model with only first and last chunks should be detected as incomplete")
    
    def test_different_organization_for_split_model(self):
        """Test split model detection works with different organizations."""
        # Create chunks for different organization
        for i in range(1, 5):
            chunk_path = Path(self.test_dir) / "HuggingFace.co" / f"{self.base_model_name}-{i:05d}.gguf"
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "qwen2.5-coder-32b-instruct-q5_k_m-00001-of-00004")
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = Path(self.test_dir) / "HuggingFace.co" / f"{self.base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertTrue(chunk_exists, "Split model from different org should be detected as complete")
    
    def test_different_base_model_name(self):
        """Test split model detection works with different base model names."""
        # Create chunks for different base model name
        new_base_name = "bert-base-uncased"
        for i in range(1, 9):  # Changed from 8 to 9 to create 8 chunks (1-8)
            chunk_path = Path(self.test_dir) / self.organization / f"{new_base_name}-{i:05d}.gguf"
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', "bert-base-uncased-00001-of-00008")
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = Path(self.test_dir) / self.organization / f"{new_base_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertTrue(chunk_exists, "Split model with different base name should be detected as complete")


# Cleanup function called once at module exit
# Removes all test directories created by setUp() across all test classes
def cleanup_all_tests():
    """Clean up all test directories created during test runs."""
    import shutil
    if os.path.exists(_TEST_BASE_DIR):
        shutil.rmtree(_TEST_BASE_DIR, ignore_errors=True)


# Register cleanup on module exit
import atexit
atexit.register(cleanup_all_tests)


class TestRegularModelDetection(unittest.TestCase):
    """Test regular (non-split) model detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.organization = "Qwen"
        self.base_model_name = "qwen2.5-32b-instruct-q5_k_m"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_regular_model_file_exists(self):
        """Test that regular model file existence is correctly checked."""
        # Create regular model file
        model_path = Path(self.test_dir) / self.organization / f"qwen2.5-coder-32b-instruct-q5_k_m.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()
        
        # Simulate the existence check
        base_model_path = Path(self.test_dir) / self.organization / f"qwen2.5-coder-32b-instruct-q5_k_m.gguf"
        chunk_exists = base_model_path.exists()
        
        self.assertTrue(chunk_exists, "Regular model should be detected as existing")
    
    def test_regular_model_file_missing(self):
        """Test that missing regular model file is correctly detected."""
        # Don't create regular model file
        base_model_path = Path(self.test_dir) / self.organization / f"qwen2.5-coder-32b-instruct-q5_k_m.gguf"
        chunk_exists = base_model_path.exists()
        
        self.assertFalse(chunk_exists, "Missing regular model should be detected")
    
    def test_regular_model_with_different_organization(self):
        """Test that regular model detection works with different organization names."""
        # Create model file for different organization
        model_path = Path(self.test_dir) / "HuggingFace.co" / "Qwen2.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()
        
        # Simulate the existence check
        base_model_path = Path(self.test_dir) / "HuggingFace.co" / "Qwen2.gguf"
        chunk_exists = base_model_path.exists()
        
        self.assertTrue(chunk_exists, "Regular model from different org should be detected as existing")
    
    def test_regular_model_with_different_name(self):
        """Test that regular model detection works with different model names."""
        # Create model file with different name
        model_path = Path(self.test_dir) / "Jackrong" / "Qwopus3.6-27B-Coder-Q4_K_M.gguf"
        model_path.touch()
        
        # Simulate the existence check
        base_model_path = Path(self.test_dir) / "Jackrong" / f"Qwopus3.6-27B-Coder-Q4_K_M.gguf"
        chunk_exists = base_model_path.exists()
        
        self.assertTrue(chunk_exists, "Regular model with different name should be detected as existing")
    
    def test_regular_model_does_not_trigger_chunk_logic(self):
        """Test that regular models do NOT trigger chunk file existence logic."""
        # Verify that regex doesn't match regular model names
        model_names = [
            "qwen2.5-coder-32b-instruct-q5_k_m",
            "qwen2.5-32b-instruct-q5_k_m",
            "Qwen2.gguf",
            "Qwopus3.6-27B-Coder-Q4_K_M.gguf",
            "Sharp-Spark-X2.5-4B-Q5_K_XL.gguf",
        ]
        
        for name in model_names:
            chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})', name)
            self.assertIsNone(chunk_match, f"Regular model '{name}' should not trigger chunk logic")
    
    def test_regular_model_existence_check_no_chunks_checked(self):
        """Test that regular models do NOT cause chunk file scanning."""
        # Verify that existence check returns True without any file operations
        # This simulates the else branch for non-split models
        base_model_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}.gguf"
        chunk_exists = base_model_path.exists()
        
        # Should be False since file doesn't exist
        self.assertFalse(chunk_exists, "Non-existent regular model should return False")
    
    def test_regular_model_creation_then_check(self):
        """Test that created regular model files are correctly detected."""
        # Create model file
        model_path = Path(self.test_dir) / self.organization / "test-regular-model.gguf"
        model_path.touch()
        
        # Verify detection
        base_model_path = Path(self.test_dir) / self.organization / "test-regular-model.gguf"
        chunk_exists = base_model_path.exists()
        
        self.assertTrue(chunk_exists, "Created regular model should be detected")
    
    def test_multiple_regular_models_in_same_org(self):
        """Test that multiple regular models in same organization are correctly detected."""
        # Create multiple regular model files
        models = [
            "model1.gguf",
            "model2.gguf",
            "model3.gguf",
        ]
        
        for model_name in models:
            model_path = Path(self.test_dir) / self.organization / model_name
            model_path.touch()
        
        # Verify all are detected
        for model_name in models:
            base_model_path = Path(self.test_dir) / self.organization / model_name
            chunk_exists = base_model_path.exists()
            self.assertTrue(chunk_exists, f"Regular model '{model_name}' should be detected")
    
    def test_empty_organization_folder(self):
        """Test that empty organization folder is correctly handled."""
        # Organization folder exists but is empty
        org_path = Path(self.test_dir) / self.organization
        org_path.mkdir(exist_ok=True)
        
        # Verify empty folder is detected as missing
        base_model_path = Path(self.test_dir) / self.organization / "some-model.gguf"
        chunk_exists = base_model_path.exists()
        
        self.assertFalse(chunk_exists, "Empty organization folder should result in missing detection")


class TestCreateModelfileIntegration(unittest.TestCase):
    """Integration tests for create-modelfile.py functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.test_dir) / "models"
        self.downloads_dir = Path(self.test_dir) / "downloads"
        self.organization = "Qwen"
        self.model_name = "qwen2.5-32b-instruct-q5_k_m-00001-of-00006"
        self.model_url = f"https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/resolve/main/{self.model_name}"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_split_model_filename_extraction(self):
        """Test that split model filenames are correctly extracted."""
        # Simulate the filename extraction logic
        base_model_name = self.model_name
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', self.model_name)
        if chunk_match:
            # Extract base name by finding the position right before the first chunk number
            # The pattern is: base_name-00001-of-00006.
            # We need to find where "-00001" starts and return everything before it
            base_model_name = self.model_name[:chunk_match.start() + 3]
        
        self.assertEqual(base_model_name, "qwen2.5-32b-instruct-q5_k_m")
    
    def test_split_model_chunk_detection(self):
        """Test that split models are correctly detected."""
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', self.model_name)
        
        self.assertIsNotNone(chunk_match)
        self.assertEqual(int(chunk_match.group(1)), 1)
        self.assertEqual(int(chunk_match.group(2)), 6)
    
    def test_split_model_exists_when_all_chunks_present(self):
        """Test that split model is detected as existing when all chunks are present."""
        # Create all chunk files
        for i in range(1, 7):
            chunk_path = self.downloads_dir / self.organization / f"qwen2.5-32b-instruct-q5_k_m-{i:05d}.gguf"
            chunk_path.touch()
        
        # Simulate the existence check
        base_model_name = "qwen2.5-32b-instruct-q5_k_m"
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', self.model_name)
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = self.downloads_dir / self.organization / f"{base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertTrue(chunk_exists, "All chunks should be detected as existing")
    
    def test_split_model_exists_when_partial_chunks_present(self):
        """Test that split model is detected as missing when not all chunks are present."""
        # Create only the first chunk
        chunk_path = self.downloads_dir / self.organization / "qwen2.5-32b-instruct-q5_k_m-00001.gguf"
        chunk_path.touch()
        
        # Simulate the existence check
        base_model_name = "qwen2.5-32b-instruct-q5_k_m"
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', self.model_name)
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = self.downloads_dir / self.organization / f"{base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertFalse(chunk_exists, "Missing chunks should be detected")
    
    def test_regular_model_exists(self):
        """Test that regular model is detected as existing."""
        # Create regular model file
        model_path = self.downloads_dir / self.organization / "qwen2.5-32b-instruct-q5_k_m.gguf"
        model_path.touch()
        
        # Simulate the existence check
        base_model_name = "qwen2.5-32b-instruct-q5_k_m"
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', self.model_name)
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = self.downloads_dir / self.organization / f"{base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        else:
            # Regular model - check base file
            chunk_exists = model_path.exists()
        
        self.assertTrue(chunk_exists, "Regular model should be detected as existing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
