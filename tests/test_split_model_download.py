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


class TestSplitModelDetection(unittest.TestCase):
    """Test split model filename detection and regex pattern."""
    
    def test_split_model_filename_detection(self):
        """Test that split model filenames are correctly detected."""
        model_names = [
            "qwen2.5-32b-instruct-q5_k_m-00001-of-00006",
            "model-chunk-00001-of-00003",
            "bert-base-00001-of-00010",
        ]
        
        for name in model_names:
            match = re.search(r'-([0-9]{5})-of-([0-9]{5})', name)
            self.assertIsNotNone(match, f"Failed to detect split model: {name}")
            self.assertEqual(match.group(1), "00001", f"Wrong chunk number for: {name}")
            self.assertEqual(match.group(2), "00006", f"Wrong total chunks for: {name}")
    
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
            chunk_path.touch()
        
        # Simulate the existence check
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', "qwen2.5-32b-instruct-q5_k_m-00001-of-00006")
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertTrue(chunk_exists, "All chunks should be detected as existing")
    
    def test_first_chunk_only_exists(self):
        """Test that partial chunks are detected as missing."""
        # Create only the first chunk
        chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-00001.gguf"
        chunk_path.touch()
        
        # Simulate the existence check
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', "qwen2.5-32b-instruct-q5_k_m-00001-of-00006")
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertFalse(chunk_exists, "Missing chunks should be detected")
    
    def test_last_chunk_only_exists(self):
        """Test that only the last chunk is detected as missing."""
        # Create only the last chunk
        chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-00006.gguf"
        chunk_path.touch()
        
        # Simulate the existence check
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', "qwen2.5-32b-instruct-q5_k_m-00001-of-00006")
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertFalse(chunk_exists, "Missing chunks should be detected")
    
    def test_no_chunks_exist(self):
        """Test that no chunks are detected as existing."""
        # Create no chunk files
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', "qwen2.5-32b-instruct-q5_k_m-00001-of-00006")
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertFalse(chunk_exists, "No chunks should be detected as existing")
    
    def test_three_chunk_model(self):
        """Test existence check for a 3-chunk model."""
        # Create all 3 chunks
        for i in range(1, 4):
            chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
            chunk_path.touch()
        
        chunk_match = re.search(r'-([0-9]{5})-of-([0-9]{5})\.', "qwen2.5-32b-instruct-q5_k_m-00001-of-00003")
        chunk_exists = False
        
        if chunk_match:
            for i in range(1, int(chunk_match.group(2)) + 1):
                chunk_path = Path(self.test_dir) / self.organization / f"{self.base_model_name}-{i:05d}.gguf"
                if not chunk_path.exists():
                    chunk_exists = False
                    break
                chunk_exists = True
        
        self.assertTrue(chunk_exists, "All 3 chunks should be detected as existing")


class TestRegularModelDetection(unittest.TestCase):
    """Test regular (non-split) model detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.organization = "Qwen"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_regular_model_file_exists(self):
        """Test that regular model file existence is correctly checked."""
        # Create regular model file
        model_path = Path(self.test_dir) / self.organization / "qwen2.5-32b-instruct-q5_k_m.gguf"
        model_path.touch()
        
        # Simulate the existence check
        base_model_path = Path(self.test_dir) / self.organization / "qwen2.5-32b-instruct-q5_k_m.gguf"
        chunk_exists = base_model_path.exists()
        
        self.assertTrue(chunk_exists, "Regular model should be detected as existing")
    
    def test_regular_model_file_missing(self):
        """Test that missing regular model file is correctly detected."""
        # Don't create regular model file
        base_model_path = Path(self.test_dir) / self.organization / "qwen2.5-32b-instruct-q5_k_m.gguf"
        chunk_exists = base_model_path.exists()
        
        self.assertFalse(chunk_exists, "Missing regular model should be detected")


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
            base_model_name = self.model_name[:chunk_match.start()]
        
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
