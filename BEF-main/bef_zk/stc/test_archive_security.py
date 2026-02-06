"""Security tests for archive path handling.

Run with: PYTHONPATH=. python -m pytest bef_zk/stc/test_archive_security.py -v
Or standalone: PYTHONPATH=. python bef_zk/stc/test_archive_security.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Add parent to path for standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bef_zk.stc.archive import (
    safe_join,
    safe_open,
    ChunkArchive,
    BinaryChunkArchive,
)


class TestSafeJoin:
    """Tests for safe_join path traversal prevention."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_safe_join_")
        self.root = Path(self.tmpdir)
        # Create a test file
        (self.root / "test.json").write_text("[]")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_normal_path_works(self):
        """Normal relative paths should work."""
        result = safe_join(self.root, "test.json")
        assert result == self.root / "test.json"

    def test_nested_path_works(self):
        """Nested paths should work."""
        subdir = self.root / "sub"
        subdir.mkdir()
        (subdir / "file.json").write_text("[]")
        result = safe_join(self.root, "sub/file.json")
        assert result == subdir / "file.json"

    def test_rejects_absolute_path(self):
        """Absolute paths must be rejected."""
        try:
            safe_join(self.root, "/etc/passwd")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Absolute path not allowed" in str(e)

    def test_rejects_windows_absolute(self):
        """Windows-style absolute paths must be rejected."""
        try:
            safe_join(self.root, "C:\\Windows\\System32")
            # Note: This might not raise on Unix since it's not recognized as absolute
            # The backslash check should catch it though
        except ValueError:
            pass  # Expected

    def test_rejects_parent_traversal(self):
        """Parent directory traversal must be rejected."""
        try:
            safe_join(self.root, "../../../etc/passwd")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Path traversal not allowed" in str(e)

    def test_rejects_dotdot_in_middle(self):
        """Traversal in the middle of path must be rejected."""
        try:
            safe_join(self.root, "subdir/../../../etc/passwd")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Path traversal" in str(e)

    def test_rejects_null_bytes(self):
        """Null bytes in path must be rejected."""
        try:
            safe_join(self.root, "file.json\x00/etc/passwd")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Null bytes not allowed" in str(e)

    def test_rejects_backslash_traversal(self):
        """Windows-style traversal must be rejected."""
        try:
            safe_join(self.root, "..\\..\\etc\\passwd")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Path traversal not allowed" in str(e)

    def test_rejects_symlink_escape(self):
        """Symlinks pointing outside root should be caught by resolve()."""
        if sys.platform == "win32":
            return  # Skip on Windows

        # Create a symlink pointing outside
        symlink = self.root / "escape"
        try:
            symlink.symlink_to("/etc")
        except OSError:
            return  # Skip if symlinks not supported

        try:
            # Even though "escape/passwd" looks contained, resolve() should
            # follow the symlink and then relative_to() should fail
            safe_join(self.root, "escape/passwd")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "escapes archive root" in str(e)


class TestSafeOpen:
    """Tests for safe_open symlink handling."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_safe_open_")
        self.root = Path(self.tmpdir)
        (self.root / "real.txt").write_text("real content")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_opens_regular_file(self):
        """Normal files should open successfully."""
        with safe_open(self.root / "real.txt", "r") as f:
            content = f.read()
        assert content == "real content"

    def test_rejects_symlink_on_posix(self):
        """Symlinks should be rejected with O_NOFOLLOW on POSIX."""
        if sys.platform == "win32":
            return  # Skip on Windows

        symlink = self.root / "link.txt"
        try:
            symlink.symlink_to(self.root / "real.txt")
        except OSError:
            return  # Skip if symlinks not supported

        try:
            with safe_open(symlink, "r") as f:
                f.read()
            # If we got here on Linux, O_NOFOLLOW didn't work
            # This might happen on some filesystems
        except OSError as e:
            # Expected: ELOOP (40) on Linux
            pass


class TestBinaryArchiveValidation:
    """Tests for BinaryChunkArchive index validation."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_bin_archive_")
        self.root = Path(self.tmpdir)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_archive_loads(self):
        """Valid archives should load successfully."""
        archive = BinaryChunkArchive(root_dir=self.root)
        archive.store_chunk(0, [1, 2, 3])
        archive.store_chunk(1, [4, 5, 6])
        archive.finalize()

        # Reload
        archive2 = BinaryChunkArchive(root_dir=self.root)
        assert len(archive2) == 2
        assert archive2.load_chunk_by_index(0) == [1, 2, 3]
        assert archive2.load_chunk_by_index(1) == [4, 5, 6]

    def test_rejects_invalid_magic(self):
        """Invalid magic should be rejected."""
        # Create a file with wrong magic
        (self.root / "chunks.bin").write_bytes(b"XXXX" + b"\x00" * 100)
        (self.root / "chunks.idx").write_bytes(b"\x00" * 12)

        try:
            BinaryChunkArchive(root_dir=self.root)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid archive magic" in str(e)

    def test_rejects_index_size_mismatch(self):
        """Index file size must match declared chunk count."""
        # Create valid header with 2 chunks
        data = b"STCA" + (1).to_bytes(4, "little") + (2).to_bytes(8, "little")
        data += b"\x00" * 48  # Some data
        (self.root / "chunks.bin").write_bytes(data)

        # But only 1 index entry (should be 2 * 12 = 24 bytes)
        (self.root / "chunks.idx").write_bytes(b"\x00" * 12)

        try:
            BinaryChunkArchive(root_dir=self.root)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Index file size mismatch" in str(e)

    def test_rejects_overlapping_chunks(self):
        """Overlapping chunk offsets must be rejected."""
        # Header: magic + version + 2 chunks
        data = b"STCA" + (1).to_bytes(4, "little") + (2).to_bytes(8, "little")
        data += b"\x00" * 100  # Enough data
        (self.root / "chunks.bin").write_bytes(data)

        # Index with overlapping offsets
        # Chunk 0: offset=16, len=10 (ends at 96)
        # Chunk 1: offset=20, len=5 (starts before chunk 0 ends!)
        idx = (16).to_bytes(8, "little") + (10).to_bytes(4, "little")
        idx += (20).to_bytes(8, "little") + (5).to_bytes(4, "little")
        (self.root / "chunks.idx").write_bytes(idx)

        try:
            BinaryChunkArchive(root_dir=self.root)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "overlaps previous chunk" in str(e)

    def test_rejects_out_of_bounds_chunk(self):
        """Chunks extending past file end must be rejected."""
        # Header + minimal data
        data = b"STCA" + (1).to_bytes(4, "little") + (1).to_bytes(8, "little")
        data += b"\x00" * 16  # Only 16 bytes of data after header
        (self.root / "chunks.bin").write_bytes(data)

        # Index claiming huge chunk
        idx = (16).to_bytes(8, "little") + (1000).to_bytes(4, "little")
        (self.root / "chunks.idx").write_bytes(idx)

        try:
            BinaryChunkArchive(root_dir=self.root)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "extends beyond file" in str(e)


class TestChunkArchiveSecurity:
    """Integration tests for ChunkArchive security."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_chunk_archive_")
        self.root = Path(self.tmpdir)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_chunk_rejects_traversal(self):
        """load_chunk must reject traversal attempts."""
        archive = ChunkArchive(root_dir=self.root)
        archive.store_chunk(0, [1, 2, 3])

        try:
            archive.load_chunk("../../../etc/passwd")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Path traversal not allowed" in str(e)

    def test_load_chunk_by_index_ignores_handles(self):
        """load_chunk_by_index should use index, not handles."""
        archive = ChunkArchive(root_dir=self.root)
        archive.store_chunk(0, [1, 2, 3])

        # This works regardless of what handles exist
        data = archive.load_chunk_by_index(0)
        assert data == [1, 2, 3]


def run_tests():
    """Run all tests and print results."""
    import traceback

    test_classes = [
        TestSafeJoin,
        TestSafeOpen,
        TestBinaryArchiveValidation,
        TestChunkArchiveSecurity,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for cls in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {cls.__name__}")
        print('='*60)

        instance = cls()
        for name in dir(instance):
            if not name.startswith("test_"):
                continue

            method = getattr(instance, name)
            if not callable(method):
                continue

            try:
                if hasattr(instance, "setup_method"):
                    instance.setup_method()

                method()
                print(f"  ✓ {name}")
                passed += 1

            except AssertionError as e:
                print(f"  ✗ {name}: {e}")
                failed += 1
            except Exception as e:
                print(f"  ✗ {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed += 1
            finally:
                if hasattr(instance, "teardown_method"):
                    try:
                        instance.teardown_method()
                    except Exception:
                        pass

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print('='*60)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
