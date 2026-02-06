"""Simple chunk archive for streaming STC accumulators.

Security Notes:
    - All path operations use safe_join() to prevent traversal attacks
    - On POSIX systems, safe_open() uses O_NOFOLLOW to prevent symlink TOCTOU races
    - Archive directories should not be shared or attacker-writable during verification
    - Binary archives validate index bounds to prevent seek bombs
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import IO, List, Sequence

# O_NOFOLLOW is POSIX-only; on Windows we fall back to regular open
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_IS_POSIX = sys.platform != "win32"


def safe_join(root: Path, rel: str) -> Path:
    """Safely join a relative path to a root, rejecting path traversal attacks.

    Raises:
        ValueError: If the path escapes the root directory or contains unsafe components.
    """
    # Reject obviously malicious patterns upfront
    if rel.startswith("/") or rel.startswith("\\"):
        raise ValueError(f"Absolute path not allowed: {rel}")
    if ".." in rel.split("/") or ".." in rel.split("\\"):
        raise ValueError(f"Path traversal not allowed: {rel}")
    if "\x00" in rel:
        raise ValueError(f"Null bytes not allowed in path: {rel}")

    # Resolve and verify containment
    root_resolved = root.resolve()
    candidate = (root_resolved / rel).resolve()

    # Check that resolved path is actually under root
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"Path escapes archive root: {rel}")

    return candidate


def safe_open(path: Path, mode: str = "r") -> IO:
    """Open a file with O_NOFOLLOW to prevent symlink TOCTOU attacks.

    On POSIX systems, this uses O_NOFOLLOW to refuse to follow symlinks.
    On Windows, falls back to regular open (symlink attacks less common).

    Raises:
        OSError: If the path is a symlink (ELOOP on Linux)
        FileNotFoundError: If the file doesn't exist
    """
    if _IS_POSIX and _O_NOFOLLOW:
        # Use low-level open with O_NOFOLLOW
        flags = os.O_RDONLY | os.O_CLOEXEC | _O_NOFOLLOW
        if "b" not in mode:
            # Text mode needs additional handling
            fd = os.open(str(path), flags)
            return os.fdopen(fd, mode)
        else:
            fd = os.open(str(path), flags)
            return os.fdopen(fd, mode)
    else:
        # Fallback for Windows or systems without O_NOFOLLOW
        return open(path, mode)


class ChunkArchive:
    """Disk-backed storage for chunk values (optional persistent root)."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="stc_chunks_")
            self.root = Path(self._tempdir.name)
            self._persistent = False
        else:
            self.root = Path(root_dir)
            self.root.mkdir(parents=True, exist_ok=True)
            self._tempdir = None
            self._persistent = True

    def store_chunk(self, chunk_index: int, values: Sequence[int]) -> str:
        path = self.root / f"chunk_{chunk_index}.json"
        path.write_text(json.dumps([int(v) for v in values]))
        rel = path.relative_to(self.root)
        return rel.as_posix()

    def load_chunk(self, handle: str) -> List[int]:
        """Load chunk by handle with path confinement.

        The handle is confined to the archive root directory. Absolute paths
        and path traversal attempts are rejected with ValueError.

        Uses O_NOFOLLOW on POSIX to prevent symlink TOCTOU attacks.

        Raises:
            ValueError: If handle attempts path traversal or escapes root.
            FileNotFoundError: If the chunk file doesn't exist.
            OSError: If the path is a symlink (on POSIX).
        """
        # Use safe_join to confine path to archive root
        path = safe_join(self.root, handle)
        with safe_open(path, "r") as f:
            return [int(v) for v in json.loads(f.read())]

    def load_chunk_by_index(self, chunk_index: int) -> List[int]:
        """Load chunk by numeric index (safest method - ignores prover handles).

        This is the preferred method when the chunk index is known, as it
        completely ignores any prover-supplied path information.

        Uses O_NOFOLLOW on POSIX to prevent symlink TOCTOU attacks.

        Raises:
            FileNotFoundError: If the chunk file doesn't exist.
            OSError: If the path is a symlink (on POSIX).
        """
        path = self.root / f"chunk_{chunk_index}.json"
        with safe_open(path, "r") as f:
            return [int(v) for v in json.loads(f.read())]

    def cleanup(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()

    def write_chunk(self, chunk_index: int, values: Sequence[int]) -> None:
        """Alias for store_chunk for interface compatibility."""
        self.store_chunk(chunk_index, values)


# Binary archive constants
_BINARY_MAGIC = b"STCA"  # STC Archive
_BINARY_VERSION = 1
_HEADER_SIZE = 16  # 4 (magic) + 4 (version) + 8 (num_chunks)
_INDEX_ENTRY_SIZE = 12  # 8 (offset) + 4 (length)


class BinaryChunkArchive:
    """High-performance binary archive for chunk storage.

    Uses a single binary file for data and a separate index file,
    eliminating filesystem overhead from many small JSON files.

    File format:
        chunks.bin:
            [4 bytes: magic "STCA"]
            [4 bytes: version (1)]
            [8 bytes: num_chunks as u64 LE]
            [chunk_0 data: len_0 * 8 bytes of u64 LE values]
            [chunk_1 data: len_1 * 8 bytes of u64 LE values]
            ...

        chunks.idx:
            [12 bytes per chunk: offset_u64_LE + length_u32_LE]
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="stc_chunks_bin_")
            self.root = Path(self._tempdir.name)
            self._persistent = False
        else:
            self.root = Path(root_dir)
            self.root.mkdir(parents=True, exist_ok=True)
            self._tempdir = None
            self._persistent = True

        self._data_path = self.root / "chunks.bin"
        self._index_path = self.root / "chunks.idx"
        self._num_chunks = 0
        self._current_offset = _HEADER_SIZE
        self._index: List[tuple[int, int]] = []  # (offset, length)
        self._data_file = None
        self._index_file = None
        self._read_handle = None
        self._finalized = False

        # Check if we're reopening an existing archive
        if self._data_path.exists() and self._index_path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        """Load index from existing archive (read-only mode).

        Validates index integrity to prevent DoS via malformed archives:
        - Offsets must be monotonically increasing
        - All data must be within file bounds
        - Chunk counts and lengths must be reasonable
        """
        # Get file size for bounds checking
        data_file_size = self._data_path.stat().st_size
        index_file_size = self._index_path.stat().st_size

        with open(self._data_path, "rb") as f:
            magic = f.read(4)
            if magic != _BINARY_MAGIC:
                raise ValueError(f"Invalid archive magic: {magic!r}")
            version = int.from_bytes(f.read(4), "little")
            if version != _BINARY_VERSION:
                raise ValueError(f"Unsupported archive version: {version}")
            self._num_chunks = int.from_bytes(f.read(8), "little")

        # Sanity check: num_chunks should match index file size
        expected_index_size = self._num_chunks * _INDEX_ENTRY_SIZE
        if index_file_size != expected_index_size:
            raise ValueError(
                f"Index file size mismatch: expected {expected_index_size}, got {index_file_size}"
            )

        # Sanity limit: prevent memory exhaustion (100M chunks = 1.2GB index)
        if self._num_chunks > 100_000_000:
            raise ValueError(f"Archive has too many chunks: {self._num_chunks}")

        # Load and validate index
        prev_end = _HEADER_SIZE
        with open(self._index_path, "rb") as f:
            for i in range(self._num_chunks):
                offset = int.from_bytes(f.read(8), "little")
                length = int.from_bytes(f.read(4), "little")

                # Validate monotonicity: offset must be >= previous end
                if offset < prev_end:
                    raise ValueError(
                        f"Chunk {i} offset {offset} overlaps previous chunk ending at {prev_end}"
                    )

                # Validate bounds: data must be within file
                chunk_end = offset + length * 8
                if chunk_end > data_file_size:
                    raise ValueError(
                        f"Chunk {i} extends beyond file: offset={offset}, len={length}, "
                        f"end={chunk_end}, file_size={data_file_size}"
                    )

                # Sanity limit: individual chunk length (16M values = 128MB)
                if length > 16_000_000:
                    raise ValueError(f"Chunk {i} too large: {length} values")

                self._index.append((offset, length))
                prev_end = chunk_end

        if self._index:
            last_offset, last_len = self._index[-1]
            self._current_offset = last_offset + last_len * 8

        self._finalized = True

    def _ensure_open(self) -> None:
        """Ensure files are open for writing."""
        if self._finalized:
            raise RuntimeError("Archive is finalized; cannot write more chunks")
        if self._data_file is None:
            self._data_file = open(self._data_path, "wb")
            # Write header with placeholder num_chunks
            self._data_file.write(_BINARY_MAGIC)
            self._data_file.write(_BINARY_VERSION.to_bytes(4, "little"))
            self._data_file.write((0).to_bytes(8, "little"))  # placeholder
            self._index_file = open(self._index_path, "wb")

    def store_chunk(self, chunk_index: int, values: Sequence[int]) -> str:
        """Store chunk and return a handle (for interface compatibility)."""
        self._ensure_open()

        if chunk_index != self._num_chunks:
            raise ValueError(f"Expected chunk {self._num_chunks}, got {chunk_index}")

        offset = self._current_offset
        length = len(values)

        # Write data
        for v in values:
            self._data_file.write(int(v).to_bytes(8, "little"))

        # Write index entry
        self._index_file.write(offset.to_bytes(8, "little"))
        self._index_file.write(length.to_bytes(4, "little"))

        self._index.append((offset, length))
        self._current_offset += length * 8
        self._num_chunks += 1

        return f"bin:{chunk_index}"

    def write_chunk(self, chunk_index: int, values: Sequence[int]) -> None:
        """Alias for store_chunk for interface compatibility."""
        self.store_chunk(chunk_index, values)

    def finalize(self) -> None:
        """Finalize the archive (update header with final chunk count)."""
        if self._data_file is not None:
            # Seek back and update num_chunks in header
            self._data_file.seek(8)  # After magic + version
            self._data_file.write(self._num_chunks.to_bytes(8, "little"))
            self._data_file.close()
            self._data_file = None
        if self._index_file is not None:
            self._index_file.close()
            self._index_file = None
        self._finalized = True

    def load_chunk(self, handle: str) -> List[int]:
        """Load chunk by handle (for interface compatibility)."""
        if handle.startswith("bin:"):
            chunk_index = int(handle[4:])
            return self.load_chunk_by_index(chunk_index)
        # Fallback: try parsing as plain index
        try:
            chunk_index = int(handle)
            return self.load_chunk_by_index(chunk_index)
        except ValueError:
            raise ValueError(f"Invalid binary archive handle: {handle}")

    def load_chunk_by_index(self, chunk_index: int) -> List[int]:
        """Load chunk by numeric index (efficient random access)."""
        if chunk_index < 0 or chunk_index >= len(self._index):
            raise IndexError(f"Chunk index {chunk_index} out of range [0, {len(self._index)})")

        offset, length = self._index[chunk_index]

        # Use cached reader if available
        if self._read_handle is None:
            self._read_handle = open(self._data_path, "rb")

        self._read_handle.seek(offset)
        data = self._read_handle.read(length * 8)

        # Use struct for faster unpacking
        import struct
        fmt = f"<{length}Q"  # Little-endian unsigned 64-bit
        return list(struct.unpack(fmt, data))

    def close_reader(self) -> None:
        """Close the cached read handle."""
        if hasattr(self, "_read_handle") and self._read_handle is not None:
            self._read_handle.close()
            self._read_handle = None

    def __len__(self) -> int:
        return self._num_chunks

    def cleanup(self) -> None:
        """Clean up resources."""
        self.close_reader()
        self.finalize()
        if self._tempdir is not None:
            self._tempdir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        return False


__all__ = ["ChunkArchive", "BinaryChunkArchive", "safe_join", "safe_open"]
