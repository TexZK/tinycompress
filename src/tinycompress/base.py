# BSD 2-Clause License
#
# Copyright (c) 2025, Andrea Zoppi
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Base classes and utilities for compression/decompression.

This module provides abstract base classes and utility functions for implementing
compression and decompression algorithms. It includes file handling capabilities
and streaming interfaces.
"""

import abc
import io
import os
import sys
from builtins import open as _builtins_open
from typing import IO
from typing import Any
from typing import BinaryIO
from typing import Iterable
from typing import cast as _cast

ByteString = bytes | bytearray | memoryview
ByteIterable = ByteString | Iterable[int]

BUFFER_SIZE = io.DEFAULT_BUFFER_SIZE
"""Compressed data read chunk size."""


class BaseCompressor(abc.ABC):
    """Abstract base class for implementing data compressors.

    This class defines the interface that all compressor classes must implement.
    Compressors process input data in chunks and can be flushed to get any remaining
    compressed data.
    """

    @abc.abstractmethod
    def compress(
            self,
            data: ByteIterable,
    ) -> bytes | bytearray:
        """Compresses the given data.

        Args:
            data: Input data to compress, can be bytes, bytearray, memoryview or
                any iterable of integers in range(256).

        Returns:
            The compressed data as bytes or bytearray.

        Raises:
            Exception: If the compressor has been flushed.
        """
        ...

    @abc.abstractmethod
    def flush(self) -> bytes | bytearray:
        """Flushes any remaining data from the compressor.

        This method finalizes the compression process by returning any remaining
        compressed data that may be buffered internally.

        Returns:
            Any remaining compressed data as bytes or bytearray.

        Raises:
            Exception: If the compressor has already been flushed.
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the compressor to its initial state.

        This allows reusing the same compressor instance for multiple compression tasks
        by clearing any internal state and buffers.
        """
        ...

    @property
    @abc.abstractmethod
    def eof(self) -> bool:
        """Whether the compressor has finished processing all input.

        Returns:
            True if the compressor has been flushed and can no longer accept data,
            False if it can still process more input.
        """
        ...


class BaseDecompressor(abc.ABC):
    """Abstract base class for implementing data decompressors.

    This class defines the interface that all decompressor classes must implement.
    Decompressors process compressed input data in chunks and can be flushed to
    get any remaining decompressed data.
    """

    @abc.abstractmethod
    def decompress(
            self,
            data: ByteIterable,
            max_length: int = -1,
            /,
    ) -> bytes | bytearray:
        """Decompresses the given data.

        Args:
            data: Compressed input data, can be bytes, bytearray, memoryview or
                any iterable of integers in range(256).
            max_length: Maximum number of bytes to decompress. Default -1 means no limit.

        Returns:
            The decompressed data as bytes or bytearray.

        Raises:
            Exception: If the decompressor has been flushed.
        """
        ...

    @abc.abstractmethod
    def flush(self) -> bytes | bytearray:
        """Flushes any pending decompressed data.

        Returns:
            Any remaining decompressed data as bytes or bytearray.

        Raises:
            Exception: If the decompressor has already been flushed.
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the decompressor to its initial state.

        This allows reusing the same decompressor instance for multiple decompression tasks.
        """
        ...

    @property
    @abc.abstractmethod
    def eof(self) -> bool:
        """Whether the decompressor has reached the end of the input stream.

        Returns:
            True if all input has been processed and flushed, False otherwise.
        """
        ...

    @property
    @abc.abstractmethod
    def unused_data(self) -> bytes | bytearray:
        """Data found after the end of the compressed stream.

        Returns:
            Any data found after the end of the compressed stream,
            or empty bytes/bytearray if none.
        """
        ...

    @property
    @abc.abstractmethod
    def needs_input(self) -> bool:
        """Whether the decompressor needs more data to continue.

        Returns:
            True if more input data is needed, False otherwise.
        """
        ...


class DecompressorStream(io.RawIOBase):
    """Raw I/O stream that decompresses data from an underlying binary stream.

    This class wraps a decompressor and binary input stream to create a readable
    stream interface that transparently decompresses data as it is read.

    Args:
        stream: The binary input stream containing compressed data.
        decomp: The decompressor instance to use.
    """

    def __init__(
            self,
            stream: BinaryIO,
            decomp: BaseDecompressor,
    ) -> None:

        super().__init__()

        self._stream = stream
        self._decomp = decomp
        self._tell = 0

    def read(self, size: int | None = -1, /) -> bytes:
        """Read and decompress up to size bytes from the stream.

        Args:
            size: Maximum number of bytes to read and decompress.
                  If None or negative, read until EOF.

        Returns:
            The decompressed data as bytes.
        """
        if size is None or size < 0:
            size = sys.maxsize
        elif not size:
            return b''

        decomp = self._decomp
        stream = self._stream
        inchunk = b''
        outchunk = decomp.decompress(inchunk, size)

        while not outchunk:
            inchunk = stream.read(BUFFER_SIZE)
            outchunk = decomp.decompress(inchunk, size)
            if not inchunk and not outchunk:
                break

        self._tell += len(outchunk)
        return outchunk

    def readable(self) -> bool:
        """Check if the stream is readable.

        Returns:
            True, as DecompressorStream is always readable.
        """
        return True

    def readall(self) -> bytes:
        """Read and decompress all remaining data from the stream.

        Returns:
            All remaining decompressed data as bytes.
        """
        chunks = []
        chunk = self.read()
        while chunk:
            chunks.append(chunk)
            chunk = self.read()
        return b''.join(chunks)

    def readinto(self, buffer: ByteString) -> int:  # type: ignore
        """Read decompressed data into a pre-allocated buffer.

        Args:
            buffer: A writable buffer to store the decompressed data.

        Returns:
            Number of bytes read and stored in the buffer.
        """
        with memoryview(buffer) as view:
            chunk = self.read(len(view))
            view[:len(chunk)] = chunk
        return len(chunk)

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:
        """Change the stream position to the given byte offset.

        Args:
            offset: The offset to seek to, relative to the position specified by whence.
            whence: The reference point for offset.
                   SEEK_SET: Start of stream (default).
                   SEEK_CUR: Current position.
                   SEEK_END: End of stream.

        Returns:
            The new absolute position in the decompressed data stream.

        Raises:
            ValueError: If whence is invalid.

        Note:
            Seeking in a compressed stream requires:
            1. For backwards seeks: Reset to start and decompress up to target
            2. For forward seeks: Read and decompress data until target reached
        """
        offset = offset.__index__()
        if whence == io.SEEK_CUR:
            offset += self._tell
        elif whence == io.SEEK_END:
            chunk = self.read(BUFFER_SIZE)
            while chunk:
                chunk = self.read(BUFFER_SIZE)
            offset += self._tell
        elif whence != io.SEEK_SET:
            raise ValueError(f'invalid whence: {whence!r}')

        if offset < self._tell:
            self._stream.seek(0)
            self._tell = 0
            self._decomp.reset()
        else:
            offset -= self._tell

        while offset:
            chunk = self.read(offset if offset < BUFFER_SIZE else BUFFER_SIZE)
            offset -= len(chunk) if chunk else offset

        return self._tell

    def seekable(self) -> bool:
        """Check if the stream supports seeking.

        Returns:
            True if the underlying stream is seekable, False otherwise.
        """
        return self._stream.seekable()

    def tell(self) -> int:
        """Get the current stream position in the decompressed data.

        Returns:
            Current position as number of bytes from start of decompressed data.
        """
        return self._tell


class CodecFile(io.BufferedIOBase):
    """File-like object that handles compression/decompression.

    This class provides a buffered I/O interface for reading compressed files
    or writing compressed output, similar to Python's built-in file objects.

    Args:
        filename: Path to the file, or an existing file object.
        mode: File open mode, similar to built-in open().
              'r' or 'rb' for reading, 'w'/'wb'/'x'/'xb'/'a'/'ab' for writing.
        comp: Compressor instance for writing compressed data. Required for write modes.
        decomp: Decompressor instance for reading compressed data. Required for read modes.

    Raises:
        ValueError: If mode is invalid or required codec is missing.
        TypeError: If filename is not a str, bytes, file, or PathLike object.
    """

    def __init__(
            self,
            filename: str | bytes | os.PathLike | IO,
            mode: str = 'r',
            comp: BaseCompressor | None = None,
            decomp: BaseDecompressor | None = None,
    ) -> None:

        if mode in ('', 'r', 'rb'):
            way = -1
        elif mode in ('w', 'wb', 'x', 'xb', 'a', 'ab'):
            way = +1
        else:
            raise ValueError(f'invalid mode: {mode!r}')

        if way < 0:
            if decomp is None:
                raise ValueError('decompressor object required')
        else:
            if comp is None:
                raise ValueError('compressor object required')

        reader = io.BufferedReader(io.BytesIO())  # dummy

        if isinstance(filename, (str, bytes, os.PathLike)):
            direct = True
            stream = _builtins_open(filename, mode=mode)

        elif hasattr(filename, 'read') or hasattr(filename, 'write'):
            direct = False
            stream = _cast(BinaryIO, filename)
            if way < 0:
                assert decomp is not None
                reader = io.BufferedReader(DecompressorStream(stream, decomp))
        else:
            raise TypeError('filename must be a str, bytes, file, or PathLike object')

        self._way = way
        self._direct = direct
        self._stream = stream
        self._reader = reader
        self._comp = comp
        self._decomp = decomp
        self._tell = 0

    def _check_open(self) -> None:
        """Verify that the file is open.

        Raises:
            ValueError: If the file has been closed.
        """
        if self.closed:
            raise ValueError('closed')

    def _check_readable(self) -> None:
        """Verify that the file is readable.

        Raises:
            UnsupportedOperation: If the file was not opened in read mode.
        """
        if not self.readable():
            raise io.UnsupportedOperation('not readable')

    def _check_seekable(self) -> None:
        """Verify that the file supports seeking.

        Raises:
            UnsupportedOperation: If the file does not support seeking operations
                (e.g., not readable or underlying stream not seekable).
        """
        if not self.seekable():
            raise io.UnsupportedOperation('not seekable')

    def _check_writable(self) -> None:
        """Verify that the file is writable.

        Raises:
            UnsupportedOperation: If the file was not opened in write mode.
        """
        if not self.writable():
            raise io.UnsupportedOperation('not writable')

    def close(self) -> None:
        """Close the file and flush any unwritten compressed data.

        If opened for writing, this will flush the compressor and write any
        remaining compressed data to the underlying stream before closing.
        """
        if self._way:
            try:
                if self._way < 0:
                    self._reader.close()
                else:
                    assert self._comp is not None
                    chunk = self._comp.flush()
                    self._stream.write(chunk)
            finally:
                try:
                    if self._direct:
                        self._stream.close()
                finally:
                    self._way = 0

    @property
    def closed(self) -> bool:
        """Check if the file is closed.

        Returns:
            True if the file is closed, False if it is still open.
        """
        return not self._way

    def fileno(self) -> int:
        """Get the file descriptor number of the underlying stream.

        Returns:
            File descriptor number as an integer.

        Raises:
            UnsupportedOperation: If the underlying stream does not support fileno.
        """
        return self._stream.fileno()

    def read(self, size: int | None = -1, /) -> bytes:
        """Read and decompress up to size bytes from the file.

        Args:
            size: Maximum number of bytes to read. If None or negative, read until EOF.

        Returns:
            Decompressed data as bytes.

        Raises:
            ValueError: If file is closed.
            UnsupportedOperation: If file is not readable.
        """
        self._check_readable()
        return self._reader.read(size)

    def read1(self, size: int = -1, /) -> bytes:
        """Read up to size decompressed bytes, using at most one call to the underlying stream.

        Args:
            size: Maximum number of bytes to read. If negative, read until EOF.

        Returns:
            Decompressed data as bytes.

        Raises:
            ValueError: If file is closed.
            UnsupportedOperation: If file is not readable.
        """
        self._check_readable()
        if size < 0:
            size = BUFFER_SIZE
        return self._reader.read1(size)

    def readable(self) -> bool:
        """Check if the file was opened for reading.

        Returns:
            True if file is readable, False otherwise.

        Raises:
            ValueError: If file is closed.
        """
        self._check_open()
        return self._way < 0

    def readall(self) -> bytes:
        """Read and decompress the entire contents of the file.

        Returns:
            All decompressed data from the file as bytes.
        """
        return self._reader.read()

    def readline(self, size: int | None = -1, /) -> bytes:
        """Read and decompress a single line from the file.

        Args:
            size: Maximum number of bytes to read. If None or negative, read entire line.

        Returns:
            The next line from the file, including trailing newline if present.

        Raises:
            ValueError: If file is closed.
            UnsupportedOperation: If file is not readable.
        """
        self._check_readable()
        return self._reader.readline(size)

    def readlines(self, hint: int | None = -1, /) -> list[bytes]:
        """Read and decompress all remaining lines from the file.

        Args:
            hint: Maximum number of bytes to read. If None or negative, read all lines.
                 This is a hint, not a strict limit.

        Returns:
            List of lines from the file, each including trailing newline if present.

        Raises:
            ValueError: If file is closed.
            UnsupportedOperation: If file is not readable.
        """
        self._check_readable()
        hint = -1 if hint is None else hint.__index__()
        return self._reader.readlines(hint)

    def readinto(self, buffer: Any, /) -> int:
        """Read decompressed data directly into a pre-allocated buffer.

        Args:
            buffer: A writable buffer to store the decompressed data.

        Returns:
            Number of bytes read and stored in the buffer.

        Raises:
            ValueError: If file is closed.
            UnsupportedOperation: If file is not readable.
        """
        self._check_readable()
        return self._reader.readinto(buffer)

    def readinto1(self, buffer: Any, /) -> int:
        """Read decompressed data into a buffer, using at most one underlying read.

        Args:
            buffer: A writable buffer to store the decompressed data.

        Returns:
            Number of bytes read and stored in the buffer.

        Note:
            This implementation is equivalent to readinto() since the underlying
            buffered reader already handles read buffering.
        """
        return self.readinto(buffer)

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:
        """Change the stream position to the given offset.

        Args:
            offset: The offset to seek to, relative to the position specified by whence.
            whence: The reference point for offset.
                   SEEK_SET: Start of stream (default).
                   SEEK_CUR: Current position.
                   SEEK_END: End of stream.

        Returns:
            The new absolute position.

        Raises:
            ValueError: If file is closed.
            UnsupportedOperation: If file is not seekable.
            ValueError: If whence is invalid.
        """
        self._check_seekable()
        return self._reader.seek(offset, whence)

    def seekable(self) -> bool:
        """Check if the file supports seeking.

        Returns:
            True if the file is readable and the underlying stream is seekable,
            False otherwise.
        """
        return self.readable() and self._reader.seekable()

    def tell(self) -> int:
        """Get the current file position.

        Returns:
            Current position as number of bytes from start.
            For reading, returns position in decompressed data.
            For writing, returns number of bytes written.

        Raises:
            ValueError: If file is closed.
        """
        self._check_open()
        return self._reader.tell() if self._way < 0 else self._tell

    def writable(self) -> bool:
        """Check if the file was opened for writing.

        Returns:
            True if file is writable, False otherwise.

        Raises:
            ValueError: If file is closed.
        """
        self._check_open()
        return self._way > 0

    def write(self, buffer: ByteString, /) -> int:  # type: ignore override
        """Write and compress the contents of the buffer.

        Args:
            buffer: Data to be compressed and written.

        Returns:
            Number of uncompressed bytes written.

        Raises:
            ValueError: If file is closed.
            UnsupportedOperation: If file is not writable.
        """
        self._check_writable()
        with memoryview(buffer) as view:
            assert self._comp is not None
            chunk = self._comp.compress(view)
            self._stream.write(chunk)
            size = len(view)
            self._tell += size
            return size


def codec_compress(data: ByteIterable, comp: BaseCompressor) -> bytes:
    """Compresses data using the given compressor.

    Args:
        data: Data to compress.
        comp: Compressor instance to use.

    Returns:
        The complete compressed data as bytes.
    """
    out = comp.compress(data)
    out += comp.flush()
    return out


def codec_decompress(data: ByteIterable, decomp: BaseDecompressor) -> bytes:
    """Decompresses data using the given decompressor.

    Args:
        data: Compressed data to decompress.
        decomp: Decompressor instance to use.

    Returns:
        The complete decompressed data as bytes.
    """
    out = decomp.decompress(data)
    out += decomp.flush()
    return out


def codec_open(
        filename: str | bytes | IO,
        mode: str = 'r',
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        comp: BaseCompressor | None = None,
        decomp: BaseDecompressor | None = None,
) -> CodecFile | io.TextIOWrapper:
    """Opens a file for reading/writing with compression/decompression.

    Similar to the built-in open() function, but handles compressed data.

    Args:
        filename: Path to file or file object.
        mode: File open mode, similar to built-in open().
        encoding: Text encoding for text mode.
        errors: How to handle encoding/decoding errors in text mode.
        newline: How to handle newlines in text mode.
        comp: Compressor instance for writing. Required for write modes.
        decomp: Decompressor instance for reading. Required for read modes.

    Returns:
        A CodecFile for binary mode or TextIOWrapper for text mode.

    Raises:
        ValueError: For invalid mode combinations or missing codecs.
    """

    if 't' in mode:
        if 'b' in mode:
            raise ValueError(f'invalid mode: {mode!r}')
    else:
        if encoding is not None:
            raise ValueError("argument 'encoding' not supported in binary mode")
        if errors is not None:
            raise ValueError("argument 'errors' not supported in binary mode")
        if newline is not None:
            raise ValueError("argument 'newline' not supported in binary mode")

    mode_ = mode.replace('t', '')
    file = CodecFile(filename, mode=mode_, comp=comp, decomp=decomp)

    if 't' in mode:
        encoding = io.text_encoding(encoding)
        return io.TextIOWrapper(file, encoding=encoding, errors=errors, newline=newline)  # type: ignore
    else:
        return file
