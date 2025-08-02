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

import abc
import io
import os
import sys
from builtins import open as _builtins_open
from typing import Any
from typing import BinaryIO
from typing import Iterable
from typing import IO
from typing import List
from typing import Optional
from typing import Union
from typing import cast as _cast

ByteString = Union[bytes, bytearray, memoryview]
ByteIterable = Union[ByteString, Iterable[int]]

BUFFER_SIZE = io.DEFAULT_BUFFER_SIZE  # Compressed data read chunk size


class BaseCompressor(abc.ABC):

    @abc.abstractmethod
    def compress(
            self,
            data: ByteIterable,
    ) -> Union[bytes, bytearray]:
        ...

    @abc.abstractmethod
    def flush(self) -> Union[bytes, bytearray]:
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def eof(self) -> bool:
        ...


class BaseDecompressor(abc.ABC):

    @abc.abstractmethod
    def decompress(self,
            data: ByteIterable,
            max_length: int = -1,
            /,
    ) -> Union[bytes, bytearray]:
        ...

    @abc.abstractmethod
    def flush(self) -> Union[bytes, bytearray]:
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def eof(self) -> bool:
        ...

    @property
    @abc.abstractmethod
    def unused_data(self) -> Union[bytes, bytearray]:
        ...

    @property
    @abc.abstractmethod
    def needs_input(self) -> bool:
        ...


class DecompressorStream(io.RawIOBase):

    def __init__(
            self,
            stream: BinaryIO,
            decomp: BaseDecompressor,
    ) -> None:

        super().__init__()

        self._stream = stream
        self._decomp = decomp
        self._tell = 0

    def read(self, size: Optional[int] = -1, /) -> bytes:

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

        return True

    def readall(self) -> bytes:

        chunks = []
        chunk = self.read()
        while chunk:
            chunks.append(chunk)
            chunk = self.read()
        return b''.join(chunks)

    def readinto(self, buffer: ByteString) -> int:  # type: ignore

        with memoryview(buffer) as view:
            chunk = self.read(len(view))
            view[:len(chunk)] = chunk
        return len(chunk)

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:

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

        return self._stream.seekable()

    def tell(self) -> int:

        return self._tell


class CodecFile(io.BufferedIOBase):

    def __init__(
            self,
            filename: Union[str, bytes, os.PathLike, IO],
            mode: str = 'r',
            comp: Optional[BaseCompressor] = None,
            decomp: Optional[BaseDecompressor] = None,
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

        if self.closed:
            raise ValueError('closed')

    def _check_readable(self) -> None:

        if not self.readable():
            raise io.UnsupportedOperation('not readable')

    def _check_seekable(self) -> None:

        if not self.seekable():
            raise io.UnsupportedOperation('not seekable')

    def _check_writable(self) -> None:

        if not self.writable():
            raise io.UnsupportedOperation('not writable')

    def close(self) -> None:

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

        return not self._way

    def fileno(self) -> int:

        return self._stream.fileno()

    def read(self, size: Optional[int] = -1, /) -> bytes:

        self._check_readable()
        return self._reader.read(size)

    def read1(self, size: int = -1, /) -> bytes:

        self._check_readable()
        if size < 0:
            size = BUFFER_SIZE
        return self._reader.read1(size)

    def readable(self) -> bool:

        self._check_open()
        return self._way < 0

    def readall(self) -> bytes:

        return self._reader.read()

    def readline(self, size: Optional[int] = -1, /) -> bytes:

        self._check_readable()
        return self._reader.readline(size)

    def readlines(self, hint: Optional[int] = -1, /) -> List[bytes]:

        self._check_readable()
        hint = -1 if hint is None else hint.__index__()
        return self._reader.readlines(hint)

    def readinto(self, buffer: Any, /) -> int:

        self._check_readable()
        return self._reader.readinto(buffer)

    def readinto1(self, buffer: Any, /) -> int:

        return self.readinto(buffer)

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:

        self._check_seekable()
        return self._reader.seek(offset, whence)

    def seekable(self) -> bool:

        return self.readable() and self._reader.seekable()

    def tell(self) -> int:

        self._check_open()
        return self._reader.tell() if self._way < 0 else self._tell

    def writable(self) -> bool:

        self._check_open()
        return self._way > 0

    def write(self, buffer: ByteString, /) -> int:  # type: ignore override

        self._check_writable()
        with memoryview(buffer) as view:
            assert self._comp is not None
            chunk = self._comp.compress(view)
            self._stream.write(chunk)
            size = len(view)
            self._tell += size
            return size


def codec_compress(data: ByteIterable, comp: BaseCompressor) -> bytes:

    out = comp.compress(data)
    out += comp.flush()
    return out


def codec_decompress(data: ByteIterable, decomp: BaseDecompressor) -> bytes:

    out = decomp.decompress(data)
    out += decomp.flush()
    return out


def codec_open(
        filename: Union[str, bytes, IO],
        mode: str = 'r',
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        comp: Optional[BaseCompressor] = None,
        decomp: Optional[BaseDecompressor] = None,
) -> Union[CodecFile, io.TextIOWrapper]:

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
