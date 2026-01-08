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
#
#
# Based on the original C implementation:
#
# /**************************************************************
#     LZSS.C -- A Data Compression Program
#     (tab = 4 spaces)
# ***************************************************************
#     4/6/1989 Haruhiko Okumura
#     Use, distribute, and modify this program freely.
#     Please send me your improved versions.
#         PC-VAN      SCIENCE
#         NIFTY-Serve PAF01022
#         CompuServe  74050,1022
# **************************************************************/

"""Lempel-Ziv-Storer-Szymanski via Words compression algorithm.

This module implements the LZSSW compression algorithm, a variant of LZSS.
The algorithm maintains a ring buffer and uses bit flags to indicate compressed
vs uncompressed data.

The compressor parameters can be tuned for different use cases:
- Ring buffer size controls how far back to look for matches
- Maximum match length affects compression ratio vs processing time
- Common byte value can be optimized for specific data types
"""

import argparse
import io
import math
import os
import sys
from typing import IO
from typing import Generator

from .base import BaseCompressor
from .base import BaseDecompressor
from .base import ByteIterable
from .base import CodecFile
from .base import codec_compress
from .base import codec_decompress
from .base import codec_open

_CodecGenerator = Generator[None, (int | type(Ellipsis) | None), None]

RING_SIZE_MIN = 0x0200
"""Minimum ring buffer size."""

RING_SIZE_MAX = 0x1000
"""Maximum ring buffer size."""

RING_SIZE_DEF = RING_SIZE_MAX
"""Default uses maximum size."""

MAX_MATCH_LEN_MIN = 0x10
"""Minimum match length limit."""

MAX_MATCH_LEN_MAX = 0x80
"""Maximum match length limit."""

MAX_MATCH_LEN_DEF = MAX_MATCH_LEN_MIN
"""Default match length."""

COMMON_BYTE_DEF = 0x20
"""Default byte value used to fill the initial ring buffer."""


class LZSSWException(Exception):
    """Exception raised for LZSSW compression/decompression errors."""
    pass


class LZSSWCompressor(BaseCompressor):
    """LZSSW compression implementation.

    This compressor implements the LZSSW algorithm using a sliding window approach.
    It maintains a ring buffer and binary tree structure to efficiently find repeated
    sequences in previously processed data.

    The compressor can be tuned via three main parameters:
    - Ring buffer size: Controls look-back distance for finding matches
    - Maximum match length: Limits how long matched sequences can be
    - Common byte: Value used to initialize the ring buffer
    """

    def __init__(
            self,
            ringsize: int = RING_SIZE_DEF,
            maxmatchlen: int = MAX_MATCH_LEN_DEF,
            commonbyte: int = COMMON_BYTE_DEF,
    ) -> None:
        """Initializes a new LZSSW compressor instance.

        Args:
            ringsize: Size of the ring buffer, must be between RING_SIZE_MIN and
                     RING_SIZE_MAX. Larger sizes allow finding matches further back.
            maxmatchlen: Maximum length of matched sequences, between MAX_MATCH_LEN_MIN
                        and MAX_MATCH_LEN_MAX. Larger values can improve compression
                        but increase processing time.
            commonbyte: Value used to fill the initial ring buffer, must be 0-255.
                       Should be set to a commonly occurring byte value in the input.

        Raises:
            ValueError: If any parameters are out of their valid ranges.
        """
        ringsize = ringsize.__index__()
        maxmatchlen = maxmatchlen.__index__()
        commonbyte = commonbyte.__index__()
        ringbits = math.ceil(math.log2(ringsize))
        matchbits = math.ceil(math.log2(maxmatchlen))

        if not RING_SIZE_MIN <= ringsize <= RING_SIZE_MAX:
            raise ValueError(f'not ({RING_SIZE_MIN = }) '
                             f'<= ({ringsize = }) '
                             f'<= ({RING_SIZE_MAX = })')

        if not MAX_MATCH_LEN_MIN <= maxmatchlen <= MAX_MATCH_LEN_MAX:
            raise ValueError(f'not ({MAX_MATCH_LEN_MIN = }) '
                             f'<= ({maxmatchlen = }) '
                             f'<= ({MAX_MATCH_LEN_MAX = })')

        if ringbits + matchbits > 16:
            raise ValueError(f'({ringbits = }) + ({matchbits = }) > 16')

        if not 0x00 <= commonbyte <= 0xFF:
            raise ValueError(f'not 0x00 <= ({commonbyte = }) <= 0xFF')

        self._ringsize = ringsize
        self._ringbits = ringbits
        self._maxmatchlen = 2 + maxmatchlen
        self._matchbits = matchbits
        self._commonbyte = commonbyte
        self._textbuf = bytearray(ringsize + 1 + maxmatchlen)
        self._outbuf = bytearray()
        self._textsize = 0
        self._matchpos = 0
        self._matchlen = 0
        self._lhs = [0] * (ringsize + 1)
        self._rhs = [0] * (ringsize + 1 + 0x100)
        self._upr = [0] * (ringsize + 1)
        self._initstate = 0
        self._encoder = self._iter_encode()

    def _init_tree(self) -> None:
        """Initializes the binary tree for match finding.

        This internal method sets up the initial state of the binary tree by:
        1. Setting right-hand links for all leaf nodes to point to ringsize
        2. Setting all parent pointers to ringsize to mark nodes as unconnected

        The binary tree structure is used to efficiently find matches in the ring
        buffer during compression. Each node in the tree represents a position in
        the buffer, with the tree organized to enable quick string comparisons.
        """
        ringsize = self._ringsize
        rhs = self._rhs
        upr = self._upr

        for i in range(ringsize + 1, ringsize + 1 + 0x100):
            rhs[i] = ringsize

        for i in range(ringsize):
            upr[i] = ringsize

    def _insert_node(self, r: int) -> None:
        """Inserts a new node into the binary tree.

        This internal method adds a new position in the ring buffer to the binary
        tree, maintaining the tree's balance and match-finding properties.

        Args:
            r: Position in the ring buffer to insert into the tree.

        The method also updates the match length and position if it finds a better
        match while traversing the tree during insertion.
        """
        ringsize = self._ringsize
        maxmatchlen = self._maxmatchlen
        textbuf = self._textbuf
        lhs = self._lhs
        rhs = self._rhs
        upr = self._upr

        cmp = 1
        key = r
        p = ringsize + 1 + textbuf[key]
        rhs[r] = ringsize
        lhs[r] = ringsize
        self._matchlen = 0

        while 1:
            if cmp >= 0:
                if rhs[p] != ringsize:
                    p = rhs[p]
                else:
                    rhs[p] = r
                    upr[r] = p
                    return
            else:
                if lhs[p] != ringsize:
                    p = lhs[p]
                else:
                    lhs[p] = r
                    upr[r] = p
                    return

            for i in range(maxmatchlen):
                cmp = textbuf[key + i] - textbuf[p + i]
                if cmp != 0:
                    break
            else:
                i = maxmatchlen

            if i > self._matchlen:
                self._matchlen = i
                self._matchpos = p
                if i >= maxmatchlen:
                    break

        upr[r] = upr[p]
        lhs[r] = lhs[p]
        rhs[r] = rhs[p]
        upr[lhs[p]] = r
        upr[rhs[p]] = r
        if rhs[upr[p]] == p:
            rhs[upr[p]] = r
        else:
            lhs[upr[p]] = r
        upr[p] = ringsize

    def _delete_node(self, p: int) -> None:
        """Removes a node from the binary tree.

        This internal method removes a position from the binary tree when it
        moves out of the sliding window, maintaining the tree's structure.

        Args:
            p: Position in the ring buffer to remove from the tree.

        The method handles all cases of node removal (leaf nodes, nodes with
        one child, and nodes with two children) while preserving the tree balance.
        """
        ringsize = self._ringsize
        upr = self._upr
        if upr[p] == ringsize:
            return
        lhs = self._lhs
        rhs = self._rhs

        if rhs[p] == ringsize:
            q = lhs[p]
        elif lhs[p] == ringsize:
            q = rhs[p]
        else:
            q = lhs[p]
            if rhs[q] != ringsize:
                q = rhs[q]
                while rhs[q] != ringsize:
                    q = rhs[q]
                rhs[upr[q]] = lhs[q]
                upr[lhs[q]] = upr[q]
                lhs[q] = lhs[p]
                upr[lhs[p]] = q

            rhs[q] = rhs[p]
            upr[rhs[p]] = q

        upr[q] = upr[p]
        if rhs[upr[p]] == p:
            rhs[upr[p]] = q
        else:
            lhs[upr[p]] = q
        upr[p] = ringsize

    def _iter_encode(self) -> _CodecGenerator:
        """Generator that implements the main compression loop.

        This internal method is the core of the LZSSW compression algorithm. It:
        1. Initializes the sliding window with the common byte
        2. Processes input bytes one at a time
        3. Maintains the binary tree to find matches
        4. Outputs literal bytes or match position/length pairs
        5. Updates the ring buffer as compression progresses

        Yields:
            None on each iteration, accepting the next input byte or None/Ellipsis
            to indicate end of input.
        """
        self._initstate = 1
        ringsize = self._ringsize
        maxmatchlen = self._maxmatchlen
        textbuf = self._textbuf
        outbuf = self._outbuf
        codebuf = bytearray(maxmatchlen - 1)
        posshift = 16 - self._ringbits

        self._init_tree()
        codebuf[0] = 0
        codebufptr = 1
        mask = 1
        s = 0
        r = ringsize - maxmatchlen

        commonbyte = self._commonbyte
        for i in range(s, r):
            textbuf[i] = commonbyte

        for len in range(maxmatchlen):
            c = yield  # getc()
            if c is Ellipsis:  # EOF
                self._initstate = -1
                break
            assert isinstance(c, int)
            textbuf[r + len] = c
        else:
            len = maxmatchlen

        if len == 0:
            return

        for i in range(1, 1 + maxmatchlen):
            self._insert_node(r - i)
        self._insert_node(r)

        while len > 0:
            if self._matchlen > len:
                self._matchlen = len

            if self._matchlen < 3:
                self._matchlen = 1
                codebuf[0] |= mask
                codebuf[codebufptr] = textbuf[r]
                codebufptr += 1
            else:
                code = self._matchpos
                codebuf[codebufptr] = code & 0x00FF
                codebufptr += 1
                code = (code >> 8) << posshift
                code |= self._matchlen - 3
                codebuf[codebufptr] = code
                codebufptr += 1

            mask = (mask << 1) & 0xFF
            if mask == 0:
                outbuf.extend(codebuf[i] for i in range(codebufptr))
                codebuf[0] = 0
                codebufptr = 1
                mask = 1

            lastmatchlen = self._matchlen
            for i in range(lastmatchlen):
                if self._initstate < 0:
                    break
                c = yield  # getc()
                if c is Ellipsis:  # EOF
                    self._initstate = -1
                    break

                self._delete_node(s)
                assert isinstance(c, int)
                textbuf[s] = c
                if s < maxmatchlen - 1:
                    textbuf[ringsize + s] = c
                s = (s + 1) % ringsize
                r = (r + 1) % ringsize
                self._insert_node(r)
            else:
                i = lastmatchlen

            while i < lastmatchlen:
                self._delete_node(s)
                s = (s + 1) % ringsize
                r = (r + 1) % ringsize
                len -= 1
                if len:
                    self._insert_node(r)
                i += 1
            else:
                i += 1

        if codebufptr > 1:
            outbuf.extend(codebuf[i] for i in range(codebufptr))

    def reset(self) -> None:
        """Resets the compressor to its initial state.

        Clears all internal buffers, match tracking variables, and the binary tree.
        This allows the compressor to be reused for a new compression task.
        """
        self._outbuf.clear()
        self._textsize = 0
        self._matchpos = 0
        self._matchlen = 0
        self._initstate = 0
        self._encoder.close()
        self._encoder = self._iter_encode()

    def compress(self, data: ByteIterable) -> bytearray:
        """Compresses the given data using LZSSW encoding.

        The method feeds data to the compression generator, which looks for matches
        in the sliding window and outputs either literal bytes or match references.

        Args:
            data: Input data to compress.

        Returns:
            Compressed data as bytearray.

        Raises:
            LZSSWException: If the compressor has already been flushed.
        """
        if self._initstate < 0:
            raise LZSSWException('already flushed')

        encoder = self._encoder
        send = encoder.send
        if self._initstate == 0:  # first
            send(None)

        for b in data:
            send(b)

        outbuf = self._outbuf
        chunk = outbuf[:]
        outbuf.clear()
        return chunk

    def flush(self) -> bytearray:
        """Flushes the compressor and returns any remaining compressed data.

        This method signals the end of input to the compressor, which may output
        additional compressed data based on its internal state. After calling flush,
        the compressor cannot be used for further compression without calling reset.

        Returns:
            Any remaining compressed data as a bytearray, or empty if already flushed.
        """
        if self._initstate < 0:
            return bytearray()
        else:
            if self._initstate == 0:  # first
                self._encoder.send(None)
            try:
                self._encoder.send(Ellipsis)  # EOF
            except StopIteration:
                pass
            outbuf = self._outbuf
            chunk = outbuf[:]
            outbuf.clear()
            self._initstate = -1
            return chunk

    @property
    def eof(self) -> bool:
        """Indicates whether the compressor has reached end of file state.

        Returns:
            bool: True if in EOF state (initialization state < 0), False otherwise.
        """
        return self._initstate < 0


class LZSSWDecompressor(BaseDecompressor):
    """LZSSW decompression implementation.

    This decompressor handles data compressed by the LZSSW algorithm, expanding
    match references back into their original sequences using a ring buffer that
    mirrors the one used during compression.

    The decompressor parameters must match those used for compression:
    - Ring buffer size controls the maximum back-reference distance
    - Maximum match length affects how much data a single match can represent
    - Common byte is used to initialize the ring buffer
    """

    def __init__(
            self,
            ringsize: int = RING_SIZE_DEF,
            maxmatchlen: int = MAX_MATCH_LEN_DEF,
            commonbyte: int = COMMON_BYTE_DEF,
    ) -> None:
        """Initializes a new LZSSW decompressor instance.

        Args:
            ringsize: Size of the ring buffer, must match the compressor setting.
            maxmatchlen: Maximum match length, must match the compressor setting.
            commonbyte: Initial ring buffer fill value, must match compressor setting.

        Raises:
            LZSSWException: If any parameters are out of their valid ranges.
        """
        ringsize = ringsize.__index__()
        maxmatchlen = maxmatchlen.__index__()
        commonbyte = commonbyte.__index__()
        ringbits = math.ceil(math.log2(ringsize))
        matchbits = math.ceil(math.log2(maxmatchlen))

        if not RING_SIZE_MIN <= ringsize <= RING_SIZE_MAX:
            raise ValueError(f'not ({RING_SIZE_MIN = }) '
                             f'<= ({ringsize = }) '
                             f'<= ({RING_SIZE_MAX = })')

        if not MAX_MATCH_LEN_MIN <= maxmatchlen <= MAX_MATCH_LEN_MAX:
            raise ValueError(f'not ({MAX_MATCH_LEN_MIN = }) '
                             f'<= ({maxmatchlen = }) '
                             f'<= ({MAX_MATCH_LEN_MAX = })')

        if ringbits + matchbits > 16:
            raise ValueError(f'({ringbits = }) + ({matchbits = }) > 16')

        if not 0x00 <= commonbyte <= 0xFF:
            raise ValueError(f'not 0x00 <= ({commonbyte = }) <= 0xFF')

        self._ringsize = ringsize
        self._ringbits = ringbits
        self._maxmatchlen = maxmatchlen + 2
        self._matchbits = matchbits
        self._commonbyte = commonbyte
        self._textbuf = bytearray(ringsize)
        self._outbuf = bytearray()
        self._ahead = bytearray()
        self._initstate = 0
        self._decoder = self._iter_decode()

    def _iter_decode(self) -> _CodecGenerator:
        """Generator that implements the main decompression loop.

        This internal method drives the LZSSW decompression process:
        1. Initializes the ring buffer with the common byte
        2. Processes flag bits to determine what follows (literal/match)
        3. Copies literal bytes directly to output
        4. Expands match references using the ring buffer
        5. Maintains the ring buffer for future matches

        Yields:
            None on each iteration, accepting compressed data or None/Ellipsis
            to indicate end of input.
        """
        self._initstate = 1
        ringsize = self._ringsize
        maxmatchlen = self._maxmatchlen
        textbuf = self._textbuf
        outbuf = self._outbuf
        posshift = 8 - (16 - self._ringbits)
        lenmask = (1 << self._matchbits) - 1

        commonbyte = self._commonbyte
        for i in range(ringsize - maxmatchlen):
            textbuf[i] = commonbyte
        r = ringsize - maxmatchlen
        flags = 0

        while 1:
            flags >>= 1
            if (flags & 0x0100) == 0:
                c = yield
                if c is Ellipsis:  # EOF
                    break
                assert isinstance(c, int)
                flags = c | 0xFF00

            if (flags & 0x0001) != 0:
                c = yield
                if c is Ellipsis:  # EOF
                    break
                assert isinstance(c, int)
                outbuf.append(c)
                textbuf[r] = c
                r = (r + 1) % ringsize
            else:
                i = yield
                if i is Ellipsis:  # EOF
                    break
                j = yield
                if j is Ellipsis:  # EOF
                    break
                assert isinstance(i, int)
                assert isinstance(j, int)

                i |= (j << posshift) & 0xFF00
                j = (j & lenmask) + 2

                for k in range(j + 1):
                    p = (i + k) % ringsize
                    c = textbuf[p]
                    outbuf.append(c)
                    textbuf[r] = c
                    r = (r + 1) % ringsize

        self._initstate = -1

    def decompress(
            self,
            data: ByteIterable,
            max_length: int = -1,
            /,
    ) -> bytearray:
        """Decompresses LZSSW-encoded data.

        The method feeds compressed data to the decompression generator, which
        expands match references and copies literal bytes to rebuild the original
        data.

        Args:
            data: Compressed input data to decompress.
            max_length: Maximum number of bytes to decompress. Default -1 means no limit.

        Returns:
            Decompressed data as bytearray.

        Raises:
            LZSSWException: If the decompressor has been flushed.
        """
        if self._initstate < 0:
            raise LZSSWException('already flushed')

        max_length = max_length.__index__()
        if max_length < 0:
            max_length = -1
            total = -2
            limited = False
        else:
            total = 0
            limited = True

        outbuf = self._outbuf
        ahead = self._ahead
        ahead.extend(data)
        ahead_len = len(ahead)
        ahead_idx = 0
        decoder = self._decoder
        send = decoder.send

        if self._initstate == 0:  # first
            send(None)

        while total < max_length and ahead_idx < ahead_len:
            send(ahead[ahead_idx])
            ahead_idx += 1
            if max_length > 0:
                total = len(outbuf)

        del ahead[:ahead_idx]
        if limited:
            chunk = outbuf[:max_length]
            del outbuf[:max_length]
        else:
            chunk = outbuf[:]
            outbuf.clear()
        return chunk

    def flush(self) -> bytearray:
        """Flushes any remaining data from the decompressor.

        This method finalizes the decompression process by:
        1. Processing any remaining input data
        2. Signaling EOF to the decoder
        3. Returning any remaining decompressed data

        Returns:
            Any remaining decompressed data, or empty bytearray if already flushed.
        """
        if self._initstate < 0:
            return bytearray()

        chunk = self.decompress(b'')
        try:
            self._decoder.send(Ellipsis)  # EOF
        except StopIteration:
            pass
        return chunk

    def reset(self) -> None:
        """Resets the decompressor to its initial state.

        Clears all internal buffers and state variables, allowing the decompressor
        to be reused for a new decompression task. This includes:
        - Clearing output and ahead buffers
        - Resetting state variables
        - Creating a new decoder generator
        """
        self._outbuf.clear()
        self._ahead.clear()
        self._initstate = 0
        self._decoder.close()
        self._decoder = self._iter_decode()

    @property
    def eof(self) -> bool:
        """Whether the decompressor has reached the end of the compressed stream.

        Returns:
            True if all input has been processed and flushed, False otherwise.
        """
        return self._initstate < 0

    @property
    def unused_data(self) -> bytearray:
        """Gets any unprocessed data remaining after decompression.

        Returns any data that was found after the end of the compressed stream
        or None if decompression is not yet complete.

        Returns:
            Remaining unprocessed data if eof is True, empty bytearray otherwise.
        """
        return self._ahead if self.eof else bytearray()

    @property
    def needs_input(self) -> bool:
        """Checks if more input data is needed to continue decompression.

        This property indicates whether the decompressor needs more input data
        to make progress. It returns False only when the end of the compressed
        stream has been reached.

        Returns:
            True if more input data is needed, False if decompression is complete.
        """
        return not self.eof


class LZSSWFile(CodecFile):
    """File-like object for reading/writing LZSSW compressed files.

    This class provides a high-level interface for working with LZSSW compressed
    files, supporting both reading and writing operations with automatic
    compression/decompression.
    """

    def __init__(
            self,
            filename: str | bytes | os.PathLike | IO,
            mode: str = 'r',
    ) -> None:
        """Creates a new LZSSW file object.

        Args:
            filename: Path to the file or an existing file object.
            mode: File open mode ('r'/'rb' for reading, 'w'/'wb'/'x'/'xb'/'a'/'ab' for writing).
        """
        comp = LZSSWCompressor()
        decomp = LZSSWDecompressor()
        super().__init__(filename, mode=mode, comp=comp, decomp=decomp)


def compress(
        data: ByteIterable,
        ringsize: int = RING_SIZE_DEF,
        maxmatchlen: int = MAX_MATCH_LEN_DEF,
        commonbyte: int = COMMON_BYTE_DEF,
) -> bytes:
    """Compresses data using the LZSSW algorithm.

    This is a convenience function that creates a compressor with specified
    parameters, compresses the data, and returns the result.

    Args:
        data: Data to compress.
        ringsize: Size of the ring buffer, must be between RING_SIZE_MIN and
                RING_SIZE_MAX. Larger sizes allow finding matches further back.
        maxmatchlen: Maximum length of matched sequences, between MAX_MATCH_LEN_MIN
                    and MAX_MATCH_LEN_MAX.
        commonbyte: Value used to fill the initial ring buffer, must be 0-255.

    Returns:
        Compressed data as bytes.

    Raises:
        LZSSWException: If any parameters are out of their valid ranges.
    """
    comp = LZSSWCompressor(ringsize, maxmatchlen, commonbyte)
    return codec_compress(data, comp)


def decompress(
        data: ByteIterable,
        ringsize: int = RING_SIZE_DEF,
        maxmatchlen: int = MAX_MATCH_LEN_DEF,
        commonbyte: int = COMMON_BYTE_DEF,
) -> bytes:
    """Decompresses LZSSW-compressed data.

    This is a convenience function that creates a decompressor with specified
    parameters, decompresses the data, and returns the result. The parameters
    must match those used during compression.

    Args:
        data: LZSSW-compressed data to decompress.
        ringsize: Size of the ring buffer, must match compression setting.
        maxmatchlen: Maximum length of matched sequences, must match compression.
        commonbyte: Value used to fill the initial ring buffer, must match compression.

    Returns:
        Decompressed data as bytes.

    Raises:
        LZSSWException: If any parameters are out of their valid ranges.
    """
    decomp = LZSSWDecompressor(ringsize, maxmatchlen, commonbyte)
    return codec_decompress(data, decomp)


def open(
        filename: str | bytes | IO,
        mode: str = 'r',
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
) -> CodecFile | io.TextIOWrapper:
    """Opens an LZSSW compressed file.

    This provides a high-level interface similar to the built-in open() function
    but for LZSSW compressed files. It supports both binary and text modes with
    default compression parameters.

    Args:
        filename: Path to file or file object.
        mode: File open mode ('r'/'rb' for reading, 'w'/'wb'/'x'/'xb'/'a'/'ab' for writing).
        encoding: Text encoding for text mode.
        errors: How to handle encoding/decoding errors in text mode.
        newline: How to handle newlines in text mode.

    Returns:
        A CodecFile for binary mode or TextIOWrapper for text mode.

    Raises:
        ValueError: For invalid mode combinations.
    """
    comp = LZSSWCompressor()
    decomp = LZSSWDecompressor()
    return codec_open(
        filename,
        mode=mode,
        encoding=encoding,
        errors=errors,
        newline=newline,
        comp=comp,
        decomp=decomp,
    )


def main() -> None:
    """Command-line interface for LZSSW compression/decompression.

    Provides a command-line tool for compressing or decompressing files using
    the LZSSW algorithm. Supports reading from standard input and writing to
    standard output, with configurable compression parameters.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--decompress', action='store_true',
                        help='Preform decompression instead of compression.')

    parser.add_argument('-r', '--ring-buffer-size', type=int, default=RING_SIZE_DEF,
                        help='LZSS ring buffer size.')

    parser.add_argument('-m', '--max-match-size', type=int, default=MAX_MATCH_LEN_DEF,
                        help='Maximum LZSS match size.')

    parser.add_argument('-f', '--fill-value', type=int, default=COMMON_BYTE_DEF,
                        help='Fill byte value; default: ASCII space (0x20).')

    parser.add_argument('infile', nargs='?', type=argparse.FileType('rb'), default=sys.stdin,
                        help='Input binary file; default: STDIN.')

    parser.add_argument('outfile', nargs='?', type=argparse.FileType('wb'), default=sys.stdout,
                        help='Output binary file; default: STDOUT.')

    args = parser.parse_args()

    if args.decompress:
        decomp = LZSSWDecompressor(ringsize=args.ring_buffer_size,
                                   maxmatchlen=args.max_match_size,
                                   commonbyte=args.fill_value)
        out_file = args.outfile

        with codec_open(args.infile, mode='rb', decomp=decomp) as in_file:
            out_file.write(in_file.read())
    else:
        comp = LZSSWCompressor(ringsize=args.ring_buffer_size,
                               maxmatchlen=args.max_match_size,
                               commonbyte=args.fill_value)
        in_file = args.infile

        with codec_open(args.outfile, mode='wb', comp=comp) as out_file:
            out_file.write(in_file.read())


if __name__ == '__main__':  # pragma: no cover
    main()
