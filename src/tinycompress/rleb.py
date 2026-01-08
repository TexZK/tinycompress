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

"""Run-Length Encoding for Bytes (RLEB) compression algorithm.

This module implements the RLEB compression algorithm, which combines run-length
encoding with bit flags to efficiently compress data that contains repeated bytes.
"""

import argparse
import io
import os
import sys
from typing import IO

from .base import BaseCompressor
from .base import BaseDecompressor
from .base import ByteIterable
from .base import CodecFile
from .base import codec_compress
from .base import codec_decompress
from .base import codec_open

MINSAME_MIN = 3
MINSAME_MAX = 128

MINPAST_MIN = 0
MINPAST_MAX = 127


class RLEBException(Exception):
    """Exception raised for RLEB compression/decompression errors."""
    pass


class RLEBCompressor(BaseCompressor):
    """RLEB compression implementation.

    This compressor implements Run-Length Encoding for Bytes (RLEB), which
    efficiently compresses repeated byte sequences while maintaining good performance
    for non-repeated data.

    The algorithm uses a ring buffer to track recent bytes and detect repetitions.
    When MINSAME_MIN or more identical bytes are found, they are encoded as a run. Otherwise,
    bytes are stored literally with a count.
    """

    def __init__(
            self,
            minsame: int = MINSAME_MIN,
            minpast: int = MINPAST_MIN,
    ) -> None:
        """Initializes a new RLEB compressor instance.

        The compressor starts with an empty ring buffer and no previous byte tracked.
        The end-of-file flag is initially False.

        Args:
            minsame: Minimum compressed size, within MINSAME_MIN and MINSAME_MAX.
            minpast: Minimum bytes after a same streak, within MINPAST_MIN and MINPAST_MAX.

        Raises:
            ValueError: If any parameters are out of their valid ranges.
        """
        if not MINSAME_MIN <= minsame <= MINSAME_MAX:
            raise ValueError(f'not ({MINSAME_MIN = }) '
                             f'<= ({minsame = }) '
                             f'<= ({MINSAME_MAX = })')

        if not MINPAST_MIN <= minpast <= MINPAST_MAX:
            raise ValueError(f'not ({MINPAST_MIN = }) '
                             f'<= ({minpast = }) '
                             f'<= ({MINPAST_MAX = })')

        self._eof = False
        self._head = 0
        self._minpast = minpast
        self._minsame = minsame
        self._prev = -1  # invalid
        self._ring = bytearray(0x100)
        self._same = 0
        self._tail = 0
        self._used = 0

    def compress(self, data: ByteIterable) -> bytearray:
        """Compresses the given data using RLEB encoding.

        The compression maintains a ring buffer and tracks repeated bytes.
        When MINSAME_MIN or more identical bytes are found, they are encoded as
        a run using a flag byte and count.
        Non-repeated sequences are stored with their literal values.

        Args:
            data: Input data to compress.

        Returns:
            Compressed data as a bytearray.

        Raises:
            RLEBException: If the compressor has already been flushed.
        """
        if self._eof:
            raise RLEBException('already flushed')

        head = self._head
        minpast = self._minpast
        minsame = self._minsame
        prev = self._prev
        ring = self._ring
        same = self._same
        tail = self._tail
        used = self._used
        out = bytearray()
        out_append = out.append

        for byte in data:
            assert 0x00 <= byte <= 0xFF

            # Check if continuing a same byte streak
            if byte == prev:

                # Append to the past history and streak
                ring[tail] = byte
                tail = (tail + 1) & 0xFF
                used += 1
                same += 1

                # Once hit the minimum same size, output any past garbled data
                if same == minsame:
                    used -= same
                    if used:
                        out_append(used - 1)
                        for _ in range(used):
                            byte = ring[head]
                            out_append(byte)
                            head = (head + 1) & 0xFF
                    byte = prev  # restore
                    used = same

                # Once hit the maximum same size, output the maximum same group
                elif same == minsame + 0x7F:
                    out_append((same - minsame) | 0x80)
                    out_append(prev)
                    head = (head + same) & 0xFF  # discard same block
                    used = 0
                    same = 0
                    byte = -1  # invalidate previous value after this loop

                # Once hit the maximum past size, output any past garbled data
                elif used == 1 + 0x7F:
                    used -= same
                    if used:
                        out_append(used - 1)
                        for _ in range(used):
                            byte = ring[head]
                            out_append(byte)
                            head = (head + 1) & 0xFF
                    byte = prev  # restore
                    used = same

            else:  # different

                # If ending a meaningful same byte streak, output it
                if same >= minsame:
                    out_append((same - minsame) | 0x80)
                    out_append(prev)
                    head = tail  # discard history
                    used = 0

                # Append to the past history and start a new streak
                ring[tail] = byte
                tail = (tail + 1) & 0xFF
                used += 1
                same = 1

                # Once hit the maximum past size, output past garbled data
                if used == 1 + 0x7F:
                    out_append(used - 1)
                    for _ in range(used):
                        byte = ring[head]
                        out_append(byte)
                        head = (head + 1) & 0xFF
                    used = 0
                    same = 0
                    byte = -1  # invalidate previous value after this loop

                # Skip enough same values after garbled data
                elif used <= minpast:
                    byte = -1  # invalidate previous value after this loop

            prev = byte

        self._head = head
        self._prev = prev
        self._same = same
        self._tail = tail
        self._used = used

        return out

    def flush(self) -> bytearray:
        """Flushes any remaining data from the compressor.

        This outputs any pending data in the ring buffer and marks the compressor
        as finished. After calling flush(), the compressor cannot be used again
        unless reset.

        Returns:
            Any remaining compressed data as a bytearray.
        """
        out = bytearray()

        if not self._eof:
            minsame = self._minsame
            used = self._used
            same = self._same
            out_append = out.append

            if same >= minsame:
                out_append((same - minsame) | 0x80)
                out_append(self._prev)

            elif used:
                # Output past garbled data
                ring = self._ring
                head = self._head
                out_append(used - 1)
                for _ in range(used):
                    byte = ring[head]
                    out_append(byte)
                    head = (head + 1) & 0xFF

            self._eof = True
            self._head = 0
            self._prev = -1
            self._same = 0
            self._tail = 0
            self._used = 0

        return out

    def reset(self) -> None:
        """Resets the compressor to its initial state.

        This clears all internal buffers and state, allowing the compressor
        to be reused for a new compression task.
        """
        self._eof = False
        self._head = 0
        self._prev = -1  # invalid
        self._same = 0
        self._tail = 0
        self._used = 0

    @property
    def eof(self) -> bool:
        """Whether the compressor has been flushed.

        Returns:
            True if flush() has been called, False otherwise.
        """
        return self._eof


class RLEBDecompressor(BaseDecompressor):
    """RLEB decompression implementation.

    This decompressor handles data compressed using Run-Length Encoding for Bytes
    (RLEB). It processes the compressed data in chunks, expanding runs of repeated
    bytes and copying literal byte sequences.
    """

    def __init__(self, minsame: int = MINSAME_MIN) -> None:
        """Initializes a new RLEB decompressor instance.

        Sets up internal state for processing compressed data including tracking
        whether in RLE mode and buffering input data.

        Args:
            minsame: Minimum compressed size, within MINSAME_MIN and MINSAME_MAX.

        Raises:
            ValueError: If any parameters are out of their valid ranges.
        """
        if not MINSAME_MIN <= minsame <= MINSAME_MAX:
            raise ValueError(f'not ({MINSAME_MIN = }) '
                             f'<= ({minsame = }) '
                             f'<= ({MINSAME_MAX = })')

        self._ahead = bytearray()
        self._eof = False
        self._minsame = minsame
        self._more = 0
        self._prev = -1  # invalid
        self._rle = False

    def decompress(
            self,
            data: ByteIterable,
            max_length: int = -1,
            /,
    ) -> bytearray:
        """Decompresses RLEB-encoded data.

        Processes compressed data, expanding runs of repeated bytes and copying
        literal sequences. Can optionally limit the amount of output produced.

        Args:
            data: Compressed input data to decompress.
            max_length: Maximum number of output bytes to produce. Default -1 means no limit.

        Returns:
            Decompressed data as bytearray.

        Raises:
            RLEBException: If the decompressor has already been flushed.
        """

        if self._eof:
            raise RLEBException('already flushed')

        max_length = max_length.__index__()
        if max_length < 0:
            max_length = -1
            total = -2
            limited = False
        else:
            total = 0
            limited = True

        ahead = self._ahead
        minsame = self._minsame
        more = self._more
        prev = self._prev
        rle = self._rle
        ahead.extend(data)
        ahead_len = len(ahead)
        ahead_idx = 0
        out = bytearray()
        out_append = out.append
        out_extend = out.extend

        while total < max_length:
            # The current span covers more bytes, process them
            if more:

                # Within a RLE-compressed span, expand it
                if rle:
                    if prev < 0:  # still invalid
                        if ahead_idx >= ahead_len:
                            break
                        prev = ahead[ahead_idx]
                        ahead_idx += 1

                    if limited:
                        size = max_length - total
                        if size > more:
                            size = more
                    else:
                        size = more

                    out_extend(prev for _ in range(size))
                    more -= size
                    if limited:
                        total += size

                # Within an uncompressed span, output bytes directly
                else:
                    if ahead_idx >= ahead_len:
                        break
                    prev = ahead[ahead_idx]
                    ahead_idx += 1
                    out_append(prev)
                    more -= 1
                    if limited:
                        total += 1

            # The current span ended, read the control byte of the next span
            else:
                rle = False
                prev = -1  # invalidate
                if ahead_idx >= ahead_len:
                    break

                byte = ahead[ahead_idx]
                ahead_idx += 1
                more = byte & 0x7F  # partial span size

                if byte & 0x80:  # RLE-compressed span ahead
                    rle = True
                    more += minsame  # a span compresses at least minsame bytes
                else:  # uncompressed span ahead
                    more += 1  # a span must cover at least 1 byte

        del ahead[:ahead_idx]
        self._more = more
        self._prev = prev
        self._rle = rle
        return out

    def flush(self) -> bytearray:
        """Flushes any remaining data and marks decompression as complete.

        Should be called when all input data has been provided to ensure
        any buffered output is returned.

        Returns:
            Any remaining decompressed data, or empty bytearray if already flushed.
        """
        if self._eof:
            return bytearray()

        chunk = self.decompress(b'')
        self._eof = True
        return chunk

    def reset(self) -> None:
        """Resets the decompressor to its initial state.

        Clears all internal buffers and state variables, allowing the decompressor
        to be reused for a new decompression task.
        """
        self._ahead.clear()
        self._eof = False
        self._more = 0
        self._prev = -1  # invalid
        self._rle = False

    @property
    def eof(self) -> bool:
        """Whether the decompressor has reached the end of the compressed stream.

        Returns:
            True if all input has been processed and flushed, False otherwise.
        """
        return self._eof

    @property
    def unused_data(self) -> bytearray:
        """Gets any unprocessed data remaining after decompression.

        This property returns any data that was found after the end of the
        compressed stream.

        Returns:
            Remaining unprocessed data if eof is True, empty bytearray otherwise.
        """
        return self._ahead if self.eof else bytearray()

    @property
    def needs_input(self) -> bool:
        """Checks if more input data is needed to continue decompression.

        This property indicates whether the decompressor needs more data to
        make progress. When True, no output can be produced until more data
        is provided.

        Returns:
            True if more input is needed, False if decompressor can continue
            with current data.
        """
        if self._eof:
            return False
        elif self._rle:
            return self._prev < 0  # still invalid
        else:
            return len(self._ahead) < self._more


class RLEBFile(CodecFile):
    """File-like object for reading/writing RLEB compressed files.

    This class provides a high-level interface for working with RLEB compressed files,
    supporting both reading and writing operations with automatic compression/decompression.
    """

    def __init__(
            self,
            filename: str | bytes | os.PathLike | IO,
            mode: str = 'r',
            minsame: int = MINSAME_MIN,
            minpast: int = MINPAST_MIN,
    ) -> None:
        """Creates a new RLEB file object.

        Args:
            filename: Path to the file or an existing file object.
            mode: File open mode ('r'/'rb' for reading, 'w'/'wb'/'x'/'xb'/'a'/'ab' for writing).
            minsame: Minimum compressed size, within MINSAME_MIN and MINSAME_MAX.
            minpast: Minimum bytes after a same streak, within MINPAST_MIN and MINPAST_MAX.
        """
        comp = RLEBCompressor(minsame, minpast)
        decomp = RLEBDecompressor(minsame)
        super().__init__(filename, mode=mode, comp=comp, decomp=decomp)


def compress(
        data: ByteIterable,
        minsame: int = MINSAME_MIN,
        minpast: int = MINPAST_MIN,
) -> bytes:
    """Compresses data using the RLEB algorithm.

    This is a convenience function that creates a compressor, compresses
    the data, and returns the result.

    Args:
        data: Data to compress.
        minsame: Minimum compressed size, within MINSAME_MIN and MINSAME_MAX.
        minpast: Minimum bytes after a same streak, within MINPAST_MIN and MINPAST_MAX.

    Returns:
        Compressed data as bytes.

    Raises:
        ValueError: If any parameters are out of their valid ranges.
    """
    comp = RLEBCompressor(minsame, minpast)
    return codec_compress(data, comp)


def decompress(
        data: ByteIterable,
        minsame: int = MINSAME_MIN,
) -> bytes:
    """Decompresses RLEB-compressed data.

    This is a convenience function that creates a decompressor, decompresses
    the data, and returns the result.

    Args:
        data: RLEB-compressed data to decompress.
        minsame: Minimum compressed size, within MINSAME_MIN and MINSAME_MAX.

    Returns:
        Decompressed data as bytes.

    Raises:
        ValueError: If any parameters are out of their valid ranges.
    """
    decomp = RLEBDecompressor(minsame)
    return codec_decompress(data, decomp)


def open(
        filename: str | bytes | IO,
        mode: str = 'r',
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        minsame: int = MINSAME_MIN,
        minpast: int = MINPAST_MIN,
) -> CodecFile | io.TextIOWrapper:
    """Opens an RLEB compressed file.

    This provides a high-level interface similar to the built-in open() function
    but for RLEB compressed files. It supports both binary and text modes.

    Args:
        filename: Path to file or file object.
        mode: File open mode ('r'/'rb' for reading, 'w'/'wb'/'x'/'xb'/'a'/'ab' for writing).
        encoding: Text encoding for text mode.
        errors: How to handle encoding/decoding errors in text mode.
        newline: How to handle newlines in text mode.
        minsame: Minimum compressed size, within MINSAME_MIN and MINSAME_MAX.
        minpast: Minimum bytes after a same streak, within MINPAST_MIN and MINPAST_MAX.

    Returns:
        A CodecFile for binary mode or TextIOWrapper for text mode.
    """
    comp = RLEBCompressor(minsame, minpast)
    decomp = RLEBDecompressor(minsame)
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
    """Command-line interface for RLEB compression/decompression.

    Provides a command-line tool for compressing or decompressing files
    using the RLEB algorithm. Supports reading from standard input and
    writing to standard output.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--decompress', action='store_true',
                        help='Preform decompression instead of compression.')

    parser.add_argument('-s', '--minsame', type=int, default=MINSAME_MIN,
                        help='Minimum compressed size.')

    parser.add_argument('-p', '--minpast', type=int, default=MINPAST_MIN,
                        help='Minimum bytes after a same streak.')

    parser.add_argument('infile', nargs='?', type=argparse.FileType('rb'), default=sys.stdin,
                        help='Input binary file; default: STDIN.')

    parser.add_argument('outfile', nargs='?', type=argparse.FileType('wb'), default=sys.stdout,
                        help='Output binary file; default: STDOUT.')

    args = parser.parse_args()

    if args.decompress:
        decomp = RLEBDecompressor(minsame=args.minsame)
        out_file = args.outfile

        with codec_open(args.infile, mode='rb', decomp=decomp) as in_file:
            out_file.write(in_file.read())
    else:
        comp = RLEBCompressor(minsame=args.minsame, minpast=args.minpast)
        in_file = args.infile

        with codec_open(args.outfile, mode='wb', comp=comp) as out_file:
            out_file.write(in_file.read())


if __name__ == '__main__':  # pragma: no cover
    main()
