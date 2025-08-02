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

import argparse
import io
import os
import sys
from typing import IO
from typing import Optional
from typing import Union

from base import BaseCompressor
from base import BaseDecompressor
from base import ByteIterable
from base import CodecFile
from base import codec_compress
from base import codec_decompress
from base import codec_open


class RLEBException(Exception):
    pass


class RLEBCompressor(BaseCompressor):

    def __init__(self) -> None:

        self._eof = False
        self._ring = bytearray(0x100)
        self._head = 0
        self._tail = 0
        self._used = 0
        self._same = 0
        self._prev = -1  # invalid

    def compress(self, data: ByteIterable) -> bytearray:

        if self._eof:
            raise RLEBException('already flushed')

        ring = self._ring
        head = self._head
        tail = self._tail
        used = self._used
        same = self._same
        prev = self._prev
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
                if same == 3:
                    used -= same
                    if used:
                        out_append(used - 1)
                        for _ in range(used):
                            byte = ring[head]
                            out_append(byte)
                            head = (head + 1) & 0xFF
                    byte = prev  # restore
                    used = same
                else:
                    # Once hit the maximum same size, output the maximum same group
                    if same == 3 + 0x7F:
                        out_append((same - 3) | 0x80)
                        out_append(prev)
                        head = (head + same) & 0xFF  # discard same block
                        used = 0
                        same = 0
                        byte = -1  # invalidate previous value after this loop
                    else:
                        # Once hit the maximum past size, output any past garbled data
                        if used == 1 + 0x7F:
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
                if same >= 3:
                    out_append((same - 3) | 0x80)
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

            prev = byte

        self._head = head
        self._tail = tail
        self._used = used
        self._same = same
        self._prev = prev

        return out

    def flush(self) -> bytearray:

        out = bytearray()

        if not self._eof:
            used = self._used
            same = self._same
            out_append = out.append

            if same >= 3:
                out_append((same - 3) | 0x80)
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

            self._head = 0
            self._tail = 0
            self._used = 0
            self._same = 0
            self._prev = -1
            self._eof = True

        return out

    def reset(self) -> None:

        self._eof = False
        self._head = 0
        self._tail = 0
        self._used = 0
        self._same = 0
        self._prev = -1  # invalid

    @property
    def eof(self) -> bool:

        return self._eof


class RLEBDecompressor(BaseDecompressor):

    def __init__(self) -> None:

        self._eof = False
        self._rle = False
        self._more = 0
        self._prev = -1  # invalid
        self._ahead = bytearray()

    def decompress(
            self,
            data: ByteIterable,
            max_length: int = -1,
            /,
    ) -> bytearray:

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

        rle = self._rle
        more = self._more
        prev = self._prev
        ahead = self._ahead
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
                    more += 3  # a span compresses at least 3 bytes
                else:  # uncompressed span ahead
                    more += 1  # a span must cover at least 1 byte

        del ahead[:ahead_idx]
        self._rle = rle
        self._more = more
        self._prev = prev
        return out

    def flush(self) -> bytearray:

        if self._eof:
            return bytearray()

        chunk = self.decompress(b'')
        self._eof = True
        return chunk

    def reset(self) -> None:

        self._eof = False
        self._rle = False
        self._more = 0
        self._prev = -1  # invalid
        self._ahead.clear()

    @property
    def eof(self) -> bool:

        return self._eof

    @property
    def unused_data(self) -> bytearray:

        return self._ahead if self.eof else bytearray()

    @property
    def needs_input(self) -> bool:

        if self._eof:
            return False
        elif self._rle:
            return self._prev < 0  # still invalid
        else:
            return len(self._ahead) < self._more


class RLEBFile(CodecFile):

    def __init__(
            self,
            filename: Union[str, bytes, os.PathLike, IO],
            mode: str = 'r',
    ) -> None:

        comp = RLEBCompressor()
        decomp = RLEBDecompressor()
        super().__init__(filename, mode=mode, comp=comp, decomp=decomp)


def compress(data: ByteIterable) -> bytes:

    comp = RLEBCompressor()
    return codec_compress(data, comp)


def decompress(data: ByteIterable) -> bytes:

    decomp = RLEBDecompressor()
    return codec_decompress(data, decomp)


def open(
        filename: Union[str, bytes, IO],
        mode: str = 'r',
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
) -> Union[CodecFile, io.TextIOWrapper]:

    comp = RLEBCompressor()
    decomp = RLEBDecompressor()
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
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--decompress', action='store_true',
                        help='Preform decompression instead of compression.')

    parser.add_argument('infile', nargs='?', type=argparse.FileType('rb'), default=sys.stdin,
                        help='Input binary file; default: STDIN.')

    parser.add_argument('outfile', nargs='?', type=argparse.FileType('wb'), default=sys.stdout,
                        help='Output binary file; default: STDOUT.')

    args = parser.parse_args()

    if args.decompress:
        decomp = RLEBDecompressor()
        out_file = args.outfile

        with codec_open(args.infile, mode='rb', decomp=decomp) as in_file:
            out_file.write(in_file.read())
    else:
        comp = RLEBCompressor()
        in_file = args.infile

        with codec_open(args.outfile, mode='wb', comp=comp) as out_file:
            out_file.write(in_file.read())


if __name__ == '__main__':  # pragma: no cover
    main()
