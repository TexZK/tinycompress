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

import io
import os
import pathlib
import sys

import pytest

from tinycompress.lzssw import LZSSWCompressor
from tinycompress.lzssw import LZSSWDecompressor
from tinycompress.lzssw import LZSSWException
from tinycompress.lzssw import LZSSWFile
from tinycompress.lzssw import compress
from tinycompress.lzssw import decompress
from tinycompress.lzssw import main
from tinycompress.lzssw import open as lzssw_open

lorem = b"""\
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
tempor incidunt ut labore et dolore magna aliqua.  Ut enim ad minim veniam,
quis nostrum exercitationem ullamco laboriosam, nisi ut aliquid ex ea
commodi consequatur.  Duis aute irure reprehenderit in voluptate velit esse
cillum dolore eu fugiat nulla pariatur.  Excepteur sint obcaecat cupiditat
non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

compress_partial_tags = ('ref_in', 'ref_out', 'ref_flush')
compress_partial_table = [
    (b'',     b'',  b''),
    (b'X',    b'',  b'\x01X'),
    (b'XYZ',  b'',  b'\x07XYZ'),

    (b'aaa',  b'',  b'\x07aaa'),
    (b'aaaa', b'',  b'\x01a\xEE\xF0'),

    (b'XYZaaa',   b'',  b'?XYZaaa'),
    (b'XYZaaaa',  b'',  b'\x0FXYZa\xF1\xF0'),

    (b'aaXYZ',    b'',  b'\x1FaaXYZ'),
    (b'aaaXYZ',   b'',  b'?aaaXYZ'),
    (b'aaaaXYZ',  b'',  b'\x1Da\xEE\xF0XYZ'),

    (b'XYZaaaXYZ',   b'',  b'?XYZaaa\xEE\xF0'),
    (b'XYZaaaaXYZ',  b'',  b'\x0FXYZa\xF1\xF0\xEE\xF0'),

    (b'aaXYZaa',      b'',  b'\x7FaaXYZaa'),
    (b'aaaXYZaaa',    b'',  b'?aaaXYZ\xEE\xF0'),
    (b'aaaaXYZaaaa',  b'',  b'\x1Da\xEE\xF0XYZ\xEE\xF1'),

    (1*b'Xa',       b'',  b'\x03Xa'),
    (2*b'Xa'+b'X',  b'',  b'\x03Xa\xEE\xF0'),
    (1*b'Xa',       b'',  b'\x03Xa'),
    (2*b'Xa'+b'X',  b'',  b'\x03Xa\xEE\xF0'),

    (lorem,
     (b'\xFFLorem ip\xFFsum dolo\xFFr sit am\xFFet, cons\xFFectetur \xFFadipisci\xBFng '
      b'eli\x07\x00s\xFBed\xF9\xF0 eius\xFFmod\ntemp\xFE\xFD\xF0incidun\xFFt ut '
      b'lab\xEE\xEF\xF0 et\xF9\xF3e m\xFFagna ali\xFFqua.  Ut\xDF enim\x15\x00 '
      b'm\xF5im\x01vl\x00am,\n\xFFquis nos\xFBtr\xF7\xF0exerc\x7Fitation'
      b'\xF1\xF0\x7FullamcoH\x03\xF7ios~\x00 nis\xD9iE\x01`\x02id\x8E\x00 e_a\n'
      b'com2\x00i\t\x03\xE8c\x00\x12\x00f\x00D\x83\x01aut\xDFe iruM\x00re'
      b'\xFFprehende\xFDr\x01\x00in vol\xCBup\x95\x00ey\x00"\x00 e\xFFsse\n'
      b'cill\xFC\xF7\xF5N\x00u fugi\xEFat n\x9E\x01 pa\xF9r(\x10\xD9\x02Excep'
      b'\xDBte\x13\x00siC\x00ob\xEFcaec)\x10cup\xFBid\x94\x01\nnon \x7Fproiden'
      b'$\x01\xFCB\x01\xFC\x00culpa \xFE\x82\x00 offici\x7Fa deserB\x01'),
     (b'\x97mol\x0B\x11am\x01\xBF\x01s\x1EG\x04um.\n'))
]
compress_partial_ids = [repr(v[0]) for v in compress_partial_table]

compress_whole_tags = ('ref_in', 'ref_out')
compress_whole_table = [(v[0], v[1]+v[2]) for v in compress_partial_table]
compress_whole_ids = [repr(v[0]) for v in compress_whole_table]

decompress_partial_tags = ('max_length', 'ref_in', 'ref_out', 'ahead')
decompress_partial_table = [
    ( 0,  b'',  b'',  b''),
    ( 1,  b'',  b'',  b''),
    (-1,  b'',  b'',  b''),

    ( 0,  b'\x07XYZ', b'',     b'\x07XYZ'),
    ( 1,  b'\x07XYZ', b'X',    b'YZ'),
    (-1,  b'\x07XYZ', b'XYZ',  b''),
]
decompress_partial_ids = [repr(v[:2]) for v in decompress_partial_table]

decompress_whole_tags = compress_whole_tags[:]
decompress_whole_table = [v[::-1] for v in compress_whole_table]
decompress_whole_ids = [repr(v[0]) for v in decompress_whole_table]

raw_filenames = [
    'CP437.PBM',
    'LOREM.TXT',
    'pep-0008.rst',
    'udhr_eng.xml',
]


class Test_LZSSWCompressor:

    @pytest.mark.parametrize('ringsize', [0x200, 0x200+1, 0x400, 0x800, 0x1000-1, 0x1000])
    def test___init___ringsize_pass(self, ringsize: int):

        LZSSWCompressor(ringsize=ringsize, maxmatchlen=0x10)

    @pytest.mark.parametrize('ringsize', [0x100, 0x200-1, 0x1000+1, 0x2000])
    def test___init___ringsize_raises(self, ringsize: int):

        with pytest.raises(ValueError, match='ringsize'):
            LZSSWCompressor(ringsize=ringsize, maxmatchlen=0x10)

    @pytest.mark.parametrize('maxmatchlen', [0x10, 0x10+1, 0x20, 0x40, 0x80-1, 0x80])
    def test___init___maxmatchlen_pass(self, maxmatchlen: int):

        LZSSWCompressor(ringsize=0x200, maxmatchlen=maxmatchlen)

    @pytest.mark.parametrize('maxmatchlen', [0x08, 0x10-1, 0x80+1, 0x100])
    def test___init___maxmatchlen_raises(self, maxmatchlen: int):

        with pytest.raises(ValueError, match='maxmatchlen'):
            LZSSWCompressor(ringsize=0x200, maxmatchlen=maxmatchlen)

    @pytest.mark.parametrize(('ringsize', 'maxmatchlen'), [
        (0x1000, 0x11),
        (0x0801, 0x20),
        (0x0800, 0x21),
        (0x0401, 0x40),
        (0x0400, 0x41),
        (0x0201, 0x80),
    ])
    def test___init___bits_raises(
            self,
            ringsize: int,
            maxmatchlen: int,
    ):
        with pytest.raises(ValueError, match='bits'):
            LZSSWCompressor(ringsize=ringsize, maxmatchlen=maxmatchlen)

    @pytest.mark.parametrize('commonbyte', [0x00, 0xFF])
    def test___init___commonbyte_pass(self, commonbyte: int):

        LZSSWCompressor(commonbyte=commonbyte)

    @pytest.mark.parametrize('commonbyte', [-1, 0x100])
    def test___init___commonbyte_raises(self, commonbyte: int):

        with pytest.raises(ValueError, match='commonbyte'):
            LZSSWCompressor(commonbyte=commonbyte)

    @pytest.mark.parametrize(compress_partial_tags, compress_partial_table, ids=compress_partial_ids)
    def test_compress__table(
            self,
            ref_in: bytes,
            ref_out: bytes,
            ref_flush: bytes,
    ):
        comp = LZSSWCompressor()
        ans_out = comp.compress(ref_in)
        assert ans_out == ref_out
        del ref_flush

    def test_compress__already_flushed(self):

        comp = LZSSWCompressor()
        comp.compress(b'...')
        comp.flush()
        with pytest.raises(LZSSWException, match='already flushed'):
            comp.compress(b'...')

    @pytest.mark.parametrize(compress_partial_tags, compress_partial_table, ids=compress_partial_ids)
    def test_flush__table(
            self,
            ref_in: bytes,
            ref_out: bytes,
            ref_flush: bytes,
    ):
        comp = LZSSWCompressor()
        ans_out = comp.compress(ref_in)
        assert ans_out == ref_out
        ans_out = comp.flush()
        assert ans_out == ref_flush
        ans_out = comp.flush()
        assert ans_out == b''

    def test_reset(self):

        comp = LZSSWCompressor()
        comp.compress(b'aaa')
        comp.reset()
        comp.compress(b'aaaaa')
        assert comp.eof is False
        comp.flush()
        assert comp.eof is True
        comp.reset()
        assert comp.eof is False
        comp.compress(b'XYZ')
        assert comp.eof is False

    def test_eof(self):

        comp = LZSSWCompressor()
        assert comp.eof is False
        comp.compress(b'...')
        assert comp.eof is False
        comp.flush()
        assert comp.eof is True


class Test_LZSSWDecompressor:

    @pytest.mark.parametrize('ringsize', [0x200, 0x200+1, 0x400, 0x800, 0x1000-1, 0x1000])
    def test___init___ringsize_pass(self, ringsize: int):

        LZSSWDecompressor(ringsize=ringsize, maxmatchlen=0x10)

    @pytest.mark.parametrize('ringsize', [0x100, 0x200-1, 0x1000+1, 0x2000])
    def test___init___ringsize_raises(self, ringsize: int):

        with pytest.raises(ValueError, match='ringsize'):
            LZSSWDecompressor(ringsize=ringsize, maxmatchlen=0x10)

    @pytest.mark.parametrize('maxmatchlen', [0x10, 0x10+1, 0x20, 0x40, 0x80-1, 0x80])
    def test___init___maxmatchlen_pass(self, maxmatchlen: int):

        LZSSWDecompressor(ringsize=0x200, maxmatchlen=maxmatchlen)

    @pytest.mark.parametrize('maxmatchlen', [0x08, 0x10-1, 0x80+1, 0x100])
    def test___init___maxmatchlen_raises(self, maxmatchlen: int):

        with pytest.raises(ValueError, match='maxmatchlen'):
            LZSSWDecompressor(ringsize=0x200, maxmatchlen=maxmatchlen)

    @pytest.mark.parametrize(('ringsize', 'maxmatchlen'), [
        (0x1000, 0x11),
        (0x0801, 0x20),
        (0x0800, 0x21),
        (0x0401, 0x40),
        (0x0400, 0x41),
        (0x0201, 0x80),
    ])
    def test___init___bits_raises(
            self,
            ringsize: int,
            maxmatchlen: int,
    ):
        with pytest.raises(ValueError, match='bits'):
            LZSSWDecompressor(ringsize=ringsize, maxmatchlen=maxmatchlen)

    @pytest.mark.parametrize('commonbyte', [0x00, 0xFF])
    def test___init___commonbyte_pass(self, commonbyte: int):

        LZSSWDecompressor(commonbyte=commonbyte)

    @pytest.mark.parametrize('commonbyte', [-1, 0x100])
    def test___init___commonbyte_raises(self, commonbyte: int):

        with pytest.raises(ValueError, match='commonbyte'):
            LZSSWDecompressor(commonbyte=commonbyte)

    @pytest.mark.parametrize(decompress_partial_tags, decompress_partial_table, ids=decompress_partial_ids)
    def test_decompress__table(
            self,
            max_length: int,
            ref_in: bytes,
            ref_out: bytes,
            ahead: bytes,
    ):
        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(ref_in, max_length)
        assert ans_out == ref_out
        assert decomp._ahead == ahead

    def test_decompress__already_flushed(self):

        decomp = LZSSWDecompressor()
        decomp.decompress(b'\x07XYZ')
        decomp.flush()
        with pytest.raises(LZSSWException, match='already flushed'):
            decomp.decompress(b'\x07XYZ')

    def test_decompress__ahead__diff(self):

        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(b'\x07XYZ', 1)
        assert ans_out == b'X'
        ans_out = decomp.decompress(b'')
        assert ans_out == b'YZ'

    def test_decompress__ahead__same(self):

        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(b'\x01a\xEE\xF0', 1)
        assert ans_out == b'a'
        ans_out = decomp.decompress(b'')
        assert ans_out == b'aaa'

    def test_flush__empty(self):

        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(b'')
        assert ans_out == b''
        ans_out = decomp.flush()
        assert ans_out == b''
        ans_out = decomp.flush()
        assert ans_out == b''

    def test_flush__diff_ahead(self):

        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(b'\x07XYZ', 1)
        assert ans_out == b'X'
        ans_out = decomp.flush()
        assert ans_out == b'YZ'
        ans_out = decomp.flush()
        assert ans_out == b''

    def test_flush__diff_early(self):

        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(b'\x07X')
        assert ans_out == b'X'
        ans_out = decomp.decompress(b'YZ')
        assert ans_out == b'YZ'
        ans_out = decomp.flush()
        assert ans_out == b''

    def test_flush__same_ahead(self):

        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(b'\x01a\xEE\xF0', 1)
        assert ans_out == b'a'
        ans_out = decomp.decompress(b'', 1)
        assert ans_out == b'a'
        ans_out = decomp.flush()
        assert ans_out == b'aa'

    def test_flush__same_early(self):

        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(b'\x01')
        assert ans_out == b''
        ans_out = decomp.decompress(b'a\xEE\xF0')
        assert ans_out == b'aaaa'
        ans_out = decomp.flush()
        assert ans_out == b''

    def test_flush__break_early(self):

        decomp = LZSSWDecompressor()
        ans_out = decomp.flush()
        assert ans_out == b''

        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(b'\x01')
        assert ans_out == b''
        ans_out = decomp.flush()
        assert ans_out == b''

        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(b'\x00')
        assert ans_out == b''
        ans_out = decomp.flush()
        assert ans_out == b''

        decomp = LZSSWDecompressor()
        ans_out = decomp.decompress(b'\x00\x00')
        assert ans_out == b''
        ans_out = decomp.flush()
        assert ans_out == b''

    def test_reset(self):

        decomp = LZSSWDecompressor()
        decomp.decompress(b'\x07XYZ', 1)
        assert decomp._ahead == b'YZ'
        decomp.reset()
        assert decomp._ahead == b''
        decomp.decompress(b'\x01a\xEE\xF0', 2)
        assert decomp.eof is False
        decomp.flush()
        assert decomp.eof is True
        decomp.reset()
        assert decomp.eof is False
        decomp.decompress(b'\x01a')

    def test_eof(self):

        decomp = LZSSWDecompressor()
        assert decomp.eof is False
        decomp.decompress(b'\x01a')
        assert decomp.eof is False
        decomp.flush()
        assert decomp.eof is True

    def test_unused_data(self):

        decomp = LZSSWDecompressor()
        assert decomp.unused_data == b''
        decomp.decompress(b'\x07XYZ', 1)
        assert decomp.unused_data == b''
        decomp.flush()
        assert decomp.unused_data == b''

    def test_needs_input(self):

        decomp = LZSSWDecompressor()
        assert decomp.needs_input is True

        out = decomp.decompress(b'')
        assert out == b''
        assert decomp.needs_input is True

        decomp.reset()
        out = decomp.decompress(b'\x07')
        assert out == b''
        assert decomp.needs_input is True
        out = decomp.decompress(b'XYZ')
        assert out == b'XYZ'
        assert decomp.needs_input is True

        decomp.reset()
        out = decomp.decompress(b'\x07XYZ', 1)
        assert out == b'X'
        assert decomp.needs_input is True
        out = decomp.decompress(b'')
        assert out == b'YZ'
        assert decomp.needs_input is True

        decomp.reset()
        out = decomp.decompress(b'\x01')
        assert out == b''
        assert decomp.needs_input is True
        out = decomp.decompress(b'a')
        assert out == b'a'
        assert decomp.needs_input is True

        decomp.reset()
        out = decomp.decompress(b'\x01a\xEE\xF0', 1)
        assert out == b'a'
        assert decomp.needs_input is True
        out = decomp.flush()
        assert out == b'aaa'
        assert decomp.needs_input is False


@pytest.mark.parametrize(compress_whole_tags, compress_whole_table, ids=compress_whole_ids)
def test_compress__table(
        ref_in: bytes,
        ref_out: bytes,
):
    ans_out = compress(ref_in)
    assert ans_out == ref_out


@pytest.mark.parametrize(decompress_whole_tags, decompress_whole_table, ids=decompress_whole_ids)
def test_decompress__table(
        ref_in: bytes,
        ref_out: bytes,
):
    ans_out = decompress(ref_in)
    assert ans_out == ref_out


class Test_LZSSWFile:

    def test___init___invalid_mode(self):

        with pytest.raises(ValueError, match='invalid mode'):
            LZSSWFile('dummy.bin', mode='?')

    def test___init___filename_path(self, datapath):

        filename = datapath / 'CP437.PBM.LZSSW'
        file = LZSSWFile(filename)
        assert file.fileno() >= 0
        filename = str(filename)
        file = LZSSWFile(filename)
        assert file.fileno() >= 0
        filename = filename.encode()
        file = LZSSWFile(filename)
        assert file.fileno() >= 0

    def test___init___filename_stream(self):

        stream = io.BytesIO(b'\x07XYZ')
        file = LZSSWFile(stream)
        with pytest.raises(OSError):
            file.fileno()

    def test_with(self):

        stream = io.BytesIO()
        with LZSSWFile(stream, mode='w') as file:
            file.write(b'aaaa')
            assert stream.tell() == 0
        assert stream.tell() == 4
        assert stream.getvalue() == b'\x01a\xEE\xF0'
        assert file.closed is True

    def test_close_read(self):

        stream = io.BytesIO(b'\x07XYZ')
        file = LZSSWFile(stream, mode='r')
        assert file._way == -1
        assert file._direct is False
        assert file._reader.closed is False
        assert file._stream.closed is False
        assert file.closed is False
        assert file.tell() == 0
        file.close()
        assert file._way == 0
        assert file._reader.closed is True
        assert file._stream.closed is False
        assert file.closed is True
        with pytest.raises(ValueError, match='closed'):
            file.tell()
        file.close()
        assert file._way == 0

    def test_close_write(self):

        stream = io.BytesIO()
        file = LZSSWFile(stream, mode='w')
        assert file._way == +1
        assert file._direct is False
        assert file._stream.closed is False
        assert file.closed is False
        assert file.tell() == 0
        file.write(b'a')
        assert stream.tell() == 0
        assert file.tell() == 1
        file.close()
        assert stream.tell() == 2
        assert stream.getvalue() == b'\x01a'
        assert file._way == 0
        assert file._stream.closed is False
        assert file.closed is True
        with pytest.raises(ValueError, match='closed'):
            file.tell()
        file.close()
        assert file._way == 0

    def test_read(self):

        stream = io.BytesIO(b'\x07XYZ')
        with LZSSWFile(stream, mode='r') as file:
            chunk = file.read(0)
            assert chunk == b''
            assert file.tell() == 0
            chunk = file.read(1)
            assert chunk == b'X'
            assert file.tell() == 1
            chunk = file.read()
            assert chunk == b'YZ'
            assert file.tell() == 3

    def test_read1(self):

        stream = io.BytesIO(b'\x07XYZ')
        with LZSSWFile(stream, mode='r') as file:
            chunk = file.read1(0)
            assert chunk == b''
            assert file.tell() == 0
            chunk = file.read1(1)
            assert chunk == b'X'
            assert file.tell() == 1
            chunk = file.read1()
            assert chunk == b'YZ'

    def test_readable(self):

        stream = io.BytesIO(b'\x07XYZ')
        with LZSSWFile(stream, mode='r') as file:
            assert file.readable() is True
            assert file.seekable() is True
            assert file.writable() is False
            with pytest.raises(io.UnsupportedOperation, match='not writable'):
                file.write(b'')
        with pytest.raises(ValueError, match='closed'):
            file.readable()
        with pytest.raises(ValueError, match='closed'):
            file.seekable()
        with pytest.raises(ValueError, match='closed'):
            file.writable()

    def test_readall(self):

        stream = io.BytesIO(b'\x07XYZ')
        with LZSSWFile(stream, mode='r') as file:
            chunk = file.readall()
            assert chunk == b'XYZ'
            assert file.tell() == 3

    def test_writable(self):

        stream = io.BytesIO()
        with LZSSWFile(stream, mode='w') as file:
            assert file.readable() is False
            assert file.seekable() is False
            assert file.writable() is True
            with pytest.raises(io.UnsupportedOperation, match='not readable'):
                file.read()
            with pytest.raises(io.UnsupportedOperation, match='not seekable'):
                file.seek(0)
        with pytest.raises(ValueError, match='closed'):
            file.readable()
        with pytest.raises(ValueError, match='closed'):
            file.seekable()
        with pytest.raises(ValueError, match='closed'):
            file.writable()

    def test_readline(self):

        stream = io.BytesIO(b'\x5FXYZ\na\xF2\xF0\n')
        with LZSSWFile(stream, mode='r') as file:
            line = file.readline(1)
            assert line == b'X'
            assert file.tell() == 1
            line = file.readline()
            assert line == b'YZ\n'
            assert file.tell() == 4
        with pytest.raises(ValueError, match='closed'):
            file.tell()

    def test_readlines(self):

        stream = io.BytesIO(b'\xDFXYZ\na\xF2\xF0\n1\x0723\n')
        with LZSSWFile(stream, mode='r') as file:
            lines = file.readlines(1)
            assert lines == [b'XYZ\n']
            assert file.tell() == 4
            lines = file.readlines()
            assert lines == [b'aaaa\n', b'123\n']
            assert file.tell() == 13
        with pytest.raises(ValueError, match='closed'):
            file.tell()

    def test_readinto(self):

        stream = io.BytesIO(b'\x07XYZ')
        buffer = bytearray(4)
        with LZSSWFile(stream, mode='r') as file:
            size = file.readinto(buffer)
            assert size == 3
            assert buffer == b'XYZ\0'

    def test_readinto1(self):

        stream = io.BytesIO(b'\x07XYZ')
        buffer = bytearray(4)
        with LZSSWFile(stream, mode='r') as file:
            size = file.readinto1(buffer)
            assert size == 3
            assert buffer == b'XYZ\0'

    def test_seek(self):

        stream = io.BytesIO(b'\x5FXYZ\na\xF2\xF0\n')
        with LZSSWFile(stream, mode='r') as file:
            tell = file.seek(0)
            assert tell == 0
            tell = file.seek(3, io.SEEK_SET)
            assert tell == 3
            chunk = file.read()
            assert chunk == b'\naaaa\n'
            tell = file.seek(-2, io.SEEK_END)
            assert tell == 7
            chunk = file.read()
            assert chunk == b'a\n'
            tell = file.seek(2, io.SEEK_SET)
            assert tell == 2
            chunk = file.read(1)
            assert chunk == b'Z'
            tell = file.seek(3, io.SEEK_CUR)
            assert tell == 6
            chunk = file.read(1)
            assert chunk == b'a'
            tell = file.seek(-5, io.SEEK_CUR)
            assert tell == 2
            chunk = file.read(3)
            assert chunk == b'Z\na'
            with pytest.raises(ValueError, match='whence'):
                file.seek(0, -1)
        with pytest.raises(ValueError, match='closed'):
            file.tell()

    def test_write(self):

        stream = io.BytesIO()
        with LZSSWFile(stream, mode='w') as file:
            size = file.write(b'XYZaaaa123')
            assert size == 10
            assert file.tell() == 10
            assert stream.getvalue() == b''
            size = file.write(b'aaa')
            assert size == 3
            assert file.tell() == 13
            assert stream.getvalue() == b''
            size = file.write(b'XYZ')
            assert size == 3
            assert file.tell() == 16
            assert stream.getvalue() == b''
            file.flush()
            assert file.tell() == 16
            assert stream.getvalue() == b''
        assert stream.getvalue() == b'\xEFXYZa\xF1\xF0123\x00\xF1\xF0\xEE\xF0'
        with pytest.raises(ValueError, match='closed'):
            file.tell()
        with pytest.raises(ValueError, match='closed'):
            file.write(b'')

    def test_writelines(self):

        stream = io.BytesIO()
        with LZSSWFile(stream, mode='w') as file:
            file.writelines([])
            assert file.tell() == 0
            assert stream.getvalue() == b''
            file.writelines([b''])
            assert file.tell() == 0
            assert stream.getvalue() == b''
            file.writelines([b'aaaa\n', b'XYZ\n'])
            assert file.tell() == 9
            assert stream.getvalue() == b''
            file.flush()
            assert file.tell() == 9
            assert stream.getvalue() == b''
        assert stream.getvalue() == b'\x7Da\xEE\xF0\nXYZ\n'
        with pytest.raises(ValueError, match='closed'):
            file.tell()
        with pytest.raises(ValueError, match='closed'):
            file.writelines([])


def test_open_rb():

    instream = io.BytesIO(b'\x1Da\xEE\xF0XYZ')
    with lzssw_open(instream, mode='rb') as file:
        outchunk = file.read(1)
        assert file.closed is False
        assert instream.tell() == 7
        assert outchunk == b'a'
    assert file.closed is True
    assert instream.tell() == 7


def test_open_rt():

    instream = io.BytesIO(b'\x1Da\xEE\xF0XYZ')
    with lzssw_open(instream, mode='rt') as file:
        outchunk = file.read(1)
        assert file.closed is False
        assert instream.tell() == 7
        assert outchunk == 'a'
    assert file.closed is True
    assert instream.tell() == 7


def test_open_wb():

    outstream = io.BytesIO()
    with lzssw_open(outstream, mode='wb') as file:
        file.write(b'aaaaXYZ')  # type: ignore
        assert file.closed is False
        assert outstream.tell() == 0
        assert outstream.getvalue() == b''
    assert file.closed is True
    assert outstream.tell() == 7
    assert outstream.getvalue() == b'\x1Da\xEE\xF0XYZ'


def test_open_wt():

    outstream = io.BytesIO()
    with lzssw_open(outstream, mode='wt') as file:
        file.write('aaaaXYZ')  # type: ignore
        assert file.closed is False
        assert outstream.tell() == 0
        assert outstream.getvalue() == b''
    assert file.closed is True
    assert outstream.tell() == 7
    assert outstream.getvalue() == b'\x1Da\xEE\xF0XYZ'


@pytest.fixture
def tmppath(tmpdir):
    return pathlib.Path(str(tmpdir))


@pytest.fixture(scope='module')
def datadir(request):
    dir_path, _ = os.path.splitext(request.module.__file__)
    assert os.path.isdir(str(dir_path))
    return dir_path


@pytest.fixture
def datapath(datadir):
    return pathlib.Path(str(datadir))


@pytest.mark.parametrize('filename', raw_filenames)
def test_compress__file(filename, datapath):

    inpath = datapath / filename
    refpath = datapath / f'{filename}.LZSSW'

    with open(inpath, 'rb') as instream:
        indata = instream.read()
    with open(refpath, 'rb') as refstream:
        refdata = refstream.read()

    outdata = compress(indata)
    assert outdata == refdata


@pytest.mark.parametrize('filename', raw_filenames)
def test_decompress__file(filename, datapath):

    inpath = datapath / f'{filename}.LZSSW'
    refpath = datapath / filename

    with open(inpath, 'rb') as instream:
        indata = instream.read()
    with open(refpath, 'rb') as refstream:
        refdata = refstream.read()

    outdata = decompress(indata)
    assert outdata == refdata


@pytest.mark.parametrize('filename', raw_filenames)
def test_main_compress(filename, tmppath, datapath):

    inpath = datapath / filename
    outpath = tmppath / f'{filename}.LZSSW'
    refpath = datapath / f'{filename}.LZSSW'
    _argv = sys.argv[:]
    try:
        sys.argv[:] = ['lzssw.py', str(inpath), str(outpath)]

        main()

        with open(outpath, 'rb') as outstream:
            outdata = outstream.read()
        with open(refpath, 'rb') as refstream:
            refdata = refstream.read()
        assert outdata == refdata

    finally:
        sys.argv[:] = _argv


@pytest.mark.parametrize('filename', raw_filenames)
def test_main_decompress(filename, tmppath, datapath):

    inpath = datapath / f'{filename}.LZSSW'
    outpath = tmppath / filename
    refpath = datapath / filename
    _argv = sys.argv[:]
    try:
        sys.argv[:] = ['lzssw.py', '-d', str(inpath), str(outpath)]

        main()

        with open(outpath, 'rb') as outstream:
            outdata = outstream.read()
        with open(refpath, 'rb') as refstream:
            refdata = refstream.read()
        assert outdata == refdata

    finally:
        sys.argv[:] = _argv


@pytest.mark.parametrize('commonbyte', [0x00, 0x20, 0xFF])
@pytest.mark.parametrize(('ringsize', 'maxmatchlen'), [
    (0x1000, 0x10),

    (0x0800, 0x20),
    (0x0800, 0x10),

    (0x0400, 0x40),
    (0x0400, 0x20),
    (0x0400, 0x10),

    (0x0200, 0x80),
    (0x0200, 0x40),
    (0x0200, 0x20),
    (0x0200, 0x10),
])
@pytest.mark.parametrize('filename', raw_filenames)
def test_main_compress_decompress(
        filename: str,
        ringsize: int,
        maxmatchlen: int,
        commonbyte: int,
        tmppath: pathlib.Path,
        datapath: pathlib.Path,
):
    inpath = datapath / filename
    outpath = tmppath / f'{filename}.LZSSW-{ringsize}-{maxmatchlen}-{commonbyte}'
    _argv = sys.argv[:]
    try:
        sys.argv[:] = [
            'lzssw.py',
            '-r', str(ringsize),
            '-m', str(maxmatchlen),
            '-f', str(commonbyte),
            str(inpath),
            str(outpath),
        ]

        main()

        sys.argv.insert(1, '-d')
        sys.argv[-2] = sys.argv[-1]
        sys.argv[-1] += '.BIN'

        main()

        with open(sys.argv[-1], 'rb') as outstream:
            outdata = outstream.read()
        with open(inpath, 'rb') as refstream:
            refdata = refstream.read()
        assert outdata == refdata

    finally:
        sys.argv[:] = _argv
