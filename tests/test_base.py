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

import pytest

from tinycompress.base import BUFFER_SIZE
from tinycompress.base import CodecFile
from tinycompress.base import DecompressorStream
from tinycompress.base import codec_open
from tinycompress.rleb import RLEBCompressor
from tinycompress.rleb import RLEBDecompressor


class Test_DecompressorStream:

    def test_read_zero(self):

        instream = io.BytesIO(b'\x02XYZ')
        decomp = RLEBDecompressor()
        outstream = DecompressorStream(instream, decomp)
        outchunk = outstream.read(0)
        assert outchunk == b''

    def test_seek(self):

        instream = io.BytesIO(b'\x03XYZ\n\x80a\x00\n')
        decomp = RLEBDecompressor()
        outstream = DecompressorStream(instream, decomp)
        with pytest.raises(ValueError, match='whence'):
            outstream.seek(0, -1)
        tell = outstream.seek(0)
        assert tell == 0
        tell = outstream.seek(3, io.SEEK_SET)
        assert tell == 3
        chunk = outstream.read()
        assert chunk == b'\naaa\n'
        tell = outstream.seek(-2, io.SEEK_END)
        assert tell == 6
        chunk = outstream.read()
        assert chunk == b'a\n'
        tell = outstream.seek(2, io.SEEK_SET)
        assert tell == 2
        chunk = outstream.read(1)
        assert chunk == b'Z'
        tell = outstream.seek(3, io.SEEK_CUR)
        assert tell == 6
        chunk = outstream.read(1)
        assert chunk == b'a'
        tell = outstream.seek(-5, io.SEEK_CUR)
        assert tell == 2
        chunk = outstream.read(3)
        assert chunk == b'Z\na'
        chunk = b'\x01.|' * BUFFER_SIZE
        instream.seek(0)
        instream.write(chunk)
        outstream.seek(0, io.SEEK_SET)
        tell = outstream.seek(0, io.SEEK_END)
        assert tell == 2 * BUFFER_SIZE


class Test_CodecFile:

    def test___init___invalid_mode(self):

        comp = RLEBCompressor()
        decomp = RLEBDecompressor()
        with pytest.raises(ValueError, match='invalid mode'):
            CodecFile('dummy.bin', mode='?', comp=comp, decomp=decomp)

    def test___init___invalid_read_decomp(self):

        with pytest.raises(ValueError, match='decompressor object required'):
            CodecFile('dummy.bin', mode='r', comp=None, decomp=None)

    def test___init___invalid_write_comp(self):

        with pytest.raises(ValueError, match='compressor object required'):
            CodecFile('dummy.bin', mode='w', comp=None, decomp=None)

    @pytest.mark.parametrize('filename', [(), [], {}, set(), object()])
    def test___init___invalid_filename(self, filename):

        comp = RLEBCompressor()
        decomp = RLEBDecompressor()
        with pytest.raises(TypeError, match='filename must be .* object'):
            CodecFile(filename, mode='r', comp=comp, decomp=decomp)


def test_codec_open_invalid():

    with pytest.raises(ValueError, match='invalid mode'):
        codec_open('dummy.bin', mode='rbt')

    with pytest.raises(ValueError, match="argument 'encoding' not supported in binary mode"):
        codec_open('dummy.bin', mode='rb', encoding='ascii')

    with pytest.raises(ValueError, match="argument 'errors' not supported in binary mode"):
        codec_open('dummy.bin', mode='rb', errors='strict')

    with pytest.raises(ValueError, match="argument 'newline' not supported in binary mode"):
        codec_open('dummy.bin', mode='rb', newline='\n')
