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

from tinycompress.rleb import RLEBCompressor
from tinycompress.rleb import RLEBDecompressor
from tinycompress.rleb import RLEBException
from tinycompress.rleb import RLEBFile
from tinycompress.rleb import compress
from tinycompress.rleb import decompress
from tinycompress.rleb import main
from tinycompress.rleb import open as rleb_open

lorem = b"""\
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
tempor incidunt ut labore et dolore magna aliqua.  Ut enim ad minim veniam,
quis nostrum exercitationem ullamco laboriosam, nisi ut aliquid ex ea
commodi consequatur.  Duis aute irure reprehenderit in voluptate velit esse
cillum dolore eu fugiat nulla pariatur.  Excepteur sint obcaecat cupiditat
non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

compress_partial_tags = ('ref_in', 'ref_out', 'ref_res')
compress_partial_table = [
    (b'',     b'',  b''),
    (b'X',    b'',  b'X'),
    (b'XYZ',  b'',  b'XYZ'),

    (b'aaa',    b'',       b'aaa'),
    (b'aaaa',   b'',       b'aaaa'),
    (129*b'a',  b'',       129*b'a'),
    (130*b'a',  b'\xFFa',  b''),
    (131*b'a',  b'\xFFa',  b'a'),

    (b'XYZaa',         b'',              b'XYZaa'),
    (b'XYZaaa',        b'\x02XYZ',       b'aaa'),
    (b'XYZaaaa',       b'\x02XYZ',       b'aaaa'),
    (b'XYZ'+129*b'a',  b'\x02XYZ',       129*b'a'),
    (b'XYZ'+130*b'a',  b'\x02XYZ\xFFa',  b''),
    (b'XYZ'+131*b'a',  b'\x02XYZ\xFFa',  b'a'),

    (b'aaXYZ',         b'',       b'aaXYZ'),
    (b'aaaXYZ',        b'\x80a',  b'XYZ'),
    (b'aaaaXYZ',       b'\x81a',  b'XYZ'),
    (129*b'a'+b'XYZ',  b'\xFEa',  b'XYZ'),
    (130*b'a'+b'XYZ',  b'\xFFa',  b'XYZ'),
    (131*b'a'+b'XYZ',  b'\xFFa',  b'aXYZ'),

    (b'XYZaaXYZ',             b'',              b'XYZaaXYZ'),
    (b'XYZaaaXYZ',            b'\x02XYZ\x80a',  b'XYZ'),
    (b'XYZaaaaXYZ',           b'\x02XYZ\x81a',  b'XYZ'),
    (b'XYZ'+129*b'a'+b'XYZ',  b'\x02XYZ\xFEa',  b'XYZ'),
    (b'XYZ'+130*b'a'+b'XYZ',  b'\x02XYZ\xFFa',  b'XYZ'),
    (b'XYZ'+131*b'a'+b'XYZ',  b'\x02XYZ\xFFa',  b'aXYZ'),

    (b'aaXYZaa',                b'',                    b'aaXYZaa'),
    (b'aaaXYZaaa',              b'\x80a\x02XYZ',        b'aaa'),
    (b'aaaaXYZaaaa',            b'\x81a\x02XYZ',        b'aaaa'),
    (129*b'a'+b'XYZ'+129*b'a',  b'\xFEa\x02XYZ',        129*b'a'),
    (130*b'a'+b'XYZ'+130*b'a',  b'\xFFa\x02XYZ\xFFa',   b''),
    (131*b'a'+b'XYZ'+131*b'a',  b'\xFFa\x03aXYZ\xFFa',  b'a'),

    (63*b'Xa',       b'',               63*b'Xa'),
    (63*b'Xa'+b'X',  b'',               63*b'Xa'+b'X'),
    (64*b'Xa',       b'\x7F'+64*b'Xa',  b''),
    (64*b'Xa'+b'X',  b'\x7F'+64*b'Xa',  b'X'),
    (65*b'Xa',       b'\x7F'+64*b'Xa',  b'Xa'),

    (63*b'XY'+63*b'a',  b'\x7D'+63*b'XY',  63*b'a'),

    (lorem,
        (b'\x7FLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod\ntempor incidunt ut labore et dolore magna aliqua.  Ut en'
        b'\x7Fim ad minim veniam,\nquis nostrum exercitationem ullamco laboriosam, nisi ut aliquid ex ea\ncommodi consequatur.  Duis aute irure '
        b'\x7Freprehenderit in voluptate velit esse\ncillum dolore eu fugiat nulla pariatur.  Excepteur sint obcaecat cupiditat\nnon proident, s'),
        b'unt in culpa qui officia deserunt mollit anim id est laborum.\n'),
]
compress_partial_ids = [repr(v[0]) for v in compress_partial_table]

flush_tags = ('ref_in', 'ref_out')
flush_table = [
    (b'',     b''),
    (b'X',    b'\x00X'),
    (b'XYZ',  b'\x02XYZ'),

    (b'aaa',    b'\x80a'),
    (b'aaaa',   b'\x81a'),
    (129*b'a',  b'\xFEa'),
    (130*b'a',  b'\xFFa'),
    (131*b'a',  b'\xFFa\x00a'),

    (b'XYZaa',         b'\x04XYZaa'),
    (b'XYZaaa',        b'\x02XYZ\x80a'),
    (b'XYZaaaa',       b'\x02XYZ\x81a'),
    (b'XYZ'+129*b'a',  b'\x02XYZ\xFEa'),
    (b'XYZ'+130*b'a',  b'\x02XYZ\xFFa'),
    (b'XYZ'+131*b'a',  b'\x02XYZ\xFFa\x00a'),

    (b'aaXYZ',         b'\x04aaXYZ'),
    (b'aaaXYZ',        b'\x80a\x02XYZ'),
    (b'aaaaXYZ',       b'\x81a\x02XYZ'),
    (129*b'a'+b'XYZ',  b'\xFEa\x02XYZ'),
    (130*b'a'+b'XYZ',  b'\xFFa\x02XYZ'),
    (131*b'a'+b'XYZ',  b'\xFFa\x03aXYZ'),

    (b'XYZaaXYZ',             b'\x07XYZaaXYZ'),
    (b'XYZaaaXYZ',            b'\x02XYZ\x80a\x02XYZ'),
    (b'XYZaaaaXYZ',           b'\x02XYZ\x81a\x02XYZ'),
    (b'XYZ'+129*b'a'+b'XYZ',  b'\x02XYZ\xFEa\x02XYZ'),
    (b'XYZ'+130*b'a'+b'XYZ',  b'\x02XYZ\xFFa\x02XYZ'),
    (b'XYZ'+131*b'a'+b'XYZ',  b'\x02XYZ\xFFa\x03aXYZ'),

    (b'aaXYZaa',                b'\x06aaXYZaa'),
    (b'aaaXYZaaa',              b'\x80a\x02XYZ\x80a'),
    (b'aaaaXYZaaaa',            b'\x81a\x02XYZ\x81a'),
    (129*b'a'+b'XYZ'+129*b'a',  b'\xFEa\x02XYZ\xFEa'),
    (130*b'a'+b'XYZ'+130*b'a',  b'\xFFa\x02XYZ\xFFa'),
    (131*b'a'+b'XYZ'+131*b'a',  b'\xFFa\x03aXYZ\xFFa\x00a'),

    (63*b'Xa',       b'\x7D'+63*b'Xa'),
    (63*b'Xa'+b'X',  b'\x7E'+63*b'Xa'+b'X'),
    (64*b'Xa',       b'\x7F'+64*b'Xa'),
    (64*b'Xa'+b'X',  b'\x7F'+64*b'Xa'+b'\x00X'),
    (65*b'Xa',       b'\x7F'+64*b'Xa'+b'\x01Xa'),

    (lorem,
     (b'\x7FLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod\ntempor incidunt ut labore et dolore magna aliqua.  Ut en'
      b'\x7Fim ad minim veniam,\nquis nostrum exercitationem ullamco laboriosam, nisi ut aliquid ex ea\ncommodi consequatur.  Duis aute irure '
      b'\x7Freprehenderit in voluptate velit esse\ncillum dolore eu fugiat nulla pariatur.  Excepteur sint obcaecat cupiditat\nnon proident, s'
      b'\x3Dunt in culpa qui officia deserunt mollit anim id est laborum.\n')),
]
flush_ids = [repr(v[0]) for v in flush_table]

compress_whole_tags = ('ref_in', 'ref_out')
compress_whole_table = [
    (b'',     b''),
    (b'X',    b'\x00X'),
    (b'XYZ',  b'\x02XYZ'),

    (b'aaa',    b'\x80a'),
    (b'aaaa',   b'\x81a'),
    (129*b'a',  b'\xFEa'),
    (130*b'a',  b'\xFFa'),
    (131*b'a',  b'\xFFa\x00a'),

    (b'XYZaa',         b'\x04XYZaa'),
    (b'XYZaaa',        b'\x02XYZ\x80a'),
    (b'XYZaaaa',       b'\x02XYZ\x81a'),
    (b'XYZ'+129*b'a',  b'\x02XYZ\xFEa'),
    (b'XYZ'+130*b'a',  b'\x02XYZ\xFFa'),
    (b'XYZ'+131*b'a',  b'\x02XYZ\xFFa\x00a'),

    (b'aaXYZ',         b'\x04aaXYZ'),
    (b'aaaXYZ',        b'\x80a\x02XYZ'),
    (b'aaaaXYZ',       b'\x81a\x02XYZ'),
    (129*b'a'+b'XYZ',  b'\xFEa\x02XYZ'),
    (130*b'a'+b'XYZ',  b'\xFFa\x02XYZ'),
    (131*b'a'+b'XYZ',  b'\xFFa\x03aXYZ'),

    (b'XYZaaXYZ',             b'\x07XYZaaXYZ'),
    (b'XYZaaaXYZ',            b'\x02XYZ\x80a\x02XYZ'),
    (b'XYZaaaaXYZ',           b'\x02XYZ\x81a\x02XYZ'),
    (b'XYZ'+129*b'a'+b'XYZ',  b'\x02XYZ\xFEa\x02XYZ'),
    (b'XYZ'+130*b'a'+b'XYZ',  b'\x02XYZ\xFFa\x02XYZ'),
    (b'XYZ'+131*b'a'+b'XYZ',  b'\x02XYZ\xFFa\x03aXYZ'),

    (b'aaXYZaa',                b'\x06aaXYZaa'),
    (b'aaaXYZaaa',              b'\x80a\x02XYZ\x80a'),
    (b'aaaaXYZaaaa',            b'\x81a\x02XYZ\x81a'),
    (129*b'a'+b'XYZ'+129*b'a',  b'\xFEa\x02XYZ\xFEa'),
    (130*b'a'+b'XYZ'+130*b'a',  b'\xFFa\x02XYZ\xFFa'),
    (131*b'a'+b'XYZ'+131*b'a',  b'\xFFa\x03aXYZ\xFFa\x00a'),

    (63*b'Xa',       b'\x7D'+63*b'Xa'),
    (63*b'Xa'+b'X',  b'\x7E'+63*b'Xa'+b'X'),
    (64*b'Xa',       b'\x7F'+64*b'Xa'),
    (64*b'Xa'+b'X',  b'\x7F'+64*b'Xa'+b'\x00X'),
    (65*b'Xa',       b'\x7F'+64*b'Xa'+b'\x01Xa'),

    (63*b'XY'+63*b'a',  b'\x7D'+63*b'XY'+b'\xBCa'),

    (lorem,
     (b'\x7FLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod\ntempor incidunt ut labore et dolore magna aliqua.  Ut en'
      b'\x7Fim ad minim veniam,\nquis nostrum exercitationem ullamco laboriosam, nisi ut aliquid ex ea\ncommodi consequatur.  Duis aute irure '
      b'\x7Freprehenderit in voluptate velit esse\ncillum dolore eu fugiat nulla pariatur.  Excepteur sint obcaecat cupiditat\nnon proident, s'
      b'\x3Dunt in culpa qui officia deserunt mollit anim id est laborum.\n')),
]
compress_whole_ids = [repr(v[0]) for v in compress_whole_table]

decompress_partial_tags = ('max_length', 'ref_in', 'ref_out', 'rle', 'more', 'prev', 'ahead')
decompress_partial_table = [
    ( 0,  b'',  b'',  False,  0,  -1,  b''),
    ( 1,  b'',  b'',  False,  0,  -1,  b''),
    (-1,  b'',  b'',  False,  0,  -1,  b''),

    ( 0,  b'\x00'+1*b'a',  0*b'a',  False,  0,        -1,  b'\x00'+1*b'a'),
    ( 1,  b'\x00'+1*b'a',  1*b'a',  False,  0,  ord('a'),  0*b'a'),
    (-1,  b'\x00'+1*b'a',  1*b'a',  False,  0,        -1,  0*b'a'),

    ( 0,  b'\x01'+2*b'a',  0*b'a',  False,  0,        -1,  b'\x01'+2*b'a'),
    ( 1,  b'\x01'+2*b'a',  1*b'a',  False,  1,  ord('a'),  1*b'a'),
    ( 2,  b'\x01'+2*b'a',  2*b'a',  False,  0,  ord('a'),  0*b'a'),
    (-1,  b'\x01'+2*b'a',  2*b'a',  False,  0,        -1,  0*b'a'),

    (  0,  b'\x7E'+127*b'a',    0*b'a',  False,    0,        -1,  b'\x7E'+127*b'a'),
    (  1,  b'\x7E'+127*b'a',    1*b'a',  False,  126,  ord('a'),  126*b'a'),
    (126,  b'\x7E'+127*b'a',  126*b'a',  False,    1,  ord('a'),    1*b'a'),
    (127,  b'\x7E'+127*b'a',  127*b'a',  False,    0,  ord('a'),    0*b'a'),
    ( -1,  b'\x7E'+127*b'a',  127*b'a',  False,    0,        -1,    0*b'a'),

    (  0,  b'\x7F'+128*b'a',    0*b'a',  False,    0,        -1,  b'\x7F'+128*b'a'),
    (  1,  b'\x7F'+128*b'a',    1*b'a',  False,  127,  ord('a'),  127*b'a'),
    (127,  b'\x7F'+128*b'a',  127*b'a',  False,    1,  ord('a'),    1*b'a'),
    (128,  b'\x7F'+128*b'a',  128*b'a',  False,    0,  ord('a'),    0*b'a'),
    ( -1,  b'\x7F'+128*b'a',  128*b'a',  False,    0,        -1,    0*b'a'),

    ( 0,  b'\x80a',  0*b'a',  False,  0,        -1,  b'\x80a'),
    ( 1,  b'\x80a',  1*b'a',   True,  2,  ord('a'),  b''),
    ( 2,  b'\x80a',  2*b'a',   True,  1,  ord('a'),  b''),
    ( 3,  b'\x80a',  3*b'a',   True,  0,  ord('a'),  b''),
    (-1,  b'\x80a',  3*b'a',  False,  0,        -1,  b''),

    ( 0,  b'\x81a',  0*b'a',  False,  0,        -1,  b'\x81a'),
    ( 1,  b'\x81a',  1*b'a',   True,  3,  ord('a'),  b''),
    ( 3,  b'\x81a',  3*b'a',   True,  1,  ord('a'),  b''),
    ( 4,  b'\x81a',  4*b'a',   True,  0,  ord('a'),  b''),
    (-1,  b'\x81a',  4*b'a',  False,  0,        -1,  b''),

    (  0,  b'\xFEa',    0*b'a',  False,    0,        -1,  b'\xFEa'),
    (  1,  b'\xFEa',    1*b'a',   True,  128,  ord('a'),  b''),
    (128,  b'\xFEa',  128*b'a',   True,    1,  ord('a'),  b''),
    (129,  b'\xFEa',  129*b'a',   True,    0,  ord('a'),  b''),
    ( -1,  b'\xFEa',  129*b'a',  False,    0,        -1,  b''),

    (  0,  b'\xFFa',    0*b'a',  False,    0,        -1,  b'\xFFa'),
    (  1,  b'\xFFa',    1*b'a',   True,  129,  ord('a'),  b''),
    (129,  b'\xFFa',  129*b'a',   True,    1,  ord('a'),  b''),
    (130,  b'\xFFa',  130*b'a',   True,    0,  ord('a'),  b''),
    ( -1,  b'\xFFa',  130*b'a',  False,    0,        -1,  b''),

    (3+  0,  b'\x02XYZ\xFFa',  b'XYZ'+  0*b'a',   False,    0,  ord('Z'),  b'\xFFa'),
    (3+  1,  b'\x02XYZ\xFFa',  b'XYZ'+  1*b'a',    True,  129,  ord('a'),  b''),
    (3+129,  b'\x02XYZ\xFFa',  b'XYZ'+129*b'a',    True,    1,  ord('a'),  b''),
    (3+130,  b'\x02XYZ\xFFa',  b'XYZ'+130*b'a',    True,    0,  ord('a'),  b''),
    (   -1,  b'\x02XYZ\xFFa',  b'XYZ'+130*b'a',   False,    0,        -1,  b''),
]
decompress_partial_ids = [repr(v[:2]) for v in decompress_partial_table]

decompress_whole_tags = ('ref_in', 'ref_out')
decompress_whole_table = [
    (b'',                   b''),
    (b'\x00'+1*b'a',      1*b'a'),
    (b'\x01'+2*b'a',      2*b'a'),
    (b'\x7E'+127*b'a',  127*b'a'),
    (b'\x7F'+128*b'a',  128*b'a'),
    (b'\x80a',            3*b'a'),
    (b'\x81a',            4*b'a'),
    (b'\xFEa',          129*b'a'),
    (b'\xFFa',          130*b'a'),
    (b'\x02XYZ\xFFa',   b'XYZ'+130*b'a'),
]
decompress_whole_ids = [repr(v[0]) for v in decompress_whole_table]

raw_filenames = [
    'CP437.PBM',
    'LOREM.TXT',
]


class Test_RLEBCompressor:

    @pytest.mark.parametrize('minsame', [3, 64, 128])
    def test___init___minsame_pass(self, minsame: int):

        RLEBCompressor(minsame=minsame)

    @pytest.mark.parametrize('minsame', [2, 129])
    def test___init___minsame_raises(self, minsame: int):

        with pytest.raises(ValueError, match='minsame'):
            RLEBCompressor(minsame=minsame)

    @pytest.mark.parametrize('minpast', [0, 64, 127])
    def test___init___minpast_pass(self, minpast: int):

        RLEBCompressor(minpast=minpast)

    @pytest.mark.parametrize('minpast', [-1, 128])
    def test___init___minpast_raises(self, minpast: int):

        with pytest.raises(ValueError, match='minpast'):
            RLEBCompressor(minpast=minpast)

    @pytest.mark.parametrize(compress_partial_tags, compress_partial_table, ids=compress_partial_ids)
    def test_compress__table(
            self,
            ref_in: bytes,
            ref_out: bytes,
            ref_res: bytes,
    ):  # TODO: test instance variables as per test_decompress__table()

        comp = RLEBCompressor()
        ans_out = comp.compress(ref_in)
        ans_res = bytearray()
        head = comp._head

        while head != comp._tail:
            ans_res.append(comp._ring[head])
            head = (head + 1) & 0xFF

        assert ans_out == ref_out
        assert ans_res == ref_res

    def test_compress__already_flushed(self):

        comp = RLEBCompressor()
        comp.compress(b'...')
        comp.flush()
        with pytest.raises(RLEBException, match='already flushed'):
            comp.compress(b'...')

    @pytest.mark.parametrize(flush_tags, flush_table, ids=flush_ids)
    def test_flush__table(
            self,
            ref_in: bytes,
            ref_out: bytes,
    ):  # TODO: test instance variables as per test_decompress__table()

        comp = RLEBCompressor()
        ans_out = comp.compress(ref_in)
        ans_out += comp.flush()
        assert ans_out == ref_out
        assert comp._head == comp._tail
        ans_out += comp.flush()
        assert ans_out == ref_out
        assert comp._head == comp._tail

    def test_reset(self):

        comp = RLEBCompressor()
        comp.compress(b'aaa')
        assert comp._same == 3
        assert comp._tail - comp._head == 3
        comp.reset()
        assert comp._same == 0
        assert comp._tail - comp._head == 0
        comp.compress(b'aaaaa')
        assert comp._same == 5
        assert comp._tail - comp._head == 5
        assert comp.eof is False
        comp.flush()
        assert comp.eof is True
        assert comp._same == 0
        assert comp._tail - comp._head == 0
        comp.reset()
        assert comp.eof is False
        assert comp._same == 0
        assert comp._tail - comp._head == 0
        comp.compress(b'XYZ')
        assert comp._same == 1
        assert comp._tail - comp._head == 3

    def test_eof(self):

        comp = RLEBCompressor()
        assert comp.eof is False
        comp.compress(b'...')
        assert comp.eof is False
        comp.flush()
        assert comp.eof is True


class Test_RLEBDecompressor:

    @pytest.mark.parametrize('minsame', [3, 64, 128])
    def test___init___minsame_pass(self, minsame: int):

        RLEBDecompressor(minsame=minsame)

    @pytest.mark.parametrize('minsame', [2, 129])
    def test___init___minsame_raises(self, minsame: int):

        with pytest.raises(ValueError, match='minsame'):
            RLEBDecompressor(minsame=minsame)

    @pytest.mark.parametrize(decompress_partial_tags, decompress_partial_table, ids=decompress_partial_ids)
    def test_decompress__table(
            self,
            max_length: int,
            ref_in: bytes,
            ref_out: bytes,
            rle: bool,
            more: int,
            prev: int,
            ahead: bytes,
    ):
        decomp = RLEBDecompressor()
        ans_out = decomp.decompress(ref_in, max_length)
        assert ans_out == ref_out
        assert decomp._rle == rle
        assert decomp._more == more
        assert decomp._prev == prev
        assert decomp._ahead == ahead

    @pytest.mark.parametrize('minsame', [3, 4, 5, 6, 7, 8, 16, 32, 64, 128])
    def test_decompress__minsame(self, minsame: int):
        decomp = RLEBDecompressor(minsame=minsame)

        ans_out = decomp.decompress(b'\x00b\x80a\x00c')
        ans_ref = b'b' + (b'a' * minsame) + b'c'
        assert ans_out == ans_ref

        ans_out = decomp.decompress(b'\x00b\xFFa\x00c')
        ans_ref = b'b' + (b'a' * (minsame + 0x7F)) + b'c'
        assert ans_out == ans_ref

    def test_decompress__already_flushed(self):

        decomp = RLEBDecompressor()
        decomp.decompress(b'\x02XYZ')
        decomp.flush()
        with pytest.raises(RLEBException, match='already flushed'):
            decomp.decompress(b'\x80a')

    def test_decompress__ahead__diff(self):

        decomp = RLEBDecompressor()
        ans_out = decomp.decompress(b'\x02XYZ', 1)
        assert ans_out == b'X'
        ans_out = decomp.decompress(b'')
        assert ans_out == b'YZ'

    def test_decompress__ahead__same(self):

        decomp = RLEBDecompressor()
        ans_out = decomp.decompress(b'\x81a', 1)
        assert ans_out == b'a'
        ans_out = decomp.decompress(b'')
        assert ans_out == b'aaa'

    def test_flush__empty(self):

        decomp = RLEBDecompressor()
        ans_out = decomp.decompress(b'')
        assert ans_out == b''
        ans_out = decomp.flush()
        assert ans_out == b''
        ans_out = decomp.flush()
        assert ans_out == b''

    def test_flush__diff_ahead(self):

        decomp = RLEBDecompressor()
        ans_out = decomp.decompress(b'\x02XYZ', 1)
        assert ans_out == b'X'
        ans_out = decomp.flush()
        assert ans_out == b'YZ'
        ans_out = decomp.flush()
        assert ans_out == b''

    def test_flush__diff_early(self):

        decomp = RLEBDecompressor()
        ans_out = decomp.decompress(b'\x02X')
        assert ans_out == b'X'
        ans_out = decomp.decompress(b'YZ')
        assert ans_out == b'YZ'
        ans_out = decomp.flush()
        assert ans_out == b''

    def test_flush__same_ahead(self):

        decomp = RLEBDecompressor()
        ans_out = decomp.decompress(b'\x81a', 1)
        assert ans_out == b'a'
        ans_out = decomp.decompress(b'', 1)
        assert ans_out == b'a'
        ans_out = decomp.flush()
        assert ans_out == b'aa'

    def test_flush__same_early(self):

        decomp = RLEBDecompressor()
        ans_out = decomp.decompress(b'\x80')
        assert ans_out == b''
        ans_out = decomp.decompress(b'a')
        assert ans_out == b'aaa'
        ans_out = decomp.flush()
        assert ans_out == b''

    def test_reset(self):

        decomp = RLEBDecompressor()
        decomp.decompress(b'\x02XYZ', 1)
        assert decomp._more == 2
        assert decomp._ahead == b'YZ'
        decomp.reset()
        assert decomp._more == 0
        assert decomp._ahead == b''
        decomp.decompress(b'\x80a', 2)
        assert decomp._more == 1
        assert decomp.eof is False
        decomp.flush()
        assert decomp._more == 0
        assert decomp.eof is True
        decomp.reset()
        assert decomp.eof is False
        assert decomp._more == 0
        decomp.decompress(b'\x80a')

    def test_eof(self):

        decomp = RLEBDecompressor()
        assert decomp.eof is False
        decomp.decompress(b'\x00.')
        assert decomp.eof is False
        decomp.flush()
        assert decomp.eof is True

    def test_unused_data(self):

        decomp = RLEBDecompressor()
        assert decomp.unused_data == b''
        decomp.decompress(b'\x02XYZ', 1)
        assert decomp.unused_data == b''
        decomp.flush()
        assert decomp.unused_data == b''

    def test_needs_input(self):

        decomp = RLEBDecompressor()
        assert decomp.needs_input is False

        out = decomp.decompress(b'')
        assert out == b''
        assert decomp.needs_input is False

        out = decomp.decompress(b'\x02')
        assert out == b''
        assert decomp.needs_input is True
        out = decomp.decompress(b'XYZ')
        assert out == b'XYZ'
        assert decomp.needs_input is False

        out = decomp.decompress(b'\x02XYZ', 1)
        assert out == b'X'
        assert decomp.needs_input is False
        out = decomp.decompress(b'')
        assert out == b'YZ'
        assert decomp.needs_input is False

        out = decomp.decompress(b'\x80')
        assert out == b''
        assert decomp.needs_input is True
        out = decomp.decompress(b'a')
        assert out == b'aaa'
        assert decomp.needs_input is False

        out = decomp.decompress(b'\x80a', 1)
        assert out == b'a'
        assert decomp.needs_input is False
        out = decomp.flush()
        assert out == b'aa'
        assert decomp.needs_input is False


@pytest.mark.parametrize(compress_whole_tags, compress_whole_table, ids=compress_whole_ids)
def test_compress__table(
        ref_in: bytes,
        ref_out: bytes,
):
    ans_out = compress(ref_in)
    assert ans_out == ref_out


def test_compress__minsame():
    ref_in = b'a' * 37
    ref_out = b'\x9Da'
    ans_out = compress(ref_in, minsame=8)
    assert ans_out == ref_out


def test_compress__minpast():
    ref_in = b'a' * 37
    ref_out = b'\x07aaaaaaaa\x9Aa'
    ans_out = compress(ref_in, minpast=8)
    assert ans_out == ref_out


def test_compress__minsame_minpast():
    ref_in = b'a' * 37
    ref_out = b'\x07aaaaaaaa\x95a'
    ans_out = compress(ref_in, minsame=8, minpast=8)
    assert ans_out == ref_out


@pytest.mark.parametrize(decompress_whole_tags, decompress_whole_table, ids=decompress_whole_ids)
def test_decompress__table(
        ref_in: bytes,
        ref_out: bytes,
):
    ans_out = decompress(ref_in)
    assert ans_out == ref_out


class Test_RLEBFile:

    def test___init___invalid_mode(self):

        with pytest.raises(ValueError, match='invalid mode'):
            RLEBFile('dummy.bin', mode='?')

    def test___init___filename_path(self, datapath):

        filename = datapath / 'CP437.PBM.RLEB'
        file = RLEBFile(filename)
        assert file.fileno() >= 0
        filename = str(filename)
        file = RLEBFile(filename)
        assert file.fileno() >= 0
        filename = filename.encode()
        file = RLEBFile(filename)
        assert file.fileno() >= 0

    def test___init___filename_stream(self):

        stream = io.BytesIO(b'\x02XYZ')
        file = RLEBFile(stream)
        with pytest.raises(OSError):
            file.fileno()

    def test_with(self):

        stream = io.BytesIO()
        with RLEBFile(stream, mode='w') as file:
            file.write(b'aaa')
            assert stream.tell() == 0
        assert stream.tell() == 2
        assert stream.getvalue() == b'\x80a'
        assert file.closed is True

    def test_close_read(self):

        stream = io.BytesIO(b'\x02XYZ')
        file = RLEBFile(stream, mode='r')
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
        file = RLEBFile(stream, mode='w')
        assert file._way == +1
        assert file._direct is False
        assert file._stream.closed is False
        assert file.closed is False
        assert file.tell() == 0
        file.write(b'aaa')
        assert stream.tell() == 0
        assert file.tell() == 3
        file.close()
        assert stream.tell() == 2
        assert stream.getvalue() == b'\x80a'
        assert file._way == 0
        assert file._stream.closed is False
        assert file.closed is True
        with pytest.raises(ValueError, match='closed'):
            file.tell()
        file.close()
        assert file._way == 0

    def test_read(self):

        stream = io.BytesIO(b'\x02XYZ')
        with RLEBFile(stream, mode='r') as file:
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

        stream = io.BytesIO(b'\x02XYZ')
        with RLEBFile(stream, mode='r') as file:
            chunk = file.read1(0)
            assert chunk == b''
            assert file.tell() == 0
            chunk = file.read1(1)
            assert chunk == b'X'
            assert file.tell() == 1
            chunk = file.read1()
            assert chunk == b'YZ'

    def test_readable(self):

        stream = io.BytesIO(b'\x02XYZ')
        with RLEBFile(stream, mode='r') as file:
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

        stream = io.BytesIO(b'\x02XYZ')
        with RLEBFile(stream, mode='r') as file:
            chunk = file.readall()
            assert chunk == b'XYZ'
            assert file.tell() == 3

    def test_writable(self):

        stream = io.BytesIO()
        with RLEBFile(stream, mode='w') as file:
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

        stream = io.BytesIO(b'\x03XYZ\n\x80a\x00\n')
        with RLEBFile(stream, mode='r') as file:
            line = file.readline(1)
            assert line == b'X'
            assert file.tell() == 1
            line = file.readline()
            assert line == b'YZ\n'
            assert file.tell() == 4
        with pytest.raises(ValueError, match='closed'):
            file.tell()

    def test_readlines(self):

        stream = io.BytesIO(b'\x03XYZ\n\x80a\x04\n123\n')
        with RLEBFile(stream, mode='r') as file:
            lines = file.readlines(1)
            assert lines == [b'XYZ\n']
            assert file.tell() == 4
            lines = file.readlines()
            assert lines == [b'aaa\n', b'123\n']
            assert file.tell() == 12
        with pytest.raises(ValueError, match='closed'):
            file.tell()

    def test_readinto(self):

        stream = io.BytesIO(b'\x02XYZ')
        buffer = bytearray(4)
        with RLEBFile(stream, mode='r') as file:
            size = file.readinto(buffer)
            assert size == 3
            assert buffer == b'XYZ\0'

    def test_readinto1(self):

        stream = io.BytesIO(b'\x02XYZ')
        buffer = bytearray(4)
        with RLEBFile(stream, mode='r') as file:
            size = file.readinto1(buffer)
            assert size == 3
            assert buffer == b'XYZ\0'

    def test_seek(self):

        stream = io.BytesIO(b'\x03XYZ\n\x80a\x00\n')
        with RLEBFile(stream, mode='r') as file:
            tell = file.seek(0)
            assert tell == 0
            tell = file.seek(3, io.SEEK_SET)
            assert tell == 3
            chunk = file.read()
            assert chunk == b'\naaa\n'
            tell = file.seek(-2, io.SEEK_END)
            assert tell == 6
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
        with RLEBFile(stream, mode='w') as file:
            size = file.write(b'')
            assert size == 0
            assert file.tell() == 0
            assert stream.getvalue() == b''
            size = file.write(b'aaa')
            assert size == 3
            assert file.tell() == 3
            assert stream.getvalue() == b''
            size = file.write(b'XYZ')
            assert size == 3
            assert file.tell() == 6
            assert stream.getvalue() == b'\x80a'
            file.flush()
            assert file.tell() == 6
            assert stream.getvalue() == b'\x80a'
        assert stream.getvalue() == b'\x80a\x02XYZ'
        with pytest.raises(ValueError, match='closed'):
            file.tell()
        with pytest.raises(ValueError, match='closed'):
            file.write(b'')

    def test_writelines(self):

        stream = io.BytesIO()
        with RLEBFile(stream, mode='w') as file:
            file.writelines([])
            assert file.tell() == 0
            assert stream.getvalue() == b''
            file.writelines([b''])
            assert file.tell() == 0
            assert stream.getvalue() == b''
            file.writelines([b'aaa\n', b'XYZ\n'])
            assert file.tell() == 8
            assert stream.getvalue() == b'\x80a'
            file.flush()
            assert file.tell() == 8
            assert stream.getvalue() == b'\x80a'
        assert stream.getvalue() == b'\x80a\x04\nXYZ\n'
        with pytest.raises(ValueError, match='closed'):
            file.tell()
        with pytest.raises(ValueError, match='closed'):
            file.writelines([])


def test_open_rb():

    instream = io.BytesIO(b'\x82a\x02XYZ')
    with rleb_open(instream, mode='rb') as file:
        outchunk = file.read(1)
        assert file.closed is False
        assert instream.tell() == 6
        assert outchunk == b'a'
    assert file.closed is True
    assert instream.tell() == 6


def test_open_rt():

    instream = io.BytesIO(b'\x82a\x02XYZ')
    with rleb_open(instream, mode='rt') as file:
        outchunk = file.read(1)
        assert file.closed is False
        assert instream.tell() == 6
        assert outchunk == 'a'
    assert file.closed is True
    assert instream.tell() == 6


def test_open_wb():

    outstream = io.BytesIO()
    with rleb_open(outstream, mode='wb') as file:
        file.write(b'aaaaaXYZ')  # type: ignore
        assert file.closed is False
        assert outstream.tell() == 2
        assert outstream.getvalue() == b'\x82a'
    assert file.closed is True
    assert outstream.tell() == 6
    assert outstream.getvalue() == b'\x82a\x02XYZ'


def test_open_wt():

    outstream = io.BytesIO()
    with rleb_open(outstream, mode='wt') as file:
        file.write('aaaaaXYZ')  # type: ignore
        assert file.closed is False
        assert outstream.tell() == 0
        assert outstream.getvalue() == b''
    assert file.closed is True
    assert outstream.tell() == 6
    assert outstream.getvalue() == b'\x82a\x02XYZ'


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
def test_main_compress(filename, tmppath, datapath):

    inpath = datapath / filename
    outpath = tmppath / f'{filename}.RLEB'
    refpath = datapath / f'{filename}.RLEB'
    _argv = sys.argv[:]
    try:
        sys.argv[:] = ['rleb.py', str(inpath), str(outpath)]

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

    inpath = datapath / f'{filename}.RLEB'
    outpath = tmppath / filename
    refpath = datapath / filename
    _argv = sys.argv[:]
    try:
        sys.argv[:] = ['rleb.py', '-d', str(inpath), str(outpath)]

        main()

        with open(outpath, 'rb') as outstream:
            outdata = outstream.read()
        with open(refpath, 'rb') as refstream:
            refdata = refstream.read()
        assert outdata == refdata

    finally:
        sys.argv[:] = _argv
