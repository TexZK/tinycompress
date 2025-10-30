********
Overview
********

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |gh_actions|
        | |codecov|
    * - package
      - | |version| |wheel|
        | |supported-versions|
        | |supported-implementations|

.. |docs| image:: https://app.readthedocs.org/projects/tinycompress/badge/?style=flat
    :target: https://app.readthedocs.org/projects/tinycompress
    :alt: Documentation Status

.. |gh_actions| image:: https://github.com/TexZK/tinycompress/workflows/CI/badge.svg
    :alt: GitHub Actions Status
    :target: https://github.com/TexZK/tinycompress

.. |codecov| image:: https://codecov.io/gh/TexZK/tinycompress/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/TexZK/tinycompress

.. |version| image:: https://img.shields.io/pypi/v/tinycompress.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/tinycompress/

.. |wheel| image:: https://img.shields.io/pypi/wheel/tinycompress.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/tinycompress/

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/tinycompress.svg
    :alt: Supported versions
    :target: https://pypi.org/project/tinycompress/

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/tinycompress.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/tinycompress/

.. end-badges

A tiny library for tiny binary compression.

* Free software: BSD 2-Clause License


Overview
==========

TinyCompress is a lightweight Python library that provides efficient data
compression algorithms with a focus on simplicity and ease of use.

The library implements multiple compression methods:

* **LZSSW** (Lempel-Ziv-Storer-Szymanski via Words):
  An efficient variant of LZ77 compression algorithm that uses a sliding
  window approach for finding repeated data patterns.

* **RLEB** (Run-Length Encoding for Bytes):
  A specialized compression algorithm that combines run-length encoding with
  bit flags, particularly effective for data with repeated byte sequences.

Key features:

* Pure Python implementation for maximum portability
* Streaming interface for processing large files with minimal memory usage
* File handling utilities for direct compression/decompression of files
* Extensible base classes for implementing custom compression algorithms
* Comprehensive test suite with high code coverage
* BSD 2-Clause licensed for both commercial and open-source use


Documentation
=============

For the full documentation, please refer to:

https://tinycompress.readthedocs.io/en/latest/


Installation
============

From PyPI (might not be the latest version found on *github*):

.. code-block:: sh

    $ pip install tinycompress

From the source code root directory:

.. code-block:: sh

    $ pip install .


Development
===========

To run the all the tests:

.. code-block:: sh

    $ pip install tox
    $ tox
