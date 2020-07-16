import os
import sys
import tempfile
from io import BytesIO

import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
                           suppress_warnings)
from pytest import raises, warns

from scipy.io import wavfile


def datafile(fn):
    return os.path.join(os.path.dirname(__file__), 'data', fn)


def test_read_1():
    # 32-bit PCM (which uses extensible format)
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)

        assert_equal(rate, 44100)
        assert_(np.issubdtype(data.dtype, np.int32))
        assert_equal(data.shape, (4410,))

        del data


def test_read_2():
    # 8-bit unsigned PCM
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-2ch-1byteu.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)

        assert_equal(rate, 8000)
        assert_(np.issubdtype(data.dtype, np.uint8))
        assert_equal(data.shape, (800, 2))

        del data


def test_read_3():
    # Little-endian float
    for mmap in [False, True]:
        filename = 'test-44100Hz-2ch-32bit-float-le.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)

        assert_equal(rate, 44100)
        assert_(np.issubdtype(data.dtype, np.float32))
        assert_equal(data.shape, (441, 2))

        del data


def test_read_4():
    # Contains unsupported 'PEAK' chunk
    for mmap in [False, True]:
        with suppress_warnings() as sup:
            sup.filter(wavfile.WavFileWarning,
                       "Chunk .non-data. not understood, skipping it")
            filename = 'test-48000Hz-2ch-64bit-float-le-wavex.wav'
            rate, data = wavfile.read(datafile(filename), mmap=mmap)

        assert_equal(rate, 48000)
        assert_(np.issubdtype(data.dtype, np.float64))
        assert_equal(data.shape, (480, 2))

        del data


def test_read_5():
    # Big-endian float
    for mmap in [False, True]:
        filename = 'test-44100Hz-2ch-32bit-float-be.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)

        assert_equal(rate, 44100)
        assert_(np.issubdtype(data.dtype, np.float32))
        assert_(data.dtype.byteorder == '>' or (sys.byteorder == 'big' and
                                                data.dtype.byteorder == '='))
        assert_equal(data.shape, (441, 2))

        del data


def test_read_unknown_filetype_fail():
    # Not an RIFF
    for mmap in [False, True]:
        filename = 'example_1.nc'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match="CDF.*'RIFF' and 'RIFX' supported"):
                wavfile.read(fp, mmap=mmap)


def test_read_unknown_riff_form_type():
    # RIFF, but not WAVE form
    for mmap in [False, True]:
        filename = 'Transparent Busy.ani'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='Not a WAV file.*ACON'):
                wavfile.read(fp, mmap=mmap)


def test_read_unknown_wave_format():
    # RIFF and WAVE, but not supported format
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-1ch-1byte-ulaw.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='Unknown wave file format.*MULAW.*'
                        'Supported formats'):
                wavfile.read(fp, mmap=mmap)


def test_read_early_eof_with_data():
    # File ends inside 'data' chunk, but we keep incomplete data
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes-early-eof.wav'
        with open(datafile(filename), 'rb') as fp:
            with warns(wavfile.WavFileWarning, match='Reached EOF'):
                rate, data = wavfile.read(fp, mmap=mmap)

                assert_(data.size > 0)
                assert_equal(rate, 44100)

        del data


def test_read_early_eof():
    # File ends after 'fact' chunk at boundary, no data read
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes-early-eof-no-data.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match="Unexpected end of file."):
                wavfile.read(fp, mmap=mmap)


def test_read_incomplete_chunk():
    # File ends inside 'fmt ' chunk ID, no data read
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes-incomplete-chunk.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match="Incomplete chunk ID.*b'f'"):
                wavfile.read(fp, mmap=mmap)


def _check_roundtrip(realfile, rate, dtype, channels):
    if realfile:
        fd, tmpfile = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
    else:
        tmpfile = BytesIO()
    try:
        data = np.random.rand(100, channels)
        if channels == 1:
            data = data[:, 0]
        if dtype.kind == 'f':
            # The range of the float type should be in [-1, 1]
            data = data.astype(dtype)
        else:
            data = (data*128).astype(dtype)

        wavfile.write(tmpfile, rate, data)

        for mmap in [False, True]:
            rate2, data2 = wavfile.read(tmpfile, mmap=mmap)

            assert_equal(rate, rate2)
            assert_(data2.dtype.byteorder in ('<', '=', '|'), msg=data2.dtype)
            assert_array_equal(data, data2)

            del data2
    finally:
        if realfile:
            os.unlink(tmpfile)


def test_write_roundtrip():
    for realfile in (False, True):
        for dtypechar in ('i', 'u', 'f', 'g', 'q'):
            for size in (1, 2, 4, 8):
                if size == 1 and dtypechar == 'i':
                    # signed 8-bit integer PCM is not allowed
                    continue
                if size > 1 and dtypechar == 'u':
                    # unsigned > 8-bit integer PCM is not allowed
                    continue
                if (size == 1 or size == 2) and dtypechar == 'f':
                    # 8- or 16-bit float PCM is not expected
                    continue
                if dtypechar in 'gq':
                    # no size allowed for these types
                    if size == 1:
                        size = ''
                    else:
                        continue

                for endianness in ('>', '<'):
                    if size == 1 and endianness == '<':
                        continue
                    for rate in (8000, 32000):
                        for channels in (1, 2, 5):
                            dt = np.dtype('%s%s%s' % (endianness, dtypechar,
                                                      size))
                            _check_roundtrip(realfile, rate, dt, channels)
