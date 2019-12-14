from __future__ import division, print_function

import os
import platform
import sys
from os.path import join

from numpy.distutils.system_info import platform_bits

is_msvc = (platform.platform().startswith('Windows') and
           platform.python_compiler().startswith('MS'))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_mathlibs
    config = Configuration('random', parent_package, top_path)

    def generate_libraries(ext, build_dir):
        config_cmd = config.get_config_cmd()
        libs = get_mathlibs()
        if sys.platform == 'win32':
            libs.extend(['Advapi32', 'Kernel32'])
        ext.libraries.extend(libs)
        return None

    # enable unix large file support on 32 bit systems
    # (64 bit off_t, lseek -> lseek64 etc.)
    if sys.platform[:3] == "aix":
        defs = [('_LARGE_FILES', None)]
    else:
        defs = [('_FILE_OFFSET_BITS', '64'),
                ('_LARGEFILE_SOURCE', '1'),
                ('_LARGEFILE64_SOURCE', '1')]

    defs.append(('NPY_NO_DEPRECATED_API', 0))
    config.add_data_dir('tests')
    config.add_data_files('common.pxd')
    config.add_data_files('bit_generator.pxd')

    EXTRA_LINK_ARGS = []
    # Math lib
    EXTRA_LIBRARIES = ['m'] if os.name != 'nt' else []
    # Some bit generators exclude GCC inlining
    EXTRA_COMPILE_ARGS = ['-U__GNUC_GNU_INLINE__']

    if is_msvc and platform_bits == 32:
        # 32-bit windows requires explicit sse2 option
        EXTRA_COMPILE_ARGS += ['/arch:SSE2']
    elif not is_msvc:
        # Some bit generators require c99
        EXTRA_COMPILE_ARGS += ['-std=c99']
        INTEL_LIKE = any(arch in platform.machine() 
                         for arch in ('x86', 'i686', 'i386', 'amd64'))
        if INTEL_LIKE:
            # Assumes GCC or GCC-like compiler
            EXTRA_COMPILE_ARGS += ['-msse2']

    # Use legacy integer variable sizes
    LEGACY_DEFS = [('NP_RANDOM_LEGACY', '1')]
    PCG64_DEFS = []
    # One can force emulated 128-bit arithmetic if one wants.
    #PCG64_DEFS += [('PCG_FORCE_EMULATED_128BIT_MATH', '1')]

    for gen in ['mt19937']:
        # gen.pyx, src/gen/gen.c, src/gen/gen-jump.c
        config.add_extension(gen,
                             sources=['{0}.c'.format(gen),
                                      'src/{0}/{0}.c'.format(gen),
                                      'src/{0}/{0}-jump.c'.format(gen)],
                             include_dirs=['.', 'src', join('src', gen)],
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             extra_link_args=EXTRA_LINK_ARGS,
                             depends=['%s.pyx' % gen],
                             define_macros=defs,
                             )
    for gen in ['philox', 'pcg64', 'sfc64']:
        # gen.pyx, src/gen/gen.c
        _defs = defs + PCG64_DEFS if gen == 'pcg64' else defs
        config.add_extension(gen,
                             sources=['{0}.c'.format(gen),
                                      'src/{0}/{0}.c'.format(gen)],
                             include_dirs=['.', 'src', join('src', gen)],
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             extra_link_args=EXTRA_LINK_ARGS,
                             depends=['%s.pyx' % gen, 'bit_generator.pyx',
                                      'bit_generator.pxd'],
                             define_macros=_defs,
                             )
    for gen in ['common', 'bit_generator']:
        # gen.pyx
        config.add_extension(gen,
                             sources=['{0}.c'.format(gen)],
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             extra_link_args=EXTRA_LINK_ARGS,
                             include_dirs=['.', 'src'],
                             depends=['%s.pyx' % gen, '%s.pxd' % gen,],
                             define_macros=defs,
                             )
    other_srcs = [
        'src/distributions/logfactorial.c',
        'src/distributions/distributions.c',
        'src/distributions/random_hypergeometric.c',
    ]
    for gen in ['generator', 'bounded_integers']:
        # gen.pyx, src/distributions/distributions.c
        config.add_extension(gen,
                             sources=['{0}.c'.format(gen)] + other_srcs,
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             include_dirs=['.', 'src'],
                             extra_link_args=EXTRA_LINK_ARGS,
                             depends=['%s.pyx' % gen],
                             define_macros=defs,
                             )
    config.add_extension('mtrand',
                         # mtrand does not depend on random_hypergeometric.c.
                         sources=['mtrand.c',
                                  'src/legacy/legacy-distributions.c',
                                  'src/distributions/logfactorial.c',
                                  'src/distributions/distributions.c'],
                         include_dirs=['.', 'src', 'src/legacy'],
                         libraries=EXTRA_LIBRARIES,
                         extra_compile_args=EXTRA_COMPILE_ARGS,
                         extra_link_args=EXTRA_LINK_ARGS,
                         depends=['mtrand.pyx'],
                         define_macros=defs + LEGACY_DEFS,
                         )
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(configuration=configuration)
