"""Setup script."""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
    import verysimplestats
    version = verysimplestats.__version__
except ImportError:
    version = 'Undefined'


classifiers = ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               'Programming Language :: Python',
               'Topic :: Utilities']

packages = ['verysimplestats']
requires = []

setup(name='verysimplestats',
      version=version,
      author='PierreSelim',
      author_email='ps.huard@gmail.com',
      url='http://github.com/PierreSelim/verysimplestats',
      description='Statistics for humans',
      long_description=open('README.md').read(),
      license='MIT',
      packages=packages,
      install_requires=requires,
      classifiers=classifiers)