try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
  name='cmaclp',
  version='0.1.1',
  # To provide executable scripts, use entry points in preference to the
  # "scripts" keyword. Entry points provide cross-platform support and allow
  # pip to create the appropriate form of executable for the target platform.
  entry_points={
    'console_scripts': ['SVM_prediction=cmaclp.SVM_prediction:predpars',
                        'SVM_import=cmaclp.SVM_prediction:importpars',
                        'SVM_performance=cmaclp.SVM_prediction:performpars'
    ]
  },
  install_requires=['setuptools<=57.5.0', 'wheel', 'python-build',
                    'h5py', 'numpy>=1.23.3, <1.24', 'pandas>=1.4.4',
                    'scikit-learn', 'scanpy>=1.9.1',
                    'importlib-resources', 'pytest-cov'],
  packages=find_packages(),
  long_description=long_description,
  long_description_content_type='text/markdown'
)
