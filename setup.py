from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
	name='cmaclp',
	version='0.0.1',
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'SVM_prediction=cmaclp.SVM_prediction:main'
        ]
    },
    install_requires=['python-build','h5py','numpy','pandas','scikit-learn','scanpy','rpy2','importlib-resources','pytest-cov'],
    package_data={"cmaclp.data": ["*.csv", "*.h5ad"]},
    long_description=long_description,
    long_description_content_type='text/markdown'
)