from setuptools import setup

setup(
    name='boxsers',
    url='https://github.com/ALebrun-108/BoxSERS',
    author='Alexis Lebrun',
    author_email='alexis.lebrun.1@ulaval.ca',
    # dependencies
    install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'scikit-learn', 'tensorflow',
                      'tables'],
    python_requires='>=3.6',
    # *strongly* suggested for sharing
    version='1.3.2',
    # The license can be anything you like
    license='MIT',
    description='Python package that provides a full range of functionality to process and analyze vibrational'
                ' spectra (Raman, SERS, FTIR, etc.).',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README_pypi.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["boxsers", "boxsers.machine_learning"],
)
