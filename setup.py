from setuptools import setup

setup(
    name='boxsers_7',
    url='https://github.com/ALebrun-108/BoxSERS',
    author='Alexis Lebrun',
    author_email='alexis.lebrun.1@ulaval.ca',
    # dependencies
    install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'scikit-learn', 'tensorflow'],
    python_requires='>=3.6',
    # *strongly* suggested for sharing
    version='1.0.0',
    # The license can be anything you like
    license='MIT',
    description='Provides a full range of features to process and analyze vibrational spectra.',
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
