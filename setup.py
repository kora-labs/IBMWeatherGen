from setuptools import setup, find_packages

setup(
    name='ibmweathergen',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'h5py >= 3.6.0',
        'ibmpairs >= 0.2.4',
        'matplotlib >= 3.5.2',
        'numpy >= 1.23.1',
        'pandas >= 1.4.3',
        'pmdarima >= 1.8.0',
        'pomegranate <= 0.14.8',
        'scipy >= 1.8.1',
        'statsmodels >= 0.13.2',
        'xarray >= 2022.9.0',
        'rasterio >= 1.3.3',
    ],
)