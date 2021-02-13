import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("smokelite/__init__.py", "r") as fh:
    for l in fh:
        if l.startswith('__version__'):
            exec(l)
            break
    else:
        __version__ = 'x.y.z'


setuptools.setup(
    name='smokelist',
    version=__version__,
    author='Barron H. Henderson',
    author_email='barronh@gmail.com',
    maintainer='Barron Henderson',
    maintainer_email='barronh@gmail.com',
    url='https://github.com/barronh/smokelite/',
    download_url='https://github.com/barronh/smokelite/archive/main.zip',
    long_description=(
        "smokelite has a subset of SMOKE functionality, and is intended to "
        + "to make easy emissions processing easier. It is not intended as a "
        + "replacement."
    ),
    packages=setuptools.find_packages(),
    package_dir={'smokelite': 'smokelite'}
    package_data={'pykpp.models': ['*.eqn', '*.txt', '*.kpp', '*.def']},
    },
    install_requires=[
        'numpy', 'scipy', 'pandas', 'pyproj', 'pseudonetcdf'
    ]
)
