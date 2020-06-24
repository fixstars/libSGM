from setuptools import setup, find_packages

setup(
    name='pysgm',
    version='0.1.0',
    description='Python pybind11 wrappers for the libSGM',
    long_description="",
    author='Oleksandr Slovak',
    author_email='slovak194@gmail.com',
    url="https://github.com/fixstars/libSGM",
    license="Apache License Version 2.0, January 2004 http://www.apache.org/licenses/",
    packages=find_packages(exclude=('tests', 'docs')),
    package_data={},
    scripts=[]
)
