from setuptools import setup, find_packages

setup(
    name='inchwormrf',
    version='0.0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='inchwormrf, a local inversion root finder',
    install_requires=['numpy', 'scipy'],
    test_suite='nose.collector',
    tests_require=['nose'],
    url='https://github.com/EFavDB/inchwormrf',
    author='Jonathan Landy, YongSeok Jho',
    author_email='jslandy@gmail.com'
)
