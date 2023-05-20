from distutils.core import setup
from distutils import util

pathdft = util.convert_path('AlgScComp/dft')

setup(
    name='AlgScComp',
    version='1.0',
    description='Python Algorithms for Scientific Computing. Basic Library for ',
    author='Manuel R. Wendl',
    author_email='manuel.wendl@tum.de',
    url='https://github.com/ManuelWendl/AlgScComp',
    package_dir = {
            'AlgScComp': 'AlgScComp',
            'myPackage.mySubPackage1': pathdft},
    packages=['AlgScComp', 'AlgScComp.dft']
)