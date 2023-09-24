from distutils.core import setup
from distutils import util

pathdft = util.convert_path('AlgScComp/dft')
pathhnm = util.convert_path('AlgScComp/hnm')
pathsfc = util.convert_path('AlgScComp/sfc')

setup(
    name='AlgScComp',
    version='1.0',
    description='Python Algorithms for Scientific Computing.',
    author='Manuel R. Wendl',
    author_email='manuel.wendl@tum.de',
    url='https://github.com/ManuelWendl/AlgScComp',
    package_dir = {
            'AlgScComp': 'AlgScComp',
            'AlgScComp.dft': pathdft,
            'AlgScComp.hnm': pathhnm,
            'AlgScComp.sfc': pathsfc},
    packages=['AlgScComp','AlgScComp.dft','AlgScComp.hnm','AlgScComp.sfc']
)