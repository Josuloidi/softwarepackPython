from setuptools import setup

setup(
   name='softwarepack',
   version='0.0.1',
   author='Josu Loidi',
   author_email='josu.loidi@ikasle.ehu.eus',
   packages=['softwarepack', 'softwarepack.test'],
   url='Indicar una URL para el paquete...',
   license='LICENSE.txt',
   description='Este paquete incluye funciones que debía implementar para la asignatura Software Estadístico y Matemático',
   long_description=open('README.txt').read(),
   tests_require=['pytest'],
   install_requires=[
      "seaborn >= 0.9.0",
      "pandas >= 0.25.1",
      "matplotlib >= 3.1.1",
      "numpy >=1.17.2"
   ],
)