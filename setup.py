from setuptools import setup, find_packages

setup(
      # mandatory
      name='simplify',
      # mandatory
      version='0.1',
      # mandatory
      author_email='eric.schanet@cern.ch',
      packages=['simplify'],
      package_data={},
      install_requires=['pyhf', 'click'],
      entry_points={
        'console_scripts': ['simplify = simplify.cli:cli']
      }
)
