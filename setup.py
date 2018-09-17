from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gridworlds'))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='gridworlds',
      version=0.1,
      description='Contains various gym compatible 2D environments for reinforcement learning',
      url='https://github.com/bjkomer/gridworlds',
      author='bjkomer',
      packages=[package for package in find_packages()
                if package.startswith('gym')],
      keywords = ['AI', 'Reinforcement Learning', 'Gridworld'],
      zip_safe=False,
      classifiers = [
              'Programming Language :: Python',
              'Intended Audience :: Developers',
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent',
              'Topic :: Scientific/Engineering :: Artificial Intelligence',
              ],
      long_description = read('README.md')
)
