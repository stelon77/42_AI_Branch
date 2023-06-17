from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
   name='my_minipack',
   version='1.0.0',
   author='lcoiffie',
   author_email='lcoiffie@student.42.fr',
   package_dir={"": "src"},
   packages=find_packages(where="src"),
   url='None',
   license='GPLv3',
   description='An awesome package with progressionbar and logger',
   long_description=long_description,
   long_description_content_type="text/markdown",
   install_requires=[
       "setuptools>=42",
       "pip>=20",
       "wheel"
   ],
   classifiers=[
       "Development Status :: 3 - Alpha",
       "Intended Audience :: Developers",
       "Intended Audience :: Students",
       "Topic :: Education",
       "Topic :: HowTo",
       "Topic :: Package",
       "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
       "Programming Language :: Python :: 3",
       "Programming Language :: Python :: 3 :: Only",
   ],
   python_requires='>=3.7',
)
