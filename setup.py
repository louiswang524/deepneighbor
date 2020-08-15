import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
REQUIRED_PACKAGES = [
    'h5py',
    'requests',
    'gensim==3.7.0',
    'joblib==0.13.0',
    'fastdtw==0.3.2',
    'tqdm',
    'numpy',
    'scikit-learn',
    'pandas',
    'matplotlib',
    'annoy'
]

setuptools.setup(

  name="deepneighbor",
  version="0.1.5",
  author="Yufeng Wang",
  author_email="louiswang524@gmail.com",
  description="embedding-based item nearest neighborhoods extraction",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/LouisBIGDATA/deepneighbor",
  packages=setuptools.find_packages(),
  python_requires=">=3.4",  # '>=3.4',  # 3.4.6
    install_requires=REQUIRED_PACKAGES,
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  'Intended Audience :: Developers',
 'Intended Audience :: Education',
 'Intended Audience :: Science/Research',
  'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
  ],
)
