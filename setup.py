from setuptools import setup, find_packages

setup(
    name='mylibrary',
    version='0.1',
    packages=find_packages(),
    description='A lightweight machine learning library',
    long_description=open('README.md').read(),
    author='Soufiane and Meryem',
    author_email='soufiane.elamrani@um6p.ma',
    url='https://github.com/Soufi0202/Mini_sklearn_library.git',
    install_requires=[
        'numpy>=1.18.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='machine learning, data preprocessing, machine learning library',
)
