from setuptools import setup, find_packages

requires = [
    'transformers>=2.3.0',
    'h5py>=2.10.0',
    'numpy>=1.17.4',
    'regex>=2020.1.8',
    'spacy>=2.2.3',
    'torch',
]

setup(
    name="spacyface",
    description="Aligner for spacy and huggingface tokenization",
    packages=find_packages(exclude=['tests']),
    version='0.3.0',
    license='Apache 2.0',
    author="Ben Hoover",
    author_email="benjamin.hoover@ibm.com",
    url="https://github.com/bhoov/spacyface",
    keywords=["transformer", "pytorch", "spacy", "tokenize", "tokenization", "NLP", "Natural Language Processing",
              "huggingface", "linguistic"],
    include_package_data=True,
    install_requires=requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
    python_requires='>=3.6, <3.9'
)
