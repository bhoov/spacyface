from setuptools import setup, find_packages

requires = [] # Let conda handle requires

setup(
    name="spacyface",
    description="Aligner for spacy and huggingface tokenization",
    packages=find_packages(),
    author="Ben Hoover",
    include_package_data=True,
    install_requires=requires
)
