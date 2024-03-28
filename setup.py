from setuptools import setup, find_packages

setup(
    name = "packaging_tutorial",
    version = "0.1",
    packages = find_packages(),
    # Metadata
    url = "https://github.com/jiang-junyao/FateAxis",
    author = "Junyao Jiang",
    author_email = "jyjiang@link.cuhk.edu.hk",
    description = "Identification of Key Regulatory Relationships Governing \
            Cell State Transitions from scMultiomics Data using AutoML",
    python_requires = ">=3.6",
    include_package_data = True
    #...
)