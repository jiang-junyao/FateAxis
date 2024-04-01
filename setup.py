import setuptools

setuptools.setup(
    name = "FateAxis",
    version = "v0.0.1",
    packages = setuptools.find_packages(),
    # Metadata
    url = "https://github.com/jiang-junyao/FateAxis",
    author = "Junyao Jiang",
    author_email = "jyjiang@link.cuhk.edu.hk",
    description = "Identification of Key Regulatory Relationships Governing \
            Cell State Transitions from scMultiomics Data using AutoML",
    python_requires = ">=3.6",
    include_package_data = True,
    package_data = {'': [
        'config/*',
        'tfdb/*'
    ]}
    #...
)