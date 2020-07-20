from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="supermariopy",
    version="0.2",
    url="https://github.com/theRealSuperMario/supermariopy",
    author="Sandro Braun",
    author_email="supermario94123@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=required,
    zip_safe=False,
)
