from setuptools import setup, find_packages

setup(
    name="supermariopy",
    version="0.2",
    url="https://github.com/theRealSuperMario/supermariopy",
    author="Sandro Braun",
    author_email="supermario94123@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=["matplotlib", "opencv-python", "numpy", "scikit-image", "scipy"],
    zip_safe=False,
)
