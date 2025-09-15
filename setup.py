from setuptools import setup, find_packages


setup(
    name="adv-deep-hedging-irm",
    version="0.1.0",
    description="Advanced Deep Hedging with FIRM (Invariant Risk Minimization)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)

