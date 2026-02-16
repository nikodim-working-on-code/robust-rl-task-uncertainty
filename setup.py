from setuptools import setup, find_packages

setup(
    name="robust-rl-task-uncertainty",
    version="1.0.0",
    author="Nikodim Aleksandrovich Svetlichnyi",
    description="Modular approach to robustness in RL under task-uncertainty",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium==0.29.1",
        "stable-baselines3==2.3.2",
        "numpy==1.26.0",
        "scipy==1.11.4",
        "matplotlib==3.8.2",
        "shapely==2.0.7",
        "tqdm==4.66.4",
    ],
)
