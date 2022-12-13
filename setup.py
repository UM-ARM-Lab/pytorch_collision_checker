from setuptools import setup

setup(
    name='pytorch_collision_checker',
    version='0.0.0',
    packages=['pytorch_collision_checker'],
    url='https://github.com/UM-ARM-Lab/pytorch_collision_checker',
    license='MIT',
    author='peter mitrano',
    author_email='pmitrano@umich.edu',
    description='collision checker implemented in pytorch',
    install_requires=[
        'torch',
        'numpy',
        'pytorch_kinematics',
        'arm_pytorch_utilities',
    ],
    tests_require=[
        'pytest'
    ]
)
