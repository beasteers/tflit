import setuptools

USERNAME = 'beasteers'
NAME = 'tflit'
VERSION = '0.0.11'

setuptools.setup(
    name=NAME,
    version=VERSION,
    description='tflite_runtime, but easier.',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    author='Bea Steers',
    author_email='bea.steers@gmail.com',
    url='https://github.com/{}/{}'.format(USERNAME, NAME),
    packages=setuptools.find_packages(),
    package_data={NAME: ['*/*.tflite']},
    # entry_points={'console_scripts': ['{name}={name}:main'.format(name=NAME)]},
    install_requires=[
        'numpy',
        # 'tflite_runtime@{}'.format(get_tflite_url())
    ],
    extras_require={
        'tests': ['pytest-cov'],
    },
    license='MIT License',
    keywords='tflite runtime tensorflow keras deep machine learning model edge embedded compute cnn')
