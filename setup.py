'''

Version Info: https://www.tensorflow.org/lite/guide/python

To get URLs: `$('.devsite-table-wrapper').find('a').map((i, e) => e.href).get()`

And to add jQuery
```javascript
var jq = document.createElement('script');
jq.onload = function() { jQuery.noConflict(); $ = jQuery; }
jq.src = "https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(jq);
```

[
    # Linux ARM 32
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_armv7l.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_armv7l.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_armv7l.whl",
    # Linux ARM 64
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_aarch64.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_aarch64.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_aarch64.whl",
    # Linux x86-64
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_x86_64.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_x86_64.whl",
    # Mac 10.14
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-macosx_10_14_x86_64.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-macosx_10_14_x86_64.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl",
    # Windows 10
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-win_amd64.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-win_amd64.whl",
    "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-win_amd64.whl"
]

'''
import os
import sys
import platform
import setuptools

USERNAME = 'beasteers'
NAME = 'tflit'


URL = (
    'https://dl.google.com/coral/python/tflite_runtime-'
    '{version}-cp{py}-cp{pym}-{platform}_{arch}.whl')

PLATFORMS = {
    'Linux': 'linux',
    'Darwin': 'macosx',
    'Windows': 'win',
}

VERSIONS = {
    'Linux': ['35', '36', '37', '38'],
    'Darwin': ['35', '36', '37'],
    'Windows': ['35', '36', '37'],
}
NO_M = ['38']

def get_tflite_url(version='2.1.0.post1'):
    system = platform.system()
    py_version = '{}{}'.format(*sys.version_info)

    if system == 'Linux':
        is_64 = sys.maxsize > 2**32
        is_arm = platform.uname().machine.startswith("arm")
        arch = ('aarch64' if is_64 else 'armv7l') if is_arm else 'x86_64'
    elif system == 'Darwin':
        if py_version == '38':
            py_version = '37'  # ??
        arch = '10_14_x86_64'
    elif system == 'Windows':
        arch = 'amd64'
    else:
        raise ValueError('Unknown system: {}'.format(system))

    return URL.format(
        version=version, py=py_version,
        pym=py_version if py_version in NO_M else py_version + 'm',
        platform=PLATFORMS.get(system),
        arch=arch,
    )


setuptools.setup(
    name=NAME,
    version='0.0.1',
    description='tflite_runtime, but easier.',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    author='Bea Steers',
    author_email='bea.steers@gmail.com',
    url='https://github.com/{}/{}'.format(USERNAME, NAME),
    packages=setuptools.find_packages(),
    # entry_points={'console_scripts': ['{name}={name}:main'.format(name=NAME)]},
    install_requires=[
        'numpy',
        'tflite_runtime@{}'.format(get_tflite_url())
    ],
    extras_require={
        'tests': ['pytest', 'pytest-cov', 'codecov'],
    },
    license='MIT License',
    keywords='tflite runtime tensorflow keras deep machine learning model edge embedded compute cnn')
