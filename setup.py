from distutils.core import setup

setup(
    name='gemballs',
    packages=['gemballs'],
    version='0.3',
    license='https://github.com/Kaysman/gemballs/blob/master/LICENSE.md',
    description='A powerful machine learning algorithm for binary classification',
    author='Omar Younis',
    author_email='omar.younis98@gmail.com',
    url='https://github.com/Kaysman/gemballs',
    download_url='https://github.com/Kaysman/gemballs/archive/0.3.tar.gz',
    keywords=['MACHINE LEARNING', 'GEM-balls', 'CLASSIFICATION'],
    install_requires=[
        'numpy',
        'scipy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
    ],
)
