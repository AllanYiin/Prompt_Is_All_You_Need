import os.path
import pathlib
import os
import pkg_resources
from setuptools import setup, find_packages

# with open("README.md", "r",encoding='utf-8-sig') as fh:
#     long_description = fh.read()



NAME = "prompt4all"
DIR = '.'

PACKAGES = find_packages(exclude= ["whisper",".idea","audio","generate_interviews"])
print(PACKAGES)




with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

if not os.path.exists('generate_images'):
    os.mkdir('generate_images')
if not os.path.exists('generate_text'):
    os.mkdir('generate_text')


setup(name=NAME,
      version='0.0.6',
      description='Prompt is all you need',
      # long_description=long_description,
      # long_description_content_type="text/markdown",
      long_description=open("README.md", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      author='Allan Yiin',
      author_email='allanyiin.ai@gmail.com',
      download_url='https://test.pypi.org/project/prompt4all',
      license='MIT',
      install_requires=install_requires,
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov',
                    'requests',
                    'markdown'],
      },
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      python_requires='>=3.7',
      keywords=['chatgpt', 'gpt4'],
      packages= find_packages(exclude= ["whisper",".idea","audio","generate_interviews"]),
      include_package_data=False,
      scripts=[],

      )

