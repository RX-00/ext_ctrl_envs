""" Setups the project """

import pathlib

from setuptools import setup

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the ext ctrl envs version."""
    path = CWD / "ext_ctrl_envs" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return long_description

setup(
    name="ext_ctrl_envs",
    version=get_version(),
    install_requires=["gymnasium==0.28.1", "mujoco==2.3.5"],
    long_description=get_description(),
    long_description_content_type="text/markdown",
)
