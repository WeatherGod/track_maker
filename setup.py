import os
from setuptools import setup

setup(
    name = "track_maker",
    version = "0.0.1",
    author = "Benjamin Root",
    author_email = "ben.v.root@gmail.com",
    description = "Tool to assist storm track analysis",
    license = "BSD",
    keywords = "track analysis",
    url = "https://github.com/WeatherGod/track_maker",
    scripts = ['scripts/track_maker.py', 'scripts/sync_scenario.py'],
    install_requires = ['numpy', 'BRadar', 'matplotlib', 'scipy', 'ZigZag',],
    )

