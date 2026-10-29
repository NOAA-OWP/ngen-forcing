"""
BMI Forcings Engine standalone preprocessing mode wrapper script.

Provides ability to streamline ESMF Mesh production and operational data
workflows based on Forcing Engine configuration inputs.

example usage: python preprocessing_wrapper.py ./standard_ana_config.yml
"""

import argparse
import os
import time
from types import SimpleNamespace

import yaml

import esmf_creation
