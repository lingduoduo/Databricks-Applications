import logging
import os
import sys

from environment import Environment

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = str(os.path.sep).join(parent_dir.split(os.path.sep)[:-1])
sys.path.append(src_path)


def test_environment():
    """
    Test envronment before starting other validations
    """

    environment = Environment()

    assert environment.process_env()
