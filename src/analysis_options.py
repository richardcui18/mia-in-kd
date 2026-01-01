import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--teacher_model_name', type=str, default="Pythia", help="teacher model name")
        self.parser.add_argument('--teacher_model_testing_results_path', type=str, default=None, help="path to teacher model testing results")
        self.parser.add_argument(
            '--student_model_names', 
            type=str, 
            nargs='+',
            default="DistilPythia", 
            help="List of student model names"
        )
        self.parser.add_argument(
            '--student_model_testing_results_paths', 
            type=str, 
            nargs='+',
            default=None, 
            help="List of path to student model testing results"
        )