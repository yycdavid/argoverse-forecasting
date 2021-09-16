import os
import sys
import numpy as np

class OutputManager(object):
    def __init__(self, result_path, filename='log.txt'):
        self.result_folder = result_path
        self.log_file = open(os.path.join(result_path, filename), 'w')

    def say(self, s):
        self.log_file.write("{}\n".format(s))
        self.log_file.flush()
        sys.stdout.write("{}\n".format(s))
        sys.stdout.flush()


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        print("Creating {}".format(dir_path))
        os.makedirs(dir_path)
    else:
        #raise Exception('Result folder for this experiment already exists')
        pass
