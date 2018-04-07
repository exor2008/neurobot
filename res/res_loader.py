import os
from panda3d.core import Filename

class Loader:
    def __init__(self):
        self.base_path, _ = os.path.split(__file__)

    def get_model_path(self, name):
        winpath = os.path.join(self.base_path, name)
        return Filename.fromOsSpecific(winpath).getFullpath()