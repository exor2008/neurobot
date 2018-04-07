import os
import sys
base_path, file = os.path.split(__file__)
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, 'model'))
sys.path.append(os.path.join(base_path, 'agent'))
sys.path.append(os.path.join(base_path, 'models'))
sys.path.append(os.path.join(base_path, 'res'))
import model
