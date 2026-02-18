import numpy as np


class sim(): 
    
    def __init__(self, traj_file):

        self.traj_file = traj_file
        self.traj = self.parse_traj()

    def parse_traj(self):
        traj = []
        with open(self.traj_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                traj.append(list(map(float, line.split())))

        return np.array(traj)