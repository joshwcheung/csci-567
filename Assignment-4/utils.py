import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class Figure():
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('index 0')
        self.ax.set_ylabel('index 1')
        self.ax.axis('equal')

    def savefig(self, file_name):
        file = file_name.split('/')[-1]
        directory = '/'.join(file_name.split('/')[:-1])
        if (not os.path.isdir(directory)):
            os.makedirs(directory)
        plt.savefig(file_name)
