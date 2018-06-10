import numpy as np

class LinReg:

    def __init__(self):
        pass
    
    @staticmethod
    def getCoef(_x_, y):
        return np.matmul(np.linalg.inv(np.matmul(_x_.transpose(), _x_)), np.matmul(_x_.transpose(), y))
