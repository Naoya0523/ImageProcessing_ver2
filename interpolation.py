import numpy as np

def nearestNeighbour(x:float, y:float, before_transform_image:np.ndarray) -> np.ndarray:
    # before_transform_image.shape = (1500,1500,3)
    return before_transform_image[int(np.floor(y+0.5))][int(np.floor(x+0.5))]


class BicubicInterpolation:
    def __init__(self) -> None:
        self.vec_Hx = None
        self.vec_Hy = None
        self.kernel = None

    def h(self, t:float) -> float:
        if abs(t) <= 1:
            return (abs(t)**3 -2*abs(t)**2 +1)
        elif (1 < abs(t) <= 2):
            return (-abs(t)**3 +5*abs(t)**2 -8*abs(t) +4)
        else:
            return 0
    
    def shift(self, z:int):
        return z-np.floor(z)+1

    def getVec_H(self, t:int) -> np.ndarray:
        return np.array([[self.h(self.shift(t))],
                          [self.h(self.shift(t)-1)],
                          [self.h(2-self.shift(t))],
                          [self.h(3-self.shift(t))]])

    def bicubicInterpolation(self, x:float, y:float, before_transform_image:np.ndarray) -> np.ndarray:
        self.vec_Hx = self.getVec_H(x)
        self.vec_Hy = self.getVec_H(y)
        self.kernel = np.zeros((4,4))
        
        for i in range(4):
            for j in range(4):
                self.kernel[j][i] = before_transform_image[int(i+np.floor(y)-1)][int(j+np.floor(x)-1)]

        pixcel = np.dot(np.dot(self.vec_Hx.T,self.kernel), self.vec_Hy)
        pixcel = pixcel

        return pixcel