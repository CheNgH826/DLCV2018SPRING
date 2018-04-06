import numpy as np

color_bank = [
    [128, 128, 0],
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 200],
    [145, 30, 180],
    [128, 0, 0],
    [0, 128, 128],
    [240, 50, 230]
    ]

def f(t):
    if t > 0.008856:
        return t**(1./3)
    else:
        return 7.787*t + 16./116

def RGB2Lab(rgb):
    trans_rbg2xyz = np.array([[0.412453, 0.357580, 0.180423],
                              [0.212671, 0.715160, 0.072169],
                              [0.019334, 0.119193, 0.950227]])
    Xn = 0.9515
    Yn = 1
    Zn = 1.0886
    xyz = np.dot(trans_rbg2xyz, rgb)
    X, Y, Z = xyz[0], xyz[1], xyz[2]
    L = 116*f(Y/Yn)-16
    a = 500*(f(X/Xn)-f(Y/Yn))
    b = 200*(f(Y/Yn)-f(Z/Zn))
    return np.array([L, a, b])


