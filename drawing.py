from field import Field
import numpy as np
from matplotlib import pyplot as plt

def DrawVectorField(field: Field, stepsize = 1.0):
    minx = field.domain.min_x
    maxx = field.domain.max_x
    miny = field.domain.min_y
    maxy = field.domain.max_y
    
    xs = np.arange(minx, maxx, stepsize)
    ys = np.arange(miny, maxy, stepsize)

    X, Y = np.meshgrid(xs, ys)

    U = np.zeros(X.shape)
    V = np.zeros(Y.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            tangent_vec = field(np.array([X[i, j], Y[i, j]]))
            U[i, j] = tangent_vec.v[0]
            V[i, j] = tangent_vec.v[1]

    # find magnitude of vectors
    M = np.sqrt(U**2 + V**2)
    # if magnitude is 0, set to 1 to avoid division by 0
    M[M == 0] = 1
    U = U / M
    V = V / M

    plt.quiver(X, Y, U, V, M, cmap = 'viridis')

    # draw rectangle
    plt.plot(
                [minx, minx, maxx, maxx, minx], 
                [miny, maxy, maxy, miny, miny], 
                'r'
            )

    plt.show()

