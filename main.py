from field import Field, IntegerSquareDomain, IntegerFieldToReal
from drawing import DrawVectorField
from controller import GreedyController
from simulators import SimulateControllerOnField
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

FloatVec = npt.NDArray[np.float64]
IntVec = npt.NDArray[np.int64]

def F(point: IntVec) -> FloatVec:
    mag = np.linalg.norm(point)
    return mag * np.array([-point[1], point[0]])

if __name__ == '__main__':
    minx = -10
    maxx = 10
    miny = -10
    maxy = 10
    domain = IntegerSquareDomain(minx, maxx, miny, maxy)

    intfield = Field[IntVec, FloatVec](domain, F)
    floatfield = IntegerFieldToReal(intfield)
    domain = floatfield.domain

    DrawVectorField(floatfield)

    # plot the trajectory of the point 
    plt.ion()

    target = np.array([1, 1])
    cont = GreedyController(target, 4)
    xs = []
    ys = []
    for tan, u in SimulateControllerOnField(floatfield, np.array([-5, 5]), cont, steps = 1000, stepsize=0.01):
        xs.append(tan.basepoint[0])
        ys.append(tan.basepoint[1])
        plt.plot(xs, ys)

        plt.xlim(-15, 15)
        plt.ylim(-15, 15)

        # draw rectangle
        plt.plot(
            [domain.min_x, domain.min_x, domain.max_x, domain.max_x, domain.min_x], 
            [domain.min_y, domain.max_y, domain.max_y, domain.min_y, domain.min_y], 
            'r')

        # draw target
        plt.plot(target[0], target[1], 'ro')

        # draw the direction of the controller
        plt.quiver(tan.basepoint[0], tan.basepoint[1], u[0], u[1], color='r')

        plt.draw()
        plt.pause(0.0001)
        plt.clf()



    