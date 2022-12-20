from typing import Generator, Union
from field import Field, TangentVector
from controller import Controller
import numpy as np
import numpy.typing as npt

FloatVec = npt.NDArray[np.float64]


def SimulateControllerOnField(field: Field[FloatVec, FloatVec], initial_point: FloatVec, controller: Union[Controller, None], stepsize = 0.1, steps = 10000) -> Generator:
    if controller is None:
        controller = Controller(np.zeros(field.domain.dim), 0)

    point = initial_point

    for i in range(steps):
        try: 
            tangent = field(point)
        except ValueError:
            print("Returned early due to escaping the domain.")
            break

        u = controller.direction(field, point) * controller.force
        point = tangent.basepoint + (tangent.v + u) * stepsize
        yield tangent, u

