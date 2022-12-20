import numpy as np
import numpy.typing as npt
from typing import TypeVar, Generic, List, Callable, Any

FloatVector = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]
BaseSpaceType = TypeVar('BaseSpaceType')
OutputType = TypeVar('OutputType')

class Domain(Generic[BaseSpaceType]):
    """
        A domain in a vector space.
    """
    @property
    def dim(self) -> int:
        raise NotImplementedError

    def __contains__(self, point: BaseSpaceType) -> bool:
        raise NotImplementedError

class SquareDomain(Domain, Generic[BaseSpaceType]):
    """
        A square domain in the plane.
    """

    def __init__(self, min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def __contains__(self, point: BaseSpaceType) -> bool:
        raise NotImplementedError

class RealSquareDomain(SquareDomain):
    """
        A square domain in the plane with real valued coordinates.
    """
    @property
    def dim(self) -> int:
        return 2

    def __contains__(self, point: FloatVector) -> bool:
        in_range = self.min_x <= point[0] <= self.max_x and self.min_y <= point[1] <= self.max_y
        return in_range

class IntegerSquareDomain(SquareDomain):
    """
        A square domain in the plane with integer valued coordinates.
    """
    @property
    def dim(self) -> int:
        return 2

    def __contains__(self, point: IntVector) -> bool:
        in_range = self.min_x <= point[0] <= self.max_x and self.min_y <= point[1] <= self.max_y
        return in_range

class TangentVector(Generic[BaseSpaceType, OutputType]):
    def __init__(self, basepoint: BaseSpaceType, v: OutputType) -> None:
        """
            A tangent vector at a point in a vector space.
            basepoint: The point in the base space at which the tangent vector is defined.
            v: The actual tangent vector.
        """
        self.basepoint = basepoint
        self.v = v

    def __str__(self) -> str:
        return f'({self.basepoint}, {self.v})'

class Field(Generic[BaseSpaceType, OutputType]):
    """
        A vector field over a domain is simply a function from that domain at a point to the tangent space at that point.
    """

    def __init__(self, domain: SquareDomain, function: Callable[[BaseSpaceType], OutputType]) -> None:
        """
            Parameters:
                domain: The domain of the field.
                function: The function defining the field.
        """
        self.domain = domain
        self.function = function

    def __call__(self, point: BaseSpaceType) -> TangentVector[BaseSpaceType, OutputType]:
        if point not in self.domain:
            raise ValueError('Point not in domain.')
        
        result = TangentVector(point, self.function(point))
        
        return result

def BilinearInterpolation(field: Field, point: FloatVector) -> TangentVector[FloatVector, FloatVector]:
    """
        Interpolate a field at a point using bilinear interpolation.
        field: The field to interpolate.
        point: The point at which to interpolate the field.
    """
    # get the integer part of the point
    x = int(point[0])
    y = int(point[1])

    # get the fractional part of the point
    dx = point[0] - x
    dy = point[1] - y

    # get the 4 points around the point
    points = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

    # get the 4 values at the points
    values = [np.array(field([x, y]).v) for x, y in points]

    # interpolate the values
    interpolated: FloatVector = values[0] * (1 - dx) * (1 - dy) + values[1] * dx * (1 - dy) + values[2] * (1 - dx) * dy + values[3] * dx * dy

    return TangentVector(point, interpolated)

def IntegerFieldToReal(field: Field[IntVector, FloatVector]) -> Field[FloatVector, FloatVector]:
    """
        Convert an integer field to a real field by using bilinear interpolation at each point. 
        field: The integer-pointed field to interpolate.
    """
    def new_function(point: FloatVector) -> FloatVector:
        tangent_vec = BilinearInterpolation(field, point)
        return tangent_vec.v

    return Field(RealSquareDomain(field.domain.min_x, field.domain.max_x, field.domain.min_y, field.domain.max_y), new_function)