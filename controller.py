from typing import Tuple, List
import numpy as np
import numpy.typing as npt
from field import Field, RealSquareDomain, IntegerFieldToReal, TangentVector
import networkx as nx

UnitVector = npt.NDArray[np.float64]
FloatVec = npt.NDArray[np.float64]
IntVec = npt.NDArray[np.int64]

class Controller():
    def __init__(self, target: FloatVec, force: float) -> None:
        self.target = target
        self.force = force

    def direction(self, field: Field[FloatVec, FloatVec], position: FloatVec) -> UnitVector:
        """
            Returns a unit vector in the direction to exert force.

                Parameters: 
                    field: The force field being navigated.
                    position: The position of the agent.

                Returns:
                    A unit vector in the direction to exert force.

        """        
        return np.zeros(field.domain.dim)

class GreedyController(Controller):
    def direction(self, field: Field[FloatVec, FloatVec], position: FloatVec) -> UnitVector:
        """
            Returns a unit vector in the direction to exert force.

                Parameters: 
                    field: The force field being navigated.
                    position: The position of the agent.

                Returns:
                    A unit vector in the direction to exert force.

        """        
        t = self.target        
        u = t - position

        return u / np.linalg.norm(u)

def PreSum(field: Field[FloatVec, FloatVec], stepsize: float) -> Field[IntVec, FloatVec]:
    xs = np.arange(field.domain.min_x, field.domain.max_x + stepsize, stepsize)
    ys = np.arange(field.domain.min_y, field.domain.max_y + stepsize, stepsize)
    
    cumsum = np.zeros((len(xs), len(ys), 2))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            x = np.round(x, 5)
            y = np.round(y, 5)

            cumsum[i, j] = field(np.array([x, y])).v
            if i != 0:
                cumsum[i, j] += cumsum[i - 1, j]
            if j != 0:
                cumsum[i, j] += cumsum[i, j - 1]
            if i != 0 and j != 0:
                cumsum[i, j] -= cumsum[i - 1, j - 1]
      
    def pos_to_indx(point: IntVec) -> tuple[int, int]:
        i = int(np.searchsorted(xs, point[0]))
        # check if closer to the left or right
        

        if i == len(xs) or (i != 0 and abs(xs[i - 1] - point[0]) < abs(xs[i] - point[0])):
            i -= 1
        
        j = int(np.searchsorted(ys, point[1]))
        # check if closer to the left or right
        if j == len(ys) or (j != 0 and abs(ys[j - 1] - point[1]) < abs(ys[j] - point[1])):
            j -= 1

        return i, j

    def access(point: IntVec) -> FloatVec:
        i, j = pos_to_indx(point)
        return cumsum[i, j]

    resultant_field = Field(field.domain, access)

    return resultant_field

 
def rectangle(lx, ly, tx, ty, stepsize, cs: Field) -> FloatVec:
    lx = lx - stepsize
    ly = ly - stepsize
    if [lx, ly] not in cs.domain:
        d = 0
    else:
        d = cs(np.array([lx, ly])).v
    if [lx, ty] not in cs.domain:
        b = 0
    else: 
        b = - cs(np.array([lx, ty])).v
    if [tx, ly] not in cs.domain:
        c = 0
    else:
        c = - cs(np.array([tx, ly])).v

    a = cs(np.array([tx, ty])).v 
    
    return a + b + c + d

def rect_count(lx, ly, tx, ty, stepsize) -> int:
    return len(np.arange(lx, tx + stepsize, stepsize)) * len(np.arange(ly, ty + stepsize, stepsize))

def rect_avg(lx, ly, tx, ty, stepsize, cs: Field) -> FloatVec:
    return rectangle(lx, ly, tx, ty, stepsize, cs) / rect_count(lx, ly, tx, ty, stepsize)

def get_loc(startloc: FloatVec, stepsize: float, pos: tuple[int, int]):
    """
        Translates between the position of a node in the graph and the position of the node in the field.

        Parameters:
            startloc: The origin location to position vertices relative to (a point in the vector field).
            stepsize: distance between two nodes.
            pos: The position of the node in the graph relative to the selected startloc. 
    """

    return startloc + np.array([pos[0] * stepsize, pos[1] * stepsize])

def displace(pos, displacement):
    return (pos[0] + displacement[0], pos[1] + displacement[1])

def can_make_it(direction: str, rowpower: float, velocity: FloatVec) -> bool:
    
    a = rowpower**2 - velocity[1]**2
    b = rowpower**2 - velocity[0]**2
    if direction == 'right':
        return a >= 0 and np.sqrt(a) + velocity[0] > 0
    elif direction == 'left':
        return a >= 0 and np.sqrt(a) - velocity[0] > 0
    elif direction == 'top':
        return b >= 0 and np.sqrt(b) + velocity[1] > 0
    elif direction == 'bottom':
        return b >= 0 and np.sqrt(b) - velocity[1] > 0
    else:
        raise ValueError("Direction must be 'right', 'left', 'top', or 'bottom'")

def get_travel_time(direction: str, rowpower: float, velocity: FloatVec, stepsize: float) -> float:
    if can_make_it(direction, rowpower, velocity):
        if direction == 'right':
            return stepsize / (np.sqrt(rowpower**2 - velocity[1]**2) + velocity[0])
        elif direction == 'left':
            return stepsize / (np.sqrt(rowpower**2 - velocity[1]**2) - velocity[0])
        elif direction == 'top':
            return stepsize / (np.sqrt(rowpower**2 - velocity[0]**2) + velocity[1])
        elif direction == 'bottom':
            return stepsize / (np.sqrt(rowpower**2 - velocity[0]**2) - velocity[1])
        else:
            raise ValueError("Direction must be 'right', 'left', 'top', or 'bottom'")
    else:
        return float('inf')

# def get_travel_time(target_point: FloatVec, rowpower: float, velocity: FloatVec) -> float:
#     A = velocity[0]**2 + velocity[1]**2 - rowpower**2
#     C = target_point[0]**2 + target_point[1]**2
#     B = 2 * (target_point[0] * velocity[0] + target_point[1] * velocity[1])



def graphify(field: Field[FloatVec, FloatVec], startpos: tuple[int, int], stepsize: float):
    cums = PreSum(field, stepsize)

    rowpower = 5

    graph = nx.DiGraph()
    graph.add_node(startpos, pos = get_loc(np.array([0, 0]), stepsize, startpos))

    visited = set()
    queue = [startpos]
    while len(queue) > 0:
        pos = queue.pop(0)
        if pos in visited:
            continue
        visited.add(pos)
        
        grid_mapping = {
            'right': {
                'displacement': (1, 0),
                'window_bottom_left': 'bottom',
                'window_top_right': 'top right',
                'inverse': 'left',
            },
            'left': {
                'displacement': (-1, 0),
                'window_bottom_left': 'bottom left',
                'window_top_right': 'top',
                'inverse': 'right',
            },
            'top': {
                'displacement': (0, 1),
                'window_bottom_left': 'left',
                'window_top_right': 'top right',
                'inverse': 'bottom',
            },
            'bottom': {
                'displacement': (0, -1),
                'window_bottom_left': 'bottom left',
                'window_top_right': 'right',
                'inverse': 'top',
            },
            'top right': {
                'displacement': (1, 1),
            },
            'bottom left': {
                'displacement': (-1, -1),
            },
            'top left': {
                'displacement': (-1, 1),
            },
            'bottom right': {
                'displacement': (1, -1),
            },
        }
       
        for child_direction in ['right', 'left', 'top', 'bottom']:
            child = displace(pos, grid_mapping[child_direction]['displacement'])

            if child not in graph.nodes:
                graph.add_node(child, pos = get_loc(np.array([0, 0]), stepsize, child))
                
            bl_disp = grid_mapping[grid_mapping[child_direction]['window_bottom_left']]['displacement']
            tr_disp = grid_mapping[grid_mapping[child_direction]['window_top_right']]['displacement']
            bl_loc = get_loc(np.array([0., 0.]), stepsize, displace(pos, bl_disp))
            tr_loc = get_loc(np.array([0., 0.]), stepsize, displace(pos, tr_disp))

            if bl_loc in field.domain and tr_loc in field.domain:
                avg_vec = rect_avg(bl_loc[0], 
                                bl_loc[1], 
                                tr_loc[0],
                                tr_loc[1], stepsize, cums)

                if (pos, child) not in graph.edges:
                    if can_make_it(child_direction, rowpower, avg_vec):
                        if child not in visited and child not in queue:
                            queue.append(child)
                        time = get_travel_time(child_direction, rowpower, avg_vec, stepsize)
                        graph.add_edge(pos, child, weight = time)

                if (child, pos) not in graph.edges:
                    parent_direction = grid_mapping[child_direction]['inverse']
                    if can_make_it(parent_direction, rowpower, avg_vec):
                        time = get_travel_time(parent_direction, rowpower, avg_vec, stepsize)
                        graph.add_edge(child, pos, weight = time)

    return graph

def F(point: np.ndarray) -> FloatVec:
    return np.array([np.sin(point[0] + point[1]), np.cos(point[0] - point[1])])

if __name__ == "__main__":

    from drawing import DrawVectorField

    domain = RealSquareDomain(-5, 5, -5, 5)
    field = Field[FloatVec, FloatVec](domain, F)

    DrawVectorField(field, stepsize = 1)

    graph = graphify(field, (0, 0), 1)

    import matplotlib.pyplot as plt
    pos = nx.get_node_attributes(graph,'pos')
    labels = nx.get_edge_attributes(graph, 'weight')
    # round labels to 2 decimal places
    labels = {k: round(v, 2) for k, v in labels.items()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    nx.draw(graph, pos, node_size = 10)
    plt.show()

    p = nx.shortest_path(graph, (0, 0), (10, 25), weight = 'weight')
    nx.draw(graph, pos, node_size = 10)
    path_edges = list(zip(p, p[1:]))
    nx.draw_networkx_nodes(graph, pos,nodelist=p,node_color='r', node_size = 13)
    nx.draw_networkx_edges(graph, pos,edgelist=path_edges,edge_color='r',width=2)
    plt.axis('equal')
    plt.show()

    # SS = 0.2

    # cs = PreSum(field, SS)

    # show cumsum as a quiver plot
    # DrawVectorField(field, stepsize = 1)
    # DrawVectorField(cs, stepsize = 1)
    
    # rectangle_vec_sum = rectangle(-1, -1, 1, 1, SS)
    # rectangle_vec_count = rect_count(-1, -1, 1, 1, SS)

    # res = rectangle_vec_sum / rectangle_vec_count
    # print(res)


    # start_point = np.array([0, 0])



