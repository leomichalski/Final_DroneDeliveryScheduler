import numpy as np
from delivery.label_utils import can_uav_fly


def do_coord_exist(coord, matrix_shape):
    """Check if a 2D coordinate isn't out of bounds.
    Parameters
    ----------
    coord : numpy.ndarray
        2D coordinate i.e (0,0).
    matrix_shape : numpy.ndarray
        Shape of the matrix to where the coordinate should point.
    Returns
    -------
    bool
        Whether the coordinate exists.
    """
    return np.bitwise_and(coord < matrix_shape, coord >= 0).all()


def distance_between_3d_points(x1, y1, z1, x2, y2, z2):
    """Computes the euclidean distance between two 3D points.
    Parameters
    ----------
    x1 : int or float
        x coordinate of point 1.
    y1 : int or float
        y coordinate of point 1`.
    z1 : int or float
        z coordinate of point 1`.
    x2 : int or float
        x coordinate of point 2`.
    y2 : int or float
        y coordinate of point 2`.
    z2 : int or float
        z coordinate of point 2`.
    Returns
    -------
    int or float
        Euclidean distance between two 3D points..
    """
    return ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**(1/2)


def hashable_coord(coord, mtx_shape):
    """Transforms a coord list into a hashable type.
    Parameters
    ----------
    coord : list or ndarray
        2D coordinate i.e (0,0).
    mtx_shape : list or ndarray
        Shape of the image i.e [1920, 1080, 3] or [1920, 1080].
    Returns
    -------
    int
        A representation of the coord that is hashable.
    """
    return coord[0] * mtx_shape[0] + coord[1]


class Node(object):
    """Short summary.
    Parameters
    ----------
    coord : list or ndarray
        2D coordinate i.e (0,0).
    mtx_shape : list or ndarray
        Shape of the adj_matrix i.e [7, 7].
    label : type
        Description of parameter `label`.
    height : type
        Description of parameter `height`.
    distance : type
        Description of parameter `distance`.
    path : type
        Description of parameter `path`.
    Attributes
    ----------
    hash : type
        Description of attribute `hash`.
    __hash__ : type
        Description of attribute `__hash__`.
    """

    def __init__(self, coord, mtx_shape,
                 label=None, height=None, distance=-1, path=[]):  # , visited=False):
        self.coord = coord
        self.mtx_shape = mtx_shape
        self.label = label
        self.height = height
        self.distance = distance
        self.path = path
        # self.visited = visited
        self.hash = self.__hash__(coord=self.coord, mtx_shape=self.mtx_shape)

    def __hash__(self, coord, mtx_shape):
        """Hash the node.
        Returns
        -------
        int
            A representation of the coord that is hashable.
        """
        return hashable_coord(coord=coord, mtx_shape=mtx_shape)


def find_shortest_path(start_coord, goal_coord_list, adj_matrix, weight_map):
    # TODO: [performance] move code to C/C++ or to a recursive language
    base_neighbours = np.asarray([[1, 0], [0, 1], [1, 1],
                                  [-1, 0], [0, -1], [-1, -1],
                                  [-1, 1], [1, -1]])
    shortest_paths_dict = {}
    goal_hash_list = [Node(coord=c, mtx_shape=adj_matrix.shape).hash
                      for c in goal_coord_list]

    start_node = Node(
        coord=start_coord,
        mtx_shape=adj_matrix.shape,
        path=[start_coord],
        distance=0,
        height=weight_map[start_coord[0]][start_coord[1]],
        label=adj_matrix[start_coord[0]][start_coord[1]],
    )

    shortest_paths_dict[start_node.hash] = start_node
    find_shortest_path_re(
        curr_node=start_node,
        goal_hash_list=goal_hash_list,
        adj_matrix=adj_matrix,
        weight_map=weight_map,
        shortest_paths_dict=shortest_paths_dict,
        base_neighbours=base_neighbours
    )

    path_list = []
    distance_list = []
    for h in goal_hash_list:
        node = shortest_paths_dict[h]
        path_list.append(node.path)
        distance_list.append(node.distance)

    return path_list, distance_list


def find_shortest_path_re(curr_node, goal_hash_list,
                          adj_matrix, weight_map,
                          shortest_paths_dict,
                          base_neighbours):
    """Recursive part of find_shortest_path. It doesn't return anything, it just updates the shortest_paths_dict.
    """
    neighbour_list = base_neighbours + np.asarray(curr_node.coord)
    for nb_node_coord in neighbour_list:
        # Ignore coords that do not exist i.e (-1, 99999999).
        if not do_coord_exist(nb_node_coord, curr_node.mtx_shape):
            continue
        # Ignore unreachable coords.
        nb_node_label = adj_matrix[nb_node_coord[0]][nb_node_coord[1]]
        if not can_uav_fly(nb_node_label):
            continue
        # If neighbour's already in shortest_paths_dict, access it. Otherwise,
        # create but DON'T put it into the shortest_paths_dict.
        nb_node_hash = Node.__hash__(
            self=None,
            coord=nb_node_coord,
            mtx_shape=curr_node.mtx_shape
        )
        node_was_here_before = nb_node_hash in shortest_paths_dict
        if node_was_here_before:
            nb_node = shortest_paths_dict[nb_node_hash]
        else:
            nb_node = Node(
                coord=nb_node_coord,
                mtx_shape=curr_node.mtx_shape,
                height=weight_map[nb_node_coord[0]][nb_node_coord[1]],
                label=nb_node_label,
            )
        # Calculate the distance from the current coord to the neighbour node.
        nb_node_distance = curr_node.distance + \
            distance_between_3d_points(
                curr_node.coord[0], curr_node.coord[1], abs(curr_node.height),
                nb_node.coord[0], nb_node.coord[1], abs(nb_node.height)
            )
        # Check if the calculated distance is shorter than the prev distance.
        # If so, update the values inside nb_node.
        if node_was_here_before:
            if (nb_node_distance > nb_node.distance)\
                    and (not nb_node.distance == -1):
                continue
        else:
            shortest_paths_dict[nb_node_hash] = nb_node
        nb_node.distance = nb_node_distance
        nb_node.path = curr_node.path + [nb_node.coord]
        shortest_paths_dict[nb_node.hash] = nb_node
        find_shortest_path_re(
            curr_node=nb_node,
            goal_hash_list=goal_hash_list,
            adj_matrix=adj_matrix,
            weight_map=weight_map,
            shortest_paths_dict=shortest_paths_dict,
            base_neighbours=base_neighbours
        )
        #nb_node.visited = True
        shortest_paths_dict[nb_node.hash] = nb_node

        #try:
        #    if np.array([shortest_paths_dict[h].visited
        #                 for h in goal_hash_list]).all():
        #        return
        #except KeyError:
        #    pass
