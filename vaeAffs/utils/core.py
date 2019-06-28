import numpy as np




def reorder_and_invert(affinities, offsets, number_of_attractive_channels, dist_per_dir=4):

    nr_offsets = len(offsets)
    assert affinities.shape[0] == nr_offsets  # nr of affinities should match nr of offsets
    nr_directions = nr_offsets // dist_per_dir
    assert nr_offsets == nr_directions*dist_per_dir

    indexlist = [dist_per_dir*j+i for i in range(dist_per_dir) for j in range(nr_directions)]

    affinities = affinities[indexlist]
    offsets = [offsets[ind] for ind in indexlist]

    affinities[:number_of_attractive_channels] *= -1
    affinities[:number_of_attractive_channels] += 1

    return affinities, offsets


def give_index_of_new_order(nr_offsets, dist_per_dir=4):
    """
    If you have the affinities ordered in a way that those of same
    direction are clustered together this indexlist allows for
    easy reordering such that the same distances are clustered
    :param nr_offsets: How many offsets are there
    :param dist_per_dir: How many offsets are there in one particular direction
    :return:
    """
    nr_directions = nr_offsets // dist_per_dir
    assert nr_offsets == nr_directions * dist_per_dir

    indexlist = [dist_per_dir * j + i for i in range(dist_per_dir) for j in range(nr_directions)]

    return indexlist


def exclude_some_short_edges(affinities, offsets, z_dir=True, sampling_factor=2, n_directions=8):
    """
    Warning: currently doesn`t work correctly if more than one offset per direction is to be considered

    If you want to decrease the number of short offsets, this function removes 1-1/sampling_factor of short edges
    :param affinities:
    :param offsets:
    :param z_dir:
    :param sampling_factor:
    :param n_directions:
    :return:
    """
    indexlist = []
    if z_dir:
        indexlist += [0, 1]
        indexlist += [i for i in range(2, n_directions + 2, sampling_factor)]
        indexlist += [i for i in range(n_directions + 2, affinities.shape[0])]
    else:
        indexlist += [i for i in range(0, n_directions, sampling_factor)]
        indexlist += [i for i in range(n_directions, affinities.shape[0])]


    return affinities[indexlist], [offsets[ind] for ind in indexlist]



class Annotator():
    def __init__(self, z_values):
        self.z_values = z_values

    def format_coord(self, x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        numcols, numrows = self.z_values.shape
        if 0 <= col < numcols and 0 <= row < numrows:
            z = self.z_values[row, col]
            return f'x={x:.1f}, y={y:.1f}, z={z}'
        else:
                return 'x=%1.4f, y=%1.4f' % (x, y)


