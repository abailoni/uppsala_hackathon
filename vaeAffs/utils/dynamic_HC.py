import vaeAffs

import nifty
import numpy as np
import vigra

# def generateGrid(gridSize):
#     def nid(x, y):
#         return x * gridSize[1] + y
#
#     G = nifty.graph.UndirectedGraph
#     g = G(gridSize[0] * gridSize[1])
#     for x in range(gridSize[0]):
#         for y in range(gridSize[1]):
#
#             u = nid(x, y)
#
#             if x + 1 < gridSize[0]:
#                 v = nid(x + 1, y)
#                 g.insertEdge(u, v)
#
#             if y + 1 < gridSize[1]:
#                 v = nid(x, y + 1)
#                 g.insertEdge(u, v)
#     return g



class DynamicCallback(nifty.graph.EdgeContractionGraphCallback):
    def __init__(self, merge_rule, debug=False):
        """
        :param merge_rule:
        :type merge_rule: AvgMergeRule
        """
        super(DynamicCallback, self).__init__()
        self.merge_rule = merge_rule
        self.debug = debug

    def contractEdge(self, edgeToContract):
        if self.debug:
            print("#############")
            print("Contract edge {}, weight {}".format(edgeToContract, self.merge_rule.values[edgeToContract]))

    def mergeEdges(self, aliveEdge, deadEdge):
        self.merge_rule.merge_edges(aliveEdge, deadEdge)
        if self.debug:
            print("Merge edges", aliveEdge, deadEdge)

    def mergeNodes(self, aliveNode, deadNode):
        if self.debug:
            print("Merge nodes", aliveNode, deadNode)

    def contractEdgeDone(self, contractedEdge):
        if self.debug:
            print("############")


class AvgMergeRule(object):
    def __init__(self, initial_size=100, increments=100,
                 max_size_PQ=1000, abs_value_in_PQ=True):
        assert abs_value_in_PQ, "First to use signed weights, fix push in vigra PQ"
        self.current_size = initial_size
        self.is_empty = np.ones(initial_size, dtype='bool')
        self.values = np.empty(initial_size)
        self.weights = np.empty(initial_size)
        self.increments = increments
        self.abs_value_in_PQ = abs_value_in_PQ
        self.check = False

        # self.PQ = vigra.utilities.ChangeablePriorityQueueFloat32Min(max_size_PQ)
        self.PQ = nifty.tools.ChangeablePriorityQueue(max_size_PQ)

    def extend_to(self, new_max_id):
        assert new_max_id >= self.current_size
        diff = new_max_id - self.current_size + 1
        diff = self.increments if diff < self.increments else diff

        self.values = np.concatenate([self.values, np.empty(diff)])
        self.weights = np.concatenate([self.weights, np.empty(diff)])
        self.is_empty = np.concatenate([self.is_empty, np.ones(diff, dtype='bool')])
        self.current_size += diff

    def compute_merged_values(self, val1, wgt1, val2, wgt2):
        new_wgt = wgt1 + wgt2
        new_val = (val1*wgt1 + val2*wgt2) / new_wgt
        return new_val, new_wgt

    def update_or_initialize(self, edge_id, value, weight):
        edge_ID_is_new = False
        if edge_id >= self.current_size:
            self.extend_to(edge_id)
            edge_ID_is_new = True
        if edge_ID_is_new or self.is_empty[edge_id]:
            self.is_empty[edge_id] = False
            new_value = value
            new_weight = weight
        else:
            # TODO: here I need to be careful if I vectorize it... (I could have multiple edges)
            new_value, new_weight = self.compute_merged_values(value, weight, self.values[edge_id], self.weights[edge_id])

        # Update values and push to PQ:
        self.values[edge_id], self.weights[edge_id] = new_value, new_weight
        PQ_prio = abs(new_value) if self.abs_value_in_PQ else new_value
        assert edge_id >= 0
        # FIXME: weird PQ that select first the minimum priorities
        if self.check:
            assert edge_id != 10079
        # assert not self.PQ.__contains__(-1)
        # assert 0.5 - PQ_prio >= 0. and 0.5 - PQ_prio <= 0.5
        self.PQ.push(edge_id, 0.5 - PQ_prio)
        # FIXME: in official, check if it is in...
        assert edge_id in self.PQ

    def merge_edges(self, aliveEdge, deadEdge):
        assert max(aliveEdge, deadEdge) < self.current_size
        assert not self.is_empty[aliveEdge] and not self.is_empty[deadEdge]
        self.update_or_initialize(aliveEdge, self.values[deadEdge], self.weights[deadEdge])

        # Delete dead edge from PQ:
        self.PQ.deleteItem(deadEdge)
        assert deadEdge not in self.PQ



class DynamicHC(object):
    def __init__(self, signed_weights, offsets, debug=False):
        assert len(signed_weights.shape) == 4 and signed_weights.shape[0] == len(offsets)
        self.shape = signed_weights.shape[1:]
        self.dim = len(self.shape)
        self.signed_weights = signed_weights
        self.offsets = np.array(offsets)
        self.debug = debug

        # Initialize stuff
        self.nb_nodes = np.array(signed_weights.shape[1:]).prod()
        self.is_node_active = np.zeros(self.nb_nodes, dtype='bool')
        self.coord_strides = self.get_coord_strides(self.shape)
        self.graph = nifty.graph.UndirectedGraph(self.nb_nodes)

        self.merge_rule = AvgMergeRule(max_size_PQ=self.nb_nodes * len(offsets))
        self.callback = DynamicCallback(self.merge_rule, debug=debug)
        self.c_graph = nifty.graph.edgeContractionGraph(self.graph, self.callback)

        # Initialize PQ:
        # TODO: this won't scale. Find a smarter way to do it
        # Actually the max is given by the max edges we will introduce in the graph (that will be hopefully much smaller than
        # the real number of predicted edges)


    def make_coord_active(self, coord):
        label = self.from_coord_to_label(coord)
        if self.debug:
            print("Activating node", label)
        assert not self.is_node_active[label]
        edge_values = self.signed_weights[:, coord[0], coord[1], coord[2]]
        neigh_coords, valid_coords = self.get_neighbouring_coords(coord)
        for edge_val, ngb_coord, is_valid in zip(edge_values, neigh_coords, valid_coords):
            if is_valid:
                ngb_label = self.from_coord_to_label(ngb_coord)
                self.create_or_update_edge(label, ngb_label, edge_val, weight=1)

        self.is_node_active[label] = True

    def run(self, starting_coordinate):
        self.starting_coordinate = starting_coordinate
        self.make_coord_active(starting_coordinate)
        PQ = self.merge_rule.PQ

        while True:
            if PQ.empty():
                break
            top_edge_id = PQ.top()
            # FIXME: Weird bug of the PQ that always contains -1 and negatives indices...
            # Moreover sometimes old edges deleted from the PQ are randomly added again...
            if top_edge_id < 0:
                PQ.pop()
                continue
            repr_edge_id = self.c_graph.findRepresentativeEdge(top_edge_id)
            if repr_edge_id != top_edge_id:
                PQ.pop()
                continue

            u, v = self.find_representative_linked_nodes(top_edge_id)

            assert u != v
            u_is_active = self.is_node_active[u]
            v_is_active = self.is_node_active[v]
            if u_is_active and v_is_active:
                PQ.pop()
                # Merge clusters, if the weight is positive:
                if self.merge_rule.values[repr_edge_id] > 0:
                    self.c_graph.contractEdge(repr_edge_id)
                    if self.c_graph.numberOfNodes % 500 == 0:
                        print("{} nodes remaining; {} active; {} edges inserted ({} in contracted g)".format(self.c_graph.numberOfNodes, self.is_node_active.sum(), self.graph.numberOfEdges, self.c_graph.numberOfEdges))
            else:
                # Find which one of the two nodes is not active and make it:
                assert u_is_active or v_is_active
                unactive_node = u if v_is_active else v
                # Make sure to have a single-pixel cluster:
                assert unactive_node == self.c_graph.findRepresentativeNode(unactive_node)
                unactive_coordinate = self.from_label_to_coord(unactive_node)
                self.make_coord_active(unactive_coordinate)

        print("Done")

    def run_v2(self, starting_coordinate):
        self.starting_coordinate = starting_coordinate
        self.make_coord_active(starting_coordinate)
        PQ = self.merge_rule.PQ

        while True:
            if PQ.empty():
                break
            top_edge_id = PQ.top()
            # FIXME: Weird bug of the PQ that always contains -1 and negatives indices...
            # Moreover sometimes old edges deleted from the PQ are randomly added again...
            if top_edge_id < 0:
                PQ.pop()
                continue
            repr_edge_id = self.c_graph.findRepresentativeEdge(top_edge_id)
            if repr_edge_id != top_edge_id:
                PQ.pop()
                continue

            u, v = self.find_representative_linked_nodes(top_edge_id)

            assert u != v
            u_is_active = self.is_node_active[u]
            v_is_active = self.is_node_active[v]
            PQ.pop()
            # Merge clusters, if the weight is positive:
            if self.merge_rule.values[repr_edge_id] > 0:
                self.c_graph.contractEdge(repr_edge_id)
                if self.c_graph.numberOfNodes % 500 == 0:
                    print("{} nodes remaining; {} active; {} edges inserted ({} in contracted g)".format(self.c_graph.numberOfNodes, self.is_node_active.sum(), self.graph.numberOfEdges, self.c_graph.numberOfEdges))
            if not (u_is_active and v_is_active):
                # Find which one of the two nodes is not active and make it:
                assert u_is_active or v_is_active
                unactive_node = u if v_is_active else v
                # Make sure to have a single-pixel cluster:
                unactive_coordinate = self.from_label_to_coord(unactive_node)
                self.make_coord_active(unactive_coordinate)

        print("Done")

    def run_v3(self, starting_coordinate):
        self.starting_coordinate = starting_coordinate
        self.make_coord_active(starting_coordinate)
        PQ = self.merge_rule.PQ

        while True:
            if PQ.empty():
                break
            top_edge_id = PQ.top()
            # FIXME: Weird bug of the PQ that always contains -1 and negatives indices...
            # Moreover sometimes old edges deleted from the PQ are randomly added again...
            if top_edge_id < 0:
                PQ.pop()
                continue
            repr_edge_id = self.c_graph.findRepresentativeEdge(top_edge_id)
            if repr_edge_id != top_edge_id:
                PQ.pop()
                continue

            u, v = self.find_representative_linked_nodes(top_edge_id)

            u_is_active = self.is_node_active[u]
            v_is_active = self.is_node_active[v]
            PQ.pop()
            # FIXME: another wierd stuff. In the official version I should assert u != v
            if u == v:
                continue

            # Merge clusters, if the weight is positive:
            activate_other_node = False
            if self.merge_rule.values[repr_edge_id] > 0:
                activate_other_node = True
                self.c_graph.contractEdge(repr_edge_id)
                if self.c_graph.numberOfNodes % 500 == 0:
                    print("{} nodes remaining; {} active; {} edges inserted ({} in contracted g)".format(self.c_graph.numberOfNodes, self.is_node_active.sum(), self.graph.numberOfEdges, self.c_graph.numberOfEdges))
            else:
                initial_cluster = self.c_graph.findRepresentativeNode(self.from_coord_to_label(self.starting_coordinate))
                if initial_cluster == u or initial_cluster == v:
                    activate_other_node = True
            if not (u_is_active and v_is_active) and activate_other_node:
                # Find which one of the two nodes is not active and make it:
                assert u_is_active or v_is_active
                unactive_node = u if v_is_active else v
                # Make sure to have a single-pixel cluster:
                unactive_coordinate = self.from_label_to_coord(unactive_node)
                self.make_coord_active(unactive_coordinate)

        print("Done")



    def create_or_update_edge(self, label_1, label_2, value, weight):
        # TODO: vectorize

        # We only add an edge if the two nodes were not already merged:
        repr_node_1 = self.c_graph.findRepresentativeNode(label_1)
        repr_node_2 = self.c_graph.findRepresentativeNode(label_2)
        if repr_node_1 != repr_node_2:
            # Check if we have already the edge in the contracted graph:
            edge_id = self.c_graph.findEdge(repr_node_1, repr_node_2)
            if edge_id < 0:
                # Add edge to both graphs:
                edge_id = self.graph.insertEdge(repr_node_1, repr_node_2)
                # Here the edgeUFD is also updated with a new cluster:
                self.c_graph.insertEdge(edge_id, repr_node_1, repr_node_2)
            self.merge_rule.update_or_initialize(edge_id, value, weight)

    def find_representative_linked_nodes(self, edge_id):
        assert edge_id < self.graph.numberOfEdges
        return self.c_graph.uv(edge_id)

    # def get_neighboring_node_labels(self, coord):
    #     labels = [self.c_graph.findRepresentativeNode(nd) for nd in self.from_coord_to_label(self.get_neighbouring_coords(coord, return_only_valid=True))]
    #     return np.array(labels)

    def get_neighbouring_coords(self, coords, return_only_valid=False):
        """
        :param coords: shape (nb_coords, 3) or (3, )

        The returned vector has shape: (nb_coords, nb_neighbors, 3) or (nb_neighbors, 3)
        The returned valid mask has shape (nb_coords, nb_neighbors) or (nb_neighbors)
        """
        expand_dim = False
        if len(coords.shape) == 1:
            coords = np.expand_dims(coords, axis=0)
            expand_dim = True
        assert len(coords.shape) == 2 and coords.shape[1] == 3
        assert (coords < 0).sum() == 0, "A negative coordinate was passed"

        neighbors_coord = []
        for off in self.offsets:
            neighbors_coord.append(coords + np.expand_dims(off, axis=0))
        neighbors_coord = np.stack(neighbors_coord, axis=1)
        valid_mask = neighbors_coord >= 0
        for d in range(self.dim):
            valid_mask[:,:,d] *= neighbors_coord[:,:,d] < self.shape[d]
        valid_mask = valid_mask.prod(axis=2)
        neighbors_coord *= np.expand_dims(valid_mask, axis=2)
        valid_mask = valid_mask.astype('bool')

        if neighbors_coord.shape[0] == 1 and expand_dim:
            neighbors_coord = neighbors_coord[0]
            valid_mask = valid_mask[0]

        if return_only_valid:
            assert expand_dim, "Only one coordinate is accepted in this case"
            return neighbors_coord[valid_mask]

        return neighbors_coord.astype('uint64'), valid_mask.astype('bool')

    def from_label_to_coord(self, labels):
        """
        :param labels: Array 1D or uint
        :return: Array with shape (nb_labels, 3)
        """
        expand_dim = False
        if not isinstance(labels, np.ndarray):
            labels = np.array([labels])
            expand_dim = True
        labels = labels.astype('uint64')
        assert len(labels.shape) == 1
        coords = np.zeros((labels.shape[0], 3), dtype='uint64')
        res_labels = labels.copy()
        for d in range(self.dim):
            coords_d = np.floor(res_labels / self.coord_strides[d]).astype('uint64')
            res_labels -= coords_d * self.coord_strides[d]
            coords[:, d] = coords_d
        if coords.shape[0] == 1 and expand_dim:
            coords = coords[0]
        return coords

    # def from_PQ_id_to_label(self, PQ_id):
    #     node_label = int(PQ_id / self.nb_nodes)
    #     nb_neighbour = PQ_id - node_label*self.nb_nodes
    #     return node_label, nb_neighbour

    def from_coord_to_label(self, coords):
        """
        :param coords: shape (nb_coords, 3) or (3, )
        :return:
        """
        expand_dim = False
        if len(coords.shape) == 1:
            coords = np.expand_dims(coords, axis=0)
            expand_dim = True
        assert len(coords.shape) == 2 and coords.shape[1] == 3
        assert (coords<0).sum() == 0, "A negative coordinate was passed"
        coords = coords.astype('uint64')
        node_labels = np.zeros(coords.shape[0], dtype='uint64')
        for d in range(self.dim):
            node_labels = node_labels + self.coord_strides[d]*coords[:,d]
        if node_labels.shape[0] == 1 and expand_dim:
            node_labels = node_labels[0]
        return node_labels

    def get_coord_strides(self, shape):
        DIM = len(shape)
        coord_strides = np.empty(DIM, dtype='uint64')
        coord_strides[-1] = 1
        for d in range(DIM-2, -1, -1):
            coord_strides[d] = shape[d+1] * coord_strides[d+1]
        return coord_strides

    def get_segmentation(self):
        out_segm = np.empty(self.shape, dtype='int64')
        active_nodes = np.empty(self.shape, dtype='int64')
        for n in range(self.nb_nodes):
            out_segm[tuple(self.from_label_to_coord(n))] = self.c_graph.findRepresentativeNode(n)
            active_nodes[tuple(self.from_label_to_coord(n))] = self.is_node_active[n]
        active_nodes[tuple(self.starting_coordinate)] = 2
        return out_segm, active_nodes

import os
from vaeAffs.utils.path_utils import get_abailoni_hci_home_path
import h5py
sample = "A"
data_path = os.path.join(get_abailoni_hci_home_path(), "../ialgpu1_local_home/datasets/cremi/SOA_affinities/sample{}_train.h5".format(sample))
crop_slice = ":,20:21,300:400,300:400"
from segmfriends.utils.various import parse_data_slice

crop_slice = parse_data_slice(crop_slice)
with h5py.File(data_path, 'r') as f:
    affs = f['predictions']['full_affs'][crop_slice]
    raw = f['raw'][crop_slice[1:]]

offsets = [[-1, 0, 0],
  [0, -1, 0],
  [0, 0, -1],
  [-2, 0, 0],
  [0, -3, 0],
  [0, 0, -3],
  [-3, 0, 0],
  [0, -9, 0],
  [0, 0, -9],
  [-4, 0, 0],
  [0, -27, 0],
  [0, 0, -27]]


# Fake duplicate affinities:
duplicate_affs = np.empty_like(affs)
for i, off in enumerate(offsets):
    duplicate_affs[i] = np.roll(affs[i], off)
affs = np.concatenate([affs, duplicate_affs], axis=0)
offsets = offsets + [[-off[0], -off[1], -off[2]] for off in offsets]

from neurofire.transform.affinities import Segmentation2AffinitiesDynamicOffsets, affinity_config_to_transform

from affogato.affinities import compute_multiscale_affinities, compute_affinities

_ ,mask = compute_affinities(np.zeros_like(raw, dtype='int64'), offsets)

print("Total valid edges: ", mask.sum())

dynHC = DynamicHC(affs-0.5, offsets)

# for n in range(dynHC.nb_nodes):
#     print(n, dynHC.from_label_to_coord(n))

dynHC.run_v3(np.array([0,20,20]))

print("Final number edges inserted: ", dynHC.graph.numberOfEdges)
print("Final number active nodes: ", dynHC.is_node_active.sum())

final_segm, active_nodes = dynHC.get_segmentation()
import segmfriends.vis as vis
fig, ax = vis.get_figure(1,4, figsize=(10,40))
vis.plot_segm(ax[0],final_segm, z_slice=0,background=raw)
vis.plot_output_affin(ax[1], affs, nb_offset=-1, z_slice=0)
# vis.plot_output_affin(ax[2], duplicate_affs, nb_offset=-1, z_slice=0)
vis.plot_segm(ax[3],active_nodes, z_slice=0,background=raw, highlight_boundaries=False, alpha_labels=0.7)
fig.savefig(os.path.join(get_abailoni_hci_home_path(), "../dynamic_HC_v3_2.png"))
