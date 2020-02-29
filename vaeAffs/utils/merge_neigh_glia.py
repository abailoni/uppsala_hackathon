import vigra
import numpy as np

# TODO:
# - first we compute boundary by combining offsets (-1, 0) and (0, -1)
# - we need to make sure to ignore the boundary with label 0 if possible
# - then we erode these segments
# - call this as a specific boundary-label

"""
The difference with just computing a boundary-patch with the usual method is that in this way also the other pixels inside segments are
trained to predict a conservative boundary margin.

This could create an effect with thin processes, but apparently it doesn't
"""

# -----------
# LOAD data
# -----------
from vaeAffs.utils.path_utils import get_abailoni_hci_home_path, get_trendytukan_drive_path
from segmfriends.utils.various import parse_data_slice, readHDF5, writeHDF5, readHDF5_from_volume_config
import os
import json
import h5py

from scipy.ndimage import zoom
import nifty.graph.rag as nrag



project_dir = os.path.join(get_trendytukan_drive_path(),"projects/pixel_embeddings")

IGNORE_LABELS = [0, 1]
nb_threads = 6

EXP_NAMES = [
    # TEST:
    # "v4_addSparseAffs_fullGT_eff",
    # "v4_addSparseAffs_fullGT_avgDirectVar",

    # VAL:
    "v4_addSparseAffs_eff",
    # "v4_onlySparseAffs_eff",
    # "v4_main_avgDirectVar",
    # "v4_addSparseAffs_avgDirectVar",
]

REQUIRED_STRINGS = [
    "affs_noLR"
]

EXCLUDE_STRINGS = [
    # "multicut_kerLin",
    "_mergedGlia",
]

INCLUDE_STRINGS = [
]

glia_mask_exp = "v4_onlyTrainGlia_eff"


MERGE_GLIA = True
COMPUTE_NEW_SCORES = True
PREPARE_SUBMISSION = False

# Load config:
from segmfriends.utils.various import yaml2dict
config = yaml2dict(os.path.join(get_abailoni_hci_home_path(), "pyCharm_projects/uppsala_hackathon/experiments/cremi/postproc_configs/prepare_samples_config.yml"))




import nifty


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






# for sample in ["A+", "B+", "C+"]:
for sample in ["C"]:
    crop_slice = parse_data_slice(config["volume_config"]["GT"]["crop_slice"][sample])

    # Load glia and GT:
    glia_prediction_path = os.path.join(project_dir, glia_mask_exp, "predictions_sample_{}.h5".format(sample))
    print("Loading glia for sample ", sample)
    glia_mask = readHDF5(glia_prediction_path, "glia_mask")[0]
    glia_mask = glia_mask[crop_slice]

    invalid_glia_mask = np.logical_or(glia_mask < 0., glia_mask > 1.)
    glia_mask[invalid_glia_mask] = 0
    print("Average glia mask (should be close to zero): ", glia_mask.mean())

    #  This is already cropped
    gt_segm = readHDF5_from_volume_config(sample, **config["volume_config"]["GT"])

    # Relabel GT just in case:
    # print("Nb GT before: ", np.unique(gt_segm).shape)
    gt_segm = vigra.analysis.labelVolumeWithBackground(gt_segm.astype('uint32'))
    # print("Nb GT after: ", np.unique(gt_segm).shape)


    for exp_name in EXP_NAMES:
        pred_dir = os.path.join(project_dir, exp_name, "out_segms")

        for item in os.listdir(pred_dir):
            if os.path.isfile(os.path.join(pred_dir, item)):
                filename = item
                if not filename.endswith(".h5") or filename.startswith(".") or not filename.startswith(sample):
                    continue
                skip = False
                for char in REQUIRED_STRINGS:
                    if char not in filename:
                        skip = True
                        break
                if not skip:
                    for excl_string in EXCLUDE_STRINGS:
                        if excl_string in filename:
                            skip = True
                            break
                    for excl_string in INCLUDE_STRINGS:
                        if excl_string in filename:
                            skip = False
                            break
                if skip:
                    continue

                pred_file = os.path.join(pred_dir, filename)

                # Load glia mask and predictions:
                print("Loading segm for ", exp_name)
                segm = readHDF5(pred_file, "segm_WS")

                valid_mask = np.ones_like(segm, dtype='bool')
                for ign in IGNORE_LABELS:
                    valid_mask = np.logical_and(valid_mask, segm == ign)

                if MERGE_GLIA:
                    postfix = "_mergedGlia"
                    from segmfriends.features.featurer import get_rag
                    rag, has_background = get_rag(segm, nb_threads)
                    assert not has_background
                    uv_ids = rag.uvIds()

                    assert segm.shape == glia_mask.shape
                    _, node_feat = nrag.accumulateMeanAndLength(rag, glia_mask.astype('float32'))
                    node_glia_mean = node_feat[:,0]

                    # Debug:
                    # glia_segmentation = nrag.projectScalarNodeDataToPixels(rag, (node_glia_mean>0.5).astype('float32'))
                    # writeHDF5(glia_segmentation.astype('uint32'), pred_file, "glia_segments")


                    # Compute glia-merge-probs:
                    u_glia_probs = node_glia_mean[uv_ids[:, 0]]
                    v_glia_probs = node_glia_mean[uv_ids[:, 1]]
                    # TODO: avoid to merge ignored-nodes!!
                    edges_to_be_merged = np.logical_and(u_glia_probs > 0.5, v_glia_probs > 0.5)
                    print("Number of edges to be merged: ", edges_to_be_merged.sum())

                    # Agglomeration:
                    print("Agglomerating...")
                    c_graph = nifty.graph.edgeContractionGraph(rag, nifty.graph.EdgeContractionGraphCallback())
                    uv_ids_to_merge = uv_ids[edges_to_be_merged, :]
                    for u, v in uv_ids_to_merge:
                        repr_u = c_graph.findRepresentativeNode(u)
                        repr_v = c_graph.findRepresentativeNode(v)
                        if repr_u != repr_v:
                            repr_edge = c_graph.findRepresentativeEdge(rag.findEdge(u,v))
                            c_graph.contractEdge(repr_edge)
                    node_labels = np.array([c_graph.findRepresentativeNode(u) for u in rag.nodes()])
                    segm = nrag.projectScalarNodeDataToPixels(rag, node_labels)
                else:
                    postfix = "_relabeled"

                assert gt_segm.shape == segm.shape

                # Find connected components
                gt_ignore_mask = gt_segm == 0
                segm += 1
                segm[gt_ignore_mask] = 0
                segm = vigra.analysis.labelVolumeWithBackground(segm.astype('uint32'))

                if COMPUTE_NEW_SCORES:


                    from segmfriends.utils.various import cremi_score
                    import yaml
                    evals = cremi_score(gt_segm, segm, border_threshold=None, return_all_scores=True)

                    scores_dir = os.path.join(project_dir, exp_name, "scores")
                    prev_score_file = os.path.join(scores_dir, filename.replace(".h5", ".yml"))

                    result_config = yaml2dict(prev_score_file)
                    result_config['score_WS'] = evals

                    new_score_file_path = prev_score_file.replace(".yml", "{}.yml".format(postfix))

                    with open(new_score_file_path, 'w') as f:
                        # json.dump(config_to_save, f, indent=4, sort_keys=True)
                        yaml.dump(result_config, f)

                # Save new segm:
                out_segm_path = pred_file.replace(".h5", "{}.h5".format(postfix))
                writeHDF5(segm.astype('uint32'), out_segm_path, "segm_WS")

                if PREPARE_SUBMISSION:
                    from vaeAffs.postproc.utils import prepare_submission

                    path_bbox_slice = config["volume_config"]["paths_padded_boxes"]
                    prepare_submission(sample, out_segm_path,
                                       inner_path_segm="segm_WS",
                                       path_bbox_slice=path_bbox_slice[sample],
                                       ds_factor=(1, 2, 2))



