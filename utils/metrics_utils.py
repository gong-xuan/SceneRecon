import numpy as np
import open3d as o3d


def eval_mesh(file_pred, file_trgt, threshold=.05, down_sample=.02, pcd_loaded=False):
    """ Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points
        filter: only evaluate ['dist1', 'recal', 'prec', 'fscore']
    Returns:
        Dict of mesh metrics
    """
    if pcd_loaded:
        pcd_pred = file_pred
        pcd_trgt = file_trgt
    else:
        pcd_pred = o3d.io.read_point_cloud(file_pred)
        pcd_trgt = o3d.io.read_point_cloud(file_trgt)
    #
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist1 = nn_correspondance(verts_pred, verts_trgt)
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    
    metrics = {'dist1': np.mean(dist2),
               'dist2': np.mean(dist1),
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               }
    return metrics


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def eval_depth(depth_pred, depth_trgt, dataset='scannet'):
    """ Computes 2d metrics between two depth maps

    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth

    Returns:
        Dict of metrics
    """
    mask1 = depth_pred > 0 
    if dataset=='scannet':
        # ignore values where prediction is 0 (% complete)
        mask = (depth_trgt < 10) * (depth_trgt > 0) * mask1
    elif dataset == 'sevenscenes':#results same with l114
        MIN_DEPTH, MAX_DEPTH = 0., 4 
        depth_pred[depth_pred < MIN_DEPTH] = MIN_DEPTH
        depth_pred[depth_pred > MAX_DEPTH] = MAX_DEPTH
        # pred_depth = pred_depth[0:480, 0:640]
        mask = (depth_trgt < MAX_DEPTH) * (depth_trgt > MIN_DEPTH) * mask1
    
    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred - depth_trgt)
    abs_rel = abs_diff / depth_trgt
    sq_diff = abs_diff ** 2
    sq_rel = sq_diff / depth_trgt
    sq_log_diff = (np.log(depth_pred) - np.log(depth_trgt)) ** 2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25 ** 2).astype('float')
    r3 = (thresh < 1.25 ** 3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics

def mean_each_sample(flattened_value, valid_npixel):
    batch_size = valid_npixel.shape[0]
    means = []
    startp = 0
    for b in range(batch_size):
        endp = startp+valid_npixel[b]
        means.append(flattened_value[startp: endp].mean())
        startp = endp
    return sum(means)/batch_size

def eval_depth_batch(depth_pred, depth_trgt, dataset='scannet', simplify=False):
    """ Computes 2d metrics between two depth maps

    Args:
        depth_pred: bs*h*w, unseen is -1 in pytorch3d
        depth_trgt: bs*h*w

    Returns:
        Dict of metrics
    """
    mask1 = depth_pred > 0  # ignore values where prediction is 0 (% complete)

    if dataset=='scannet':
        # ignore values where prediction is 0 (% complete)
        mask = (depth_trgt < 10) * (depth_trgt > 0) * mask1
    elif dataset == 'sevenscenes':#results same with l114
        MIN_DEPTH, MAX_DEPTH = 0., 4 
        depth_pred[depth_pred < MIN_DEPTH] = MIN_DEPTH
        depth_pred[depth_pred > MAX_DEPTH] = MAX_DEPTH
        # pred_depth = pred_depth[0:480, 0:640]
        mask = (depth_trgt < MAX_DEPTH) * (depth_trgt > MIN_DEPTH) * mask1

    depth_pred = depth_pred[mask]#flattened to 1D
    depth_trgt = depth_trgt[mask]

    abs_diff = np.abs(depth_pred - depth_trgt)
    abs_rel = abs_diff / depth_trgt
    sq_diff = abs_diff ** 2
    sq_rel = sq_diff / depth_trgt
    sq_log_diff = (np.log(depth_pred) - np.log(depth_trgt)) ** 2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25 ** 2).astype('float')
    r3 = (thresh < 1.25 ** 3).astype('float')

    #may need nanmean
    metrics = {}
    if simplify:
        metrics['AbsRel'] = np.mean(abs_rel)
        metrics['AbsDiff'] = np.mean(abs_diff)
        metrics['SqRel'] = np.mean(sq_rel)
        metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
        metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
        metrics['r1'] = np.mean(r1)
        metrics['r2'] = np.mean(r2)
        metrics['r3'] = np.mean(r3)
        metrics['complete'] = np.mean(mask1.astype('float')) 
    else:
        npixel_each = mask.sum(axis=(1,2))
        metrics['AbsRel'] = mean_each_sample(abs_rel, npixel_each)
        metrics['AbsDiff'] = mean_each_sample(abs_diff, npixel_each)
        metrics['SqRel'] = mean_each_sample(sq_rel, npixel_each)
        metrics['RMSE'] = np.sqrt(mean_each_sample(sq_diff, npixel_each))
        metrics['LogRMSE'] = np.sqrt(mean_each_sample(sq_log_diff, npixel_each))
        metrics['r1'] = mean_each_sample(r1, npixel_each)
        metrics['r2'] = mean_each_sample(r2, npixel_each)
        metrics['r3'] = mean_each_sample(r3, npixel_each)
        metrics['complete'] = mean_each_sample(mask1.astype('float'), npixel_each)    
    return metrics

