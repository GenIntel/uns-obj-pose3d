import torch

def mean_avg_std_with_id(features, ids, ids_count):
    """
    Args:
        features (torch.Tensor): NxF
        ids (torch.Tensor): N,
    Returns:
        std (float)
    """

    device = features.device
    feature_dim = features.shape[-1]

    # calculate variances
    # 1. avg
    all_features_avg_per_vertex = torch.zeros((ids_count, feature_dim)).to(device=device)
    all_features_avg_per_vertex_mask_zero = torch.zeros((ids_count)).to(device=device, dtype=bool)
    for vertex_id in range(ids_count):
        all_features_avg_per_vertex[vertex_id] = features[ids == vertex_id].mean()
        if features[ids == vertex_id, 0].numel() <= 1:
            all_features_avg_per_vertex[vertex_id] = 0.
            all_features_avg_per_vertex_mask_zero[vertex_id] = True

    # 2. deviation from avg
    all_features_dev_per_vertex = torch.zeros((ids_count, feature_dim)).to(device=device)
    for vertex_id in range(ids_count):
        vector_devs = (features[ids == vertex_id] - all_features_avg_per_vertex[vertex_id:vertex_id + 1])
        # if len(vector_devs) == 2:
        #    logger.info(vector_devs.std(dim=0))
        all_features_dev_per_vertex[vertex_id] = vector_devs.std(dim=0)
        if features[ids == vertex_id, 0].numel() <= 1:
            all_features_dev_per_vertex[vertex_id] = 0.

    return all_features_dev_per_vertex[~all_features_avg_per_vertex_mask_zero].mean()