def cal_inst_metric(seg_mask, gt_inst_mask, gt_ids):
    gt_tp_iou_thresh = 0.3
    pred_fp_iou_thresh = 0.2
    inst_area_thresh = 50
    tp = 0
    fp = 0
    fn = 0
    max_pred_dist_thresh = 10
    gt_seg = np.zeros_like(seg_mask)
    for gt_id in gt_ids:
        gt_mask = gt_inst_mask == gt_id
        gt_seg[gt_mask] = 1
        gt_area = gt_mask.sum()
        if gt_area < inst_area_thresh:
            continue
        iou = np.sum(gt_mask * seg_mask) / (gt_area + 0.000001)
        if iou > gt_tp_iou_thresh:
            tp += 1
        else:
            fn += 1
    pred_num, pred_label, _, _ = cv2.connectedComponentsWithStats(seg_mask.astype(np.uint8), connectivity=8)
    gt_seg_dist_mask = cv2.distanceTransform((~gt_seg).astype(np.uint8), cv2.DIST_L2, 0)
    for i in range(1, pred_num + 1):
        cur_pred_mask = pred_label == i
        pred_area = cur_pred_mask.sum()
        if pred_area < inst_area_thresh:
            continue
        pred_iou = np.sum(cur_pred_mask * gt_seg) / (pred_area + 0.0000001)
        if pred_iou < pred_fp_iou_thresh:
            avg_dist_to_gt = (gt_seg_dist_mask[cur_pred_mask]).mean()
            if avg_dist_to_gt < max_pred_dist_thresh:
                continue
            fp += 1
    return tp, fp, fn