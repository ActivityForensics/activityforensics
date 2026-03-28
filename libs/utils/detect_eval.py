import numpy as np
import pandas as pd
from libs.utils.Evaluation import average_recall_vs_avg_nr_proposals, compute_average_precision_detection


def compute_AP_AR(pred_time, gt_time, score,
                  iou_thresholds_ap=np.linspace(0.5, 0.95, 10),
                  iou_thresholds_ar=np.linspace(0.5, 0.95, 10),
                  ar_points=(10, 20, 50, 100),
                  subset='', max_avg_nr_proposals=100):
    """
    完全复现 ActivityNet 官方 evaluation_detection 与 evaluation_proposal 的逻辑。
    支持自定义 AR 评估点 ar_points。
    返回所有结果为 0~1 之间的浮点数（不乘 100）。
    """

    # ===== 构造 prediction DataFrame =====
    video_ids, t_start, t_end, scores = [], [], [], []
    for vid, (p, s) in enumerate(zip(pred_time, score)):
        if len(p) == 0:
            continue
        for seg, sc in zip(p, s):
            video_ids.append(f"vid{vid}")
            t_start.append(float(seg[0]))
            t_end.append(float(seg[1]))
            scores.append(float(sc))
    pred_df = pd.DataFrame({
        'video-id': video_ids,
        't-start': t_start,
        't-end': t_end,
        'score': scores,
        'label': [0] * len(video_ids)
    })

    # ===== 构造 ground truth DataFrame =====
    video_ids, t_start, t_end = [], [], []
    for vid, g in enumerate(gt_time):
        if len(g) == 0:
            continue
        for seg in g:
            video_ids.append(f"vid{vid}")
            t_start.append(float(seg[0]))
            t_end.append(float(seg[1]))
    gt_df = pd.DataFrame({
        'video-id': video_ids,
        't-start': t_start,
        't-end': t_end,
        'label': [0] * len(video_ids)
    })

    # ===== Detection (AP) =====
    ap = compute_average_precision_detection(gt_df, pred_df, iou_thresholds_ap)
    mAP = np.mean(ap)
    mAP_at_tIoU = [f"mAP@{t:.2f} {v:.4f}" for t, v in zip(iou_thresholds_ap, ap)]
    det_result = f"Detection ({subset}): average-mAP {mAP:.4f}  {' '.join(mAP_at_tIoU)}"
    # print(det_result)

    # ===== Proposal (AR) =====
    recall, avg_recall, proposals_per_video = average_recall_vs_avg_nr_proposals(
        gt_df, pred_df,
        max_avg_nr_proposals=max_avg_nr_proposals,
        tiou_thresholds=iou_thresholds_ar
    )

    # 计算 AR@N，对应 recall 的固定索引（与官方一致）
    AR_dict = {}
    for n in ar_points:
        idx = int(n - 1)  # 因为 recall 的索引从 0 开始
        AR_dict[f"AR@{n}"] = float(np.mean(recall[:, idx]))

    AR_mean = np.mean(list(AR_dict.values()))

    # 输出与官方格式一致
    ar_str = "  ".join([f"{k} {v:.4f}" for k, v in AR_dict.items()])
    prop_result = f"Proposal ({subset}): {ar_str}"
    # print(prop_result)

    # ===== 汇总结果 =====
    results = {
        "mAP": float(mAP),
        **{f"mAP@{str(round(t, 2)).rstrip('0').rstrip('.')}": float(v) for t, v in zip(iou_thresholds_ap, ap)},
        **AR_dict,
        "mAR": float(AR_mean)
    }

    return results


if __name__ == '__main__':
    IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

    pred_time = [
        np.array([[0.1, 0.4], [0.85, 1.0]]),
        np.array([[0.2, 0.5], [0.6, 0.7], [0.9, 1.0]]),
        np.array([[0.05, 0.15]])
    ]
    gt_time = [
        np.array([[0.15, 0.45], [0.5, 0.8]]),
        np.array([[0.25, 0.55]]),
        np.array([[0.08, 0.12]])
    ]
    score = [
        np.array([0.95, 0.60]),
        np.array([0.85, 0.55, 0.40]),
        np.array([0.70])
    ]

    out = compute_AP_AR(
        pred_time, gt_time, score,
        iou_thresholds_ap=IOU_THRESHOLDS,
        iou_thresholds_ar=IOU_THRESHOLDS,
        ar_points=(1, 5, 10, 100),
        subset='validation'
    )

    print(out)
