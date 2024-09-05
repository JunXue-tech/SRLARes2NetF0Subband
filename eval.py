import numpy as np
import sys

def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    """
    计算ASV系统的错误率。
    
    参数:
    tar_asv (np.ndarray): 目标说话者的ASV分数。
    non_asv (np.ndarray): 非目标说话者的ASV分数。
    spoof_asv (np.ndarray): 欺骗攻击的ASV分数。
    asv_threshold (float): ASV决策的阈值。
    
    返回:
    tuple: Pfa_asv, Pmiss_asv, Pmiss_spoof_asv 分别为ASV系统的错误接受率、错误拒绝率以及欺骗错误率。
    """
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):
    """
    计算DET曲线。

    参数:
    target_scores (np.ndarray): 目标分数。
    nontarget_scores (np.ndarray): 非目标分数。

    返回:
    tuple: 错误拒绝率(frr), 错误接受率(far), 阈值(thresholds)
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # 根据分数排序
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # 计算错误拒绝率和错误接受率
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """
    计算EER（等错误率）。
    
    参数:
    target_scores (np.ndarray): 目标分数。
    nontarget_scores (np.ndarray): 非目标分数。
    
    返回:
    tuple: EER值和对应的阈值。
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost):
    """
    计算串联检测成本函数 (t-DCF)。
    
    参数和返回值参考之前的说明文档。
    """
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    if C1 < 0 or C2 < 0:
        sys.exit(
            'You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_norm = tDCF / np.minimum(C1, C2)

    if print_cost:
        print(f't-DCF evaluation from [Nbona={bonafide_score_cm.size}, Nspoof={spoof_score_cm.size}] trials\n')
        print('t-DCF MODEL')
        print(f'   Ptar         = {cost_model["Ptar"]:8.5f}')
        print(f'   Pnon         = {cost_model["Pnon"]:8.5f}')
        print(f'   Pspoof       = {cost_model["Pspoof"]:8.5f}')
        print(f'   Cfa_asv      = {cost_model["Cfa_asv"]:8.5f}')
        print(f'   Cmiss_asv    = {cost_model["Cmiss_asv"]:8.5f}')
        print(f'   Cfa_cm       = {cost_model["Cfa_cm"]:8.5f}')
        print(f'   Cmiss_cm     = {cost_model["Cmiss_cm"]:8.5f}')

        if C2 == np.minimum(C1, C2):
            print(f'   tDCF_norm(s) = {C1 / C2:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n')
        else:
            print(f'   tDCF_norm(s) = Pmiss_cm(s) + {C2 / C1:8.5f} x Pfa_cm(s)\n')

    return tDCF_norm, CM_thresholds


def eerandtdcf(score_file, label_file, asv_label):
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,
        'Ptar': (1 - Pspoof) * 0.99,
        'Pnon': (1 - Pspoof) * 0.01,
        'Cmiss_asv': 1,
        'Cfa_asv': 10,
        'Cmiss_cm': 1,
        'Cfa_cm': 10,
    }
    
    asv_data = np.genfromtxt(asv_label, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float64)

    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    target = []
    nontarget = []
    target_score = []
    nontarget_score = []
    wav_lists = []
    score = {}
    lable_list = {}

    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                label = line[4]
                lable_list[wav_id] = label
                if label == "spoof":
                    nontarget.append(wav_id)
                else:
                    target.append(wav_id)

    with open(score_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[0]
                wav_lists.append(wav_id)
                score[wav_id] = line[3]
                
    for wav_id in target:
        target_score.append(float(score[wav_id]))
    for wav_id in nontarget:
        nontarget_score.append(float(score[wav_id]))
    
    target_score = np.array(target_score)
    nontarget_score = np.array(nontarget_score)
    
    eer_cm, Threshhold = compute_eer(target_score, nontarget_score)
    
    tDCF_curve, CM_thresholds = compute_tDCF(target_score, nontarget_score, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
                                             cost_model, True)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    print('ASV SYSTEM')
    print(f'   EER            = {eer_asv * 100:8.5f} %')
    print(f'   Pfa            = {Pfa_asv * 100:8.5f} %')
    print(f'   Pmiss          = {Pmiss_asv * 100:8.5f} %')
    print(f'   1-Pmiss,spoof  = {(1 - Pmiss_spoof_asv) * 100:8.5f} %')

    print('\nCM SYSTEM')
    print(f'   EER            = {eer_cm * 100:8.5f} %')

    print('\nTANDEM')
    print(f'   min-tDCF       = {min_tDCF:8.5f}')


def main():
    score_file = "./models/eval.txt"  # 使用变量代替硬编码路径
    label_file = "/path/to/ASVspoof2019_LA_cm_protocols.txt"  # 使用可配置变量
    asv_label = "/path/to/ASVspoof2019_LA_asv_scores.txt"  # 使用可配置变量

    eerandtdcf(score_file, label_file, asv_label)


if __name__ == '__main__':
    main()
