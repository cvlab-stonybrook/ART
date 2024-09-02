import json
from tqdm import tqdm
import cv2
from copy import deepcopy
import numpy as np
import argparse
from saliency_metrics import nss, cc
import pickle

def zero_one_similarity(a, b):
    return float(a == b)

def generate_sal_map(action_map):
    #generate saliency maps by Gaussian Blurring
    saliency_map = cv2.GaussianBlur(action_map, [0,0], 9,9) * 255.
    return saliency_map


def get_GT_maps_CC(scanpath_dict):
    #here saliency maps are aggregated per REF_ID
    sal_map_dict = {}
    agg_sal_map_dict = {}
    for ref_id, scanpaths in tqdm(scanpath_dict.items()):
        if ref_id not in sal_map_dict:
            sal_map_dict[ref_id] = {}
        for scanpath in scanpaths:
            for word_idx in range(len(scanpath['FIX_Y_synced'])):
                if word_idx not in sal_map_dict[ref_id]:
                    sal_map_dict[ref_id][word_idx] = []
                action_map = np.zeros((320, 512))
                y_pack = scanpath['FIX_Y_synced'][word_idx]
                x_pack = scanpath['FIX_X_synced'][word_idx]
                for y,x in zip(y_pack, x_pack):
                    action_map[int(y),int(x)] = 1.
                sal_map_dict[ref_id][word_idx].append(action_map)
    for ref_id in sal_map_dict.keys():
        if ref_id not in agg_sal_map_dict:
            agg_sal_map_dict[ref_id] = {}
        for word_idx in sal_map_dict[ref_id].keys():
            agg_sal_map_dict[ref_id][word_idx] = generate_sal_map(np.mean(deepcopy(sal_map_dict[ref_id][word_idx]), axis=0))
    return deepcopy(agg_sal_map_dict)

def get_GT_maps_NSS(scanpath_dict):
    sal_map_dict = {}
    for ref_id, scanpaths in tqdm(scanpath_dict.items()):
        if ref_id not in sal_map_dict:
            sal_map_dict[ref_id] = {}
        for scanpath in scanpaths:
            for word_idx in range(len(scanpath['FIX_Y_synced'])):
                if word_idx not in sal_map_dict[ref_id]:
                    sal_map_dict[ref_id][word_idx] = []
                action_map = np.zeros((320, 512))
                y_pack = scanpath['FIX_Y_synced'][word_idx]
                x_pack = scanpath['FIX_X_synced'][word_idx]
                for y,x in zip(y_pack, x_pack):
                    action_map[int(y),int(x)] = 1.
                sal_map_dict[ref_id][word_idx].append(action_map)
    return sal_map_dict


def get_saliency_maps_predictions(scanpaths):
    #here saliency maps are aggregated per REF_ID
    sal_map_dict = {}
    for scanpath in tqdm(scanpaths):
        if scanpath['REF_ID'] not in sal_map_dict:
            sal_map_dict[scanpath['REF_ID']] = {}
        for word_idx in range(len(scanpath['Y'])):
            if word_idx not in sal_map_dict[scanpath['REF_ID']]:
                sal_map_dict[scanpath['REF_ID']][word_idx] = []
            action_map = np.zeros((320, 512))
            y_pack = scanpath['Y'][word_idx]
            x_pack = scanpath['X'][word_idx]
            for y,x in zip(y_pack, x_pack):
                action_map[int(y),int(x)] = 1.
            sal_map_dict[scanpath['REF_ID']][word_idx].append(generate_sal_map(action_map))
    return sal_map_dict


def get_cc_scores(pred_dict, gt_dict, repeat_num=10):
    mean_cc = []
    for ref_id in tqdm(gt_dict.keys()):
        pred_list = pred_dict[ref_id]
        gt_list = gt_dict[ref_id]
        ref_cc = []
        for word_idx in gt_list.keys():
            word_cc = []
            gt_map = deepcopy(gt_list[word_idx])
            if word_idx not in pred_list:
                pred_maps = [np.zeros_like(gt_map) for j in range(repeat_num)]
            else:
                pred_maps = deepcopy(pred_list[word_idx])
            for pred_map in pred_maps:
                word_cc.append(cc(deepcopy(pred_map), deepcopy(gt_map)))
            ref_cc.append(np.mean(word_cc))
        mean_cc.append(np.mean(ref_cc))
    return {'CC':np.mean(mean_cc)}

def get_nss_scores(pred_dict, gt_dict, repeat_num=10):
    mean_nss = []
    for ref_id in tqdm(gt_dict.keys()):
        pred_list = pred_dict[ref_id]
        gt_list = gt_dict[ref_id]
        ref_nss = []
        for word_idx in gt_list.keys():
            word_nss = []
            gt_maps = deepcopy(gt_list[word_idx])
            for gt_map in gt_maps:
                curr_nss = []
                if word_idx not in pred_list:
                    pred_maps = [np.zeros_like(gt_map) for j in range(repeat_num)]
                else:
                    pred_maps = deepcopy(pred_list[word_idx])

                for pred_map in pred_maps:
                    score = nss(deepcopy(pred_map), deepcopy(gt_map))
                    if score is not None:
                        curr_nss.append(deepcopy(score))
                if len(curr_nss) > 0:
                    word_nss.append(np.mean(curr_nss))
            if len(word_nss) > 0:
                ref_nss.append(np.mean(word_nss))
        mean_nss.append(np.mean(ref_nss))
    return {'NSS': np.mean(mean_nss)}

def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))

def _Levenshtein_Dmatrix_initializer(len1, len2):
    Dmatrix = []

    for i in range(len1):
        Dmatrix.append([0] * len2)

    for i in range(len1):
        Dmatrix[i][0] = i

    for j in range(len2):
        Dmatrix[0][j] = j

    return Dmatrix


def _Levenshtein_cost_step(Dmatrix, string_1, string_2, i, j, substitution_cost=1):
    char_1 = string_1[i - 1]
    char_2 = string_2[j - 1]

    # insertion
    insertion = Dmatrix[i - 1][j] + 1
    # deletion
    deletion = Dmatrix[i][j - 1] + 1
    # substitution
    substitution = Dmatrix[i - 1][j - 1] + substitution_cost * (char_1 != char_2)

    # pick the cheapest
    Dmatrix[i][j] = min(insertion, deletion, substitution)

def _Levenshtein(string_1, string_2, substitution_cost=1):
    # get strings lengths and initialize Distances-matrix
    len1 = len(string_1)
    len2 = len(string_2)
    Dmatrix = _Levenshtein_Dmatrix_initializer(len1 + 1, len2 + 1)

    # compute cost for each step in dynamic programming
    for i in range(len1):
        for j in range(len2):
            _Levenshtein_cost_step(Dmatrix,
                                   string_1, string_2,
                                   i + 1, j + 1,
                                   substitution_cost=substitution_cost)

    if substitution_cost == 1:
        max_dist = max(len1, len2)
    elif substitution_cost == 2:
        max_dist = len1 + len2

    return Dmatrix[len1][len2]


def scanpath2clusters(meanshift, scanpath):
    predictions_list = []
    global_predictions_list = []
    patch_size = (16,16)
    patch_num=(20,32)
    im_h, im_w = 240, 320
    target_im_h, target_im_w = 320, 512
    for j in range(len(scanpath['X'])):
        string = []
        xs = scanpath['X'][j]
        ys = scanpath['Y'][j]
        if len(xs) == 0:
            if len(predictions_list) == 0:
                symbol = meanshift.predict([[512//2, 320//2]])[0]
                string.append(symbol)
            else:
                string.append(predictions_list[-1][-1])
        else:
            for i in range(len(xs)):
                symbol = meanshift.predict([[xs[i], ys[i]]])[0]
                string.append(deepcopy(symbol))
                global_predictions_list.append(deepcopy(symbol))
        predictions_list.append(string)
    return predictions_list, global_predictions_list

# compute sequence score
def compute_SS(predictions, clusters):
    results = []
    for scanpath in tqdm(predictions):
        key = scanpath['REF_ID']
        ms = clusters[key]
        strings = ms['strings_online']
        cluster = ms['cluster']

        predictions_list, global_predictions = scanpath2clusters(cluster, scanpath)
        scores = []
        scores_pack = []
        scores_ed = []
        scores_pack_ed = []

        for gt_list_ in strings.values():
            curr_score = []
            curr_score_ed = []
            stringlist1 = deepcopy(gt_list_)
            stringlist2 = deepcopy(predictions_list)
            if len(gt_list_) > len(predictions_list):
                stringlist2 += [[deepcopy(predictions_list[-1][-1]) for lmn in range(len(gt_list_[ijk+len(predictions_list)]))] for ijk in range(len(gt_list_) - len(predictions_list))]
            elif len(gt_list_) < len(predictions_list):
                stringlist1 += [[deepcopy(gt_list_[-1][-1]) ]for ijk in range(len(predictions_list) - len(gt_list_))]
                

            for idx, gt in enumerate(stringlist1):
                score = nw_matching(deepcopy(stringlist2[idx]), deepcopy(gt))
                curr_score.append(score)
                score_ed = _Levenshtein(deepcopy(stringlist2[idx]), deepcopy(gt))
                curr_score_ed.append(score_ed)
            scores_pack.append(np.mean(curr_score))
            scores_pack_ed.append(np.mean(curr_score_ed))
            
        for global_gt in ms['strings'].values():
            global_score = nw_matching(deepcopy(global_predictions), deepcopy(global_gt))
            global_score_ed = _Levenshtein(deepcopy(global_predictions), deepcopy(global_gt))

            scores.append(global_score)
            scores_ed.append(global_score_ed)
        result = {}
        result['REF_ID'] = key
        result['SS_pack'] = np.mean(scores_pack)
        result['SS'] = np.mean(scores)
        result['ED_pack'] = np.mean(scores_pack_ed)
        result['ED'] = np.mean(scores_ed)

        results.append(result)
    return results

def get_seq_score(predictions, clusters):
    results = compute_SS(predictions, clusters)
    ss_pack = []
    ss = []
    ed_pack = []
    ed = []
    for res in results:
        ss_pack.append(res['SS_pack'])
        ss.append(res['SS'])
        ed_pack.append(res['ED_pack'])
        ed.append(res['ED'])
    return {'SS_pack':np.mean(ss_pack), 'SS': np.mean(ss), 'ED_pack':np.mean(ed_pack), 'ED': np.mean(ed)}

def get_metrics(predictions):
    clusters = np.load('data/clusters_refcocogaze.npy', allow_pickle=True).item()

    

    gt = pickle.load(open('data/refcocogaze_test_synced.pkl', mode='rb'))

    sal_pred_dict = get_saliency_maps_predictions(deepcopy(predictions))
    sal_gt_cc_dict = get_GT_maps_CC(deepcopy(gt))
    sal_gt_nss_dict = get_GT_maps_NSS(deepcopy(gt))

    cc_dict =  get_cc_scores(deepcopy(sal_pred_dict), deepcopy(sal_gt_cc_dict))
    nss_dict =  get_nss_scores(deepcopy(sal_pred_dict), deepcopy(sal_gt_nss_dict))

    print(get_seq_score(predictions, clusters=clusters))
    print(cc_dict)
    print(nss_dict)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description ='Score metrics')
    parser.add_argument('--predicts_file', default='results/inference_scanpaths.json', type=str)
    args = parser.parse_args()
    predictions = json.load(open(args.predicts_file))
    
    get_metrics(predictions=predictions)


