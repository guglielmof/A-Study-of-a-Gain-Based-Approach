import numpy as np
import utils
import importlib
import common_parameters
import eval_run
import scipy.stats as sts
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

utils = importlib.reload(utils)
common_parameters = importlib.reload(common_parameters)
eval_run = importlib.reload(eval_run)

queries = utils.import_queries()
print("computing measure")
scores = eval_run.compute_measure()

run_dict = {}


print("importing run")
with open(common_parameters.run_path, "r") as F:
    for l in F.readlines():
        qid, _, did, rank, score, _ = l.strip().split()

        tid = qid[:3]
        if tid not in run_dict:
            run_dict[tid] = {}
        if qid not in run_dict[tid]:
            run_dict[tid][qid] = {}
        run_dict[tid][qid][did] = {'r': int(rank), 's': float(score)}

recall_levels = ['recall_1000']

for rl in recall_levels:
    count_k = 0
    rln = int(rl.split("_")[1])

    results = []
    gains = []
    # run_dict considering only the first 'rln' documents
    reduced_run_dict = {tid:
                            {qid:
                                #{d: r['r'] for d, r in run_dict[tid][qid].items() if r['r'] < rln}
                                {d: r['s'] for d, r in run_dict[tid][qid].items() if r['r'] < rln}
                             for qid in run_dict[tid]}
                        for tid in run_dict}


    fig, axes = plt.subplots(8, 7, figsize=(42, 42))
    for etid, tid in enumerate(reduced_run_dict):

        did2idx = {}
        for qid in reduced_run_dict[tid]:
            for did in reduced_run_dict[tid][qid]:
                if did not in did2idx:
                    did2idx[did] = len(did2idx)

        qid2idx = {q: e for e, q in enumerate(reduced_run_dict[tid])}

        # build te relevance matrix
        R = np.zeros((len(reduced_run_dict[tid]), len(did2idx)))
        for qid in reduced_run_dict[tid]:
            for did in reduced_run_dict[tid][qid]:
                R[qid2idx[qid], did2idx[did]] = reduced_run_dict[tid][qid][did]

        R = normalize(R, norm='l2')


        NMF_mod = NMF(n_components=7, max_iter=500)

        W = NMF_mod.fit_transform(R)
        H = NMF_mod.components_

        predictions_t = []
        scores_t = []
        sm_t = []
        for qid in qid2idx:
            #predictions[qid] = cosine_similarity(np.matmul(W[qid2idx[qid]], H).reshape(1,-1), R[qid2idx[qid]].reshape(1, -1))
            #predictions_t.append(- np.mean((np.matmul(W[qid2idx[qid]], H)-R[qid2idx[qid]])**2))
            similarities = [cosine_similarity(W[qid2idx[qid]].reshape(1, -1), W[qid2idx[qid2]].reshape(1, -1))[0][0]
                            for qid2 in qid2idx if qid2 != qid]
            sm_t.append(similarities)
            predictions_t.append(np.mean(similarities))
            scores_t.append(scores[qid][rl])

        tau, p = sts.kendalltau(predictions_t, scores_t)
        print(tau, p)

        if tau>0 and p<0.05:
            count_k+=1

            sns.heatmap(sm_t, ax=axes[etid//7, etid%7])
            axes[etid // 7, etid % 7].set_title("s")
        else:
            sns.heatmap(sm_t, ax=axes[etid//7, etid%7])

    plt.show()
    print(count_k)

        #print("\n\n")

