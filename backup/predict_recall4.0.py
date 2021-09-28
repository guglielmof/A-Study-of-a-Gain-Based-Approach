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
                                {d: r['r'] for d, r in run_dict[tid][qid].items() if r['r'] < rln}
                                #{d: r['s'] for d, r in run_dict[tid][qid].items() if r['r'] < rln}
                             for qid in run_dict[tid]}
                        for tid in run_dict}


    preds = []
    scores_t = []
    t_count = 0
    t_prob = 0
    for etid, tid in enumerate(reduced_run_dict):
        '''
        full_documents_set = set()
        for qid in reduced_run_dict[tid]:
            full_documents_set = full_documents_set.union(set(reduced_run_dict[tid][qid].keys()))
    
        asize = len(full_documents_set)

        bsize = len(set(reduced_run_dict[tid][qid].keys()))

        prediction = bsize**2/(asize*bsize)

        preds.append(prediction)
        scores_t.append(np.mean([scores[qid][rl] for qid in reduced_run_dict[tid]])/np.std([scores[qid][rl] for qid in reduced_run_dict[tid]]))
        '''

        predictions_t = []
        scores_t = []
        for qid1 in reduced_run_dict[tid]:
            similarities = []
            d1 = set(reduced_run_dict[tid][qid1])
            for qid2 in reduced_run_dict[tid]:
                d2 = set(reduced_run_dict[tid][qid2])
                intersection = d1.intersection(d2)
                tot_docs = d1.union(d2)
                did2idx = {d: e for e, d in enumerate(tot_docs)}
                v1 = np.zeros(len(did2idx))
                v2 = np.zeros(len(did2idx))
                for d in d1:
                    v1[did2idx[d]] = reduced_run_dict[tid][qid1][d]
                for d in d2:
                    v2[did2idx[d]] = reduced_run_dict[tid][qid2][d]
                v1 = v1/np.sqrt(np.sum(v1**2))
                v2 = v2/np.sqrt(np.sum(v2**2))
                similarities.append(np.dot(v1, v2))

            predictions_t.append(np.mean(similarities))
            scores_t.append(scores[qid1][rl])


        t, p = sts.kendalltau(predictions_t, scores_t)
        t_count += (p<0.05 and t>0)
        t_prob += (p<0.05)

    print(f"{t_count} ({t_prob})")