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
import pytrec_eval
import predict_recall_fun

utils = importlib.reload(utils)
common_parameters = importlib.reload(common_parameters)
eval_run = importlib.reload(eval_run)
predict_recall_fun = importlib.reload(predict_recall_fun)

queries = utils.import_queries()
print("computing measure")
original_scores = eval_run.compute_measure()

with open(common_parameters.qrels_path, "r") as F:
    original_qrels = pytrec_eval.parse_qrel(F)

expanded_qrels = utils.expand_qrels(original_qrels, queries)

print("importing run")
with open(common_parameters.run_path, "r") as F:
    original_run = pytrec_eval.parse_run(F)

recall_levels = ['recall_10000']  # ['recall_100', 'recall_1000', 'recall_5000', 'recall_10000']

# dimension of the re-sampled pool
ns = [5, 10, 20, 30, 40, 50]
qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 5, 10]
ss = [1]
rank = [True]  # [True, False]

parametrizations = [(n, q, s, r) for n in ns for q in qs for s in ss for r in rank]
parametrizations_nosup = [(5, q, 0, r) for q in qs for r in rank]
results = []

for rl in recall_levels:
    rlc = int(rl.split("_")[1])

    reduced_run = utils.reduceRun(original_run, rlc)

    scores = {t: original_scores[t][rl] for t in original_scores}

    run_dict = {}
    for t in reduced_run:
        if t[:3] + "001" not in run_dict:
            run_dict[t[:3] + "001"] = {}
        if t not in run_dict[t[:3] + "001"]:
            run_dict[t[:3] + "001"][t] = {t[:3] + "001": reduced_run[t]}

    for p in parametrizations:
        out = predict_recall_fun.predict_rec(run_dict, original_qrels, scores, *p)
        results.append([rlc, *p, *out])
        print(results[-1])

    for p in parametrizations_nosup:
        out = predict_recall_fun.predict_rec(run_dict, original_qrels, scores, *p)
        results.append([rlc, *p, *out])
        print(results[-1])

results = pd.DataFrame(results, columns=['rlc', 'sup', 'q', 's', 'rank', 'mean_kendalls_best',
                                         'mean_kendall', 'n_good', 'n_sig'])

results.loc[results['s'] == 0, 'sup'] = 0

plt.figure(figsize=(42, 20))
sns.lineplot(data=results, x='q', y='n_good', hue='sup', palette=sns.color_palette("hls", len(ns)+1))

plt.xticks(fontsize=34)
plt.yticks(fontsize=34)
plt.legend(fontsize=34)
plt.xlabel("q", fontsize=34)
plt.ylabel(r"significant kendall's $\tau$", fontsize=34)
plt.savefig('line_plot.pdf')

