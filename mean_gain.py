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


def impt(s, t):
    """
    :param s: dictionary of documents for the query s <(str) doc_id: (int) rank>
    :param t: dictionary of documents for the query t <(str) doc_id: (int) rank>
    :return imp: the expected importance of s, compared to the other queries for the same topic

    it vaguely describes the similarity between s and t
    """
    sdocs = set(s.keys())
    tdocs = set(t.keys())
    # uncomment if ranks star from 0
    imp = np.sum([1 / (t[d] + 1) for d in sdocs.intersection(tdocs)])

    # uncomment if you are using normalized scores
    #imp = np.sum([1 / (1-t[d]) for d in sdocs.intersection(tdocs)])
    return imp


def rels(s, d):
    """
    :param s: dictionary of documents for the query s <(str) doc_id: (int) rank>
    :param d: document id
    :return:
    """

    # uncomment if ranks star from 0
    return 1 / np.sqrt(s[d] + 1)

    # uncomment if you are using normalized scores
    #return 1 / np.sqrt(1-s[d])
    #return s[d]


def gain(t, q):
    # compute the Importance of each query, compared to the others
    imps = [impt(s, t) for s in q]
    erel = {}
    # Foreach document in the query, compute the expected relevance
    for d in t:
        rela = 0
        for idx, s in enumerate(q):
            rela += imps[idx] * (rels(s, d) if d in s else 0)
        rela = rela / np.sum(imps)

        erel[d] = rela

    gain = 1 - np.mean([(1 - rela) for d, rela in erel.items()])
    #gain = 1 - np.prod([(1 - rela) for d, rela in erel.items()])

    return (gain, erel)


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

# remove the "nan" query:
#del run_dict['160']['160006']
print("computing gain")
recall_levels = ['recall_100', 'recall_1000', 'recall_10000']
#recall_levels = ['recall_100']

for rl in recall_levels:

    rln = int(rl.split("_")[1])

    results = []
    gains = []
    #run_dict considering only the first 'rln' documents
    reduced_run_dict = {tid:
                            {qid:
                                {d: r['r'] for d, r in run_dict[tid][qid].items() if r['r'] < rln}
                                #{d: r['s'] for d, r in run_dict[tid][qid].items() if r['r'] < rln}
                             for qid in run_dict[tid]}
                        for tid in run_dict}

    one_freq = []
    for tid in reduced_run_dict:
        gains_t = []
        scores_t = []
        for qid in run_dict[tid]:
            g, erel = gain(reduced_run_dict[tid][qid],
                           [reduced_run_dict[tid][q] for q in run_dict[tid] if q != qid])
            if np.isnan(g):
                print(f"nan found: {qid} {queries[tid][qid]}")
                g = 0
            #print(qid, g, scores[qid][rl])
            gains_t.append(g)
            scores_t.append(scores[qid][rl])
            gains.append(g)

        g=np.array(gains_t)
        one_freq.append(g)
        tau, p = sts.kendalltau(gains_t, scores_t)


        if np.isnan(tau):
            c = "nan"
        else:
            c = ("significant" if p < 0.05 and tau>0 else "not significant")

        results.append([tid, tau if not np.isnan(tau) else 0, c])


    print(np.sum([np.sum(g==1) for g in one_freq]))
    one_freq = [str(np.sum(g==1)) for g in one_freq]
    #one_freq = {v: np.sum(one_freq==v) for v in list(set(one_freq))}
    '''
    sns.countplot(one_freq, order=['0', '1', '2', '3', '4', '5', '6', '7'], color=sns.color_palette()[0])
    plt.xlabel("frequency of gain=1")
    plt.ylabel("number of topics")
    plt.title(f"{rl}")
    plt.show()
    '''
    results = pd.DataFrame(results, columns=["topic", "tau", "pval"])
    results['x1'] = results.index

    # Plot results
    sns.set_theme(style="whitegrid")
    #fig, axes = plt.subplots(2, 1, figsize=(42, 22))
    fig, axes = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=1.)
    #fig.tight_layout()
    sns.lineplot(x=np.arange(len(gains)), y=sorted(gains), ax=axes[0])
    axes[0].set_xlabel("queries", fontsize=20)
    axes[0].set_ylabel(r"GAIN", fontsize=20)
    #axes[0].set_title("Distribution of the GAIN over all queries", fontsize=24)


    #axes[0].set_fontsize(24)

    kt =  results['tau']
    kt[np.isnan(kt)] = 0
    pval = results['pval']

    print(
        f'{np.quantile(kt, 0.25):.4f}\t{np.mean(kt):.4f} ({np.std(kt):.4f})\t{np.quantile(kt, 0.75):.4f}\t{np.sum(pval=="significant")}')



    sns.scatterplot(data=results, x='x1', y='tau', hue="pval",
                    palette=['tab:blue', 'tab:red', 'black'],
                    hue_order=['significant', 'not significant', 'nan'], ax = axes[1])
    #for index, row in results.iterrows():
    #    plt.text(row['x1'], row['tau'], row['topic'], fontsize=10)


    axes[1].set_ylim([-1.1, 1.1])
    axes[1].set_xlabel("topics", fontsize=20)
    axes[1].set_ylabel(r"Kendall's $\tau$", fontsize=20)
    sig = np.sum((results['pval']=='significant' )&(results['tau']>0))
    sig_tot = np.sum((results['pval']=='significant' ))

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].get_legend().remove()
    plt.legend(handles, labels, ncol=len(labels), fontsize=1, loc='upper center',
               bbox_to_anchor=(0.48, 1.4), frameon=False)
    plt.tight_layout()
    plt.savefig(f'../data/plots/gain_mean_{rl}.pdf')