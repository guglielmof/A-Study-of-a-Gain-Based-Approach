import utils
import importlib
import common_parameters
import pytrec_eval

utils = importlib.reload(utils)
common_parameters = importlib.reload(common_parameters)

def compute_measure(run=None, qrels=None, measure='recall.100,1000,5000,10000'):
    if run is None:
        with open(common_parameters.run_path, "r") as F:
            run = pytrec_eval.parse_run(F)
    if qrels is None:
        queries = utils.import_queries()
        with open(common_parameters.qrels_path, "r") as F:
            reduced_qrels = pytrec_eval.parse_qrel(F)

        qrels = utils.expand_qrels(reduced_qrels, queries)


    topics = list(qrels.keys())

    evaluators = {k: pytrec_eval.RelevanceEvaluator({k: q}, {measure}) for k, q in qrels.items()}
    scores = {t: evaluators[t].evaluate({t: run[t]})[t] for t in topics}

    return scores