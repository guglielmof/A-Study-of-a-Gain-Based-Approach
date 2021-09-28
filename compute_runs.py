import utils
import importlib
import common_parameters
import numpy as np
from elasticsearch import Elasticsearch

utils = importlib.reload(utils)
common_parameters = importlib.reload(common_parameters)

es = Elasticsearch(timeout=300)


queries = utils.import_queries()

n_queries = np.sum([len(queries[t]) for t in queries])

proc_queries = 0
with open(common_parameters.run_path, "w") as F:
    for t in queries:
        for q in queries[t]:
            print(f"searching for '{queries[t][q]}' ({proc_queries}/{n_queries})")

            res = es.search(index="clef2018", body={"query": {"match": {"content": queries[t][q]}}}, size=10000)
            #print("Got %d Hits:" % res['hits']['total'])

            for k, doc in enumerate(res['hits']['hits']):
                F.write(f"{q} Q0 {doc['_id']} {k} {doc['_score']} bm25_es_default\n")
                #print("%s) %s" % (doc['_id'], doc['_source']['content']))

            proc_queries+=1