import utils
import importlib
import common_parameters
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient

utils = importlib.reload(utils)
common_parameters = importlib.reload(common_parameters)


class IndexManager:

    def __init__(self, index):
        self.es = Elasticsearch(timeout=300)
        self.IC = IndicesClient(self.es)
        self.index = index
        tot_docs_q = {"query": {'match_all': {}}}
        self.tot_docs = self.es.count(index=index, body=tot_docs_q)['count']

    def change_model(self, model, **kwargs):
        """change similarity model for current index"""
        model_settings = {'type': model}
        if kwargs is not None:
            for key, value in kwargs.items():
                model_settings[key] = value
        # close index before updating
        self.es.indices.close(index=self.index)
        # update settings
        similarity_settings = {'similarity': {'custom_model': model_settings}}
        self.es.indices.put_settings(index=self.index, body=similarity_settings)
        # re-open index after updating
        self.es.indices.open(index=self.index)
        return True

    def analyzeQuery(self, query):
        analyzed_query = self.IC.analyze(index=self.index, body={"text": query, 'field': 'content'})
        qsplit = [t['token'] for t in analyzed_query['tokens']]

        return qsplit


class Preretrieval(IndexManager):

    def __init__(self, index):
        super().__init__(index)

        # This query will retrieve every document in the index.
        query = {
            'query': {
                'match_all': {}
            }
        }

        # Send a search request to Elasticsearch.
        # curl -X GET localhost:9200/goma/_search -H 'Content-Type: application/json' -d @query.json
        res = self.es.search(index=index, body=query)

        # The response is a json object, the listing is nested inside it.
        # Here we are accessing the first hit in the listing.

        self.doc_type = res['hits']['hits'][0]["_type"]

    def analyzeQuery(self, query):
        analyzed_query = self.IC.analyze(index=self.index, body={"text": query, 'field': 'content'})
        qsplit = [t['token'] for t in analyzed_query['tokens']]

        return qsplit


class PreretrievalIDF(Preretrieval):

    def __init__(self, index):
        super().__init__(index)
        self.switcher = {'avgidf': np.mean, 'maxidf': np.max, 'stdidf': np.std}

    def getIDF(self, query):

        qsplit = self.analyzeQuery(query)

        idf = np.zeros(len(qsplit))
        for e, t in enumerate(qsplit):
            q = {"query": {'term': {'content': t}}}

            df = self.es.count(index=self.index, body=q)['count']
            idf[e] = np.log10(self.tot_docs / df) if df > 0 else 0

        return idf

    def getPrediction(self, query, ptype='avgidf'):
        if ptype not in self.switcher and ptype != 'all':
            raise NotImplementedError

        idf = self.getIDF(query)
        if ptype == 'all':
            return {pt: self.switcher[pt](idf) for pt in self.switcher}
        else:
            return self.switcher[ptype](idf)

class PreretrievalSCQ(Preretrieval):
    def __init__(self, index):
        super().__init__(index)
        self.switcher = {'sumscq': np.sum, 'maxscq': np.max, 'meanscq': np.mean}

    def getSCQ(self, query):
        tv = self.es.termvectors(index=self.index, doc_type=self.doc_type, term_statistics=True, field_statistics=False,
                                 positions=False, offsets=False, fields=["content"], body={"doc": {"content": query}})

        tv = tv['term_vectors']['content']['terms']
        scq = np.array([(1 + np.log(t['ttf'])) * np.log(1 + self.tot_docs / t['doc_freq']) if ('doc_freq' in t and t['doc_freq']>0) else 0 for _, t in tv.items()])
        return scq

    def getPrediction(self, query, ptype='sumscq'):
        if ptype not in self.switcher and ptype != 'all':
            raise NotImplementedError

        scq = self.getSCQ(query)
        if ptype == 'all':
            return {pt: self.switcher[pt](scq) for pt in self.switcher}
        else:
            return self.switcher[ptype](scq)


class PreretrievalVAR(Preretrieval):
    def __init__(self, index):
        super().__init__(index)
        self.switcher = {'sumvar': np.sum, 'maxvar': np.max, 'meanvar': np.mean}

    def getIDsDocsContainingT(self, t):

        q = {"query": {'term': {'content': t}}}

        # Initialize the scroll
        page = self.es.search(index=self.index, scroll='2m', size=1000, body=q)
        sid = page['_scroll_id']
        scroll_size = page['hits']['total']

        total_ids = []

        # Start scrolling
        while scroll_size > 0:
            page = self.es.scroll(scroll_id=sid, scroll='2m')
            # Update the scroll ID
            sid = page['_scroll_id']
            # Get the number of results that we returned in the last scroll
            total_ids += [i['_id'] for i in page['hits']['hits']]
            scroll_size = len(page['hits']['hits'])

        return total_ids

    def compute_weight(self, termStats):

        return 1 + np.log(termStats['term_freq']) * np.log(1 + self.tot_docs / termStats['doc_freq'])

    def getVAR(self, query):

        qsplit = self.analyzeQuery(query)

        vars = np.zeros(len(qsplit))

        for e, t in enumerate(qsplit):
            docsContainingT = self.getIDsDocsContainingT(t)
            print(len(docsContainingT))
            weights = []
            chunk_size = 100
            for i in range(int(len(docsContainingT) / chunk_size) + 1):
                tv = self.es.mtermvectors(index=self.index, doc_type=self.doc_type, term_statistics=True,
                                          field_statistics=False,
                                          positions=False, offsets=False, fields=["content"],
                                          ids=",".join(docsContainingT[i * chunk_size:(i + 1) * chunk_size]))

                # for docStruct in tv['docs']:
                #    print(docStruct['term_vectors']['content']['terms'][t])
                weights += [self.compute_weight(docStruct['term_vectors']['content']['terms'][t]) for docStruct in
                            tv['docs']]
            weights = np.array(weights)
            print(len(weights))
            vars[e] = 1 / len(weights) * np.sum(weights - np.mean(weights))

        return vars

    def getPrediction(self, query, ptype='all'):
        if ptype not in self.switcher and ptype != 'all':
            raise NotImplementedError

        vars = self.getVAR(query)
        if ptype == 'all':
            return {pt: self.switcher[pt](vars) for pt in self.switcher}
        else:
            return self.switcher[ptype](vars)
