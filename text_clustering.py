# Text documents clustering

import sys
import json
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

docs = [
    "coffee cafe finger food cruasan sandwich",
    "pub beer food bar drink coffee snacks",
    "museum history archeology second world war",
    "hotel restaurant travel check in check out visit tourism"
]

stop_words = set(['the', 'a', 'an'])

def documents_gen(file_name):

    with open(file_name) as f:
        for line in f:
            yield  line.strip()

def main():

    vectorizer = TfidfVectorizer(min_df=0.05, token_pattern='[a-z0-9_\-:]+',
            stop_words=stop_words)
    tfidf = vectorizer.fit_transform(documents_gen(sys.argv[1]))

    feature_names = vectorizer.get_feature_names()

    for item in itertools.izip(feature_names, tfidf.data):
        print item

    km = MiniBatchKMeans(n_clusters=10, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)

    km.fit(tfidf)

    for center in km.cluster_centers_:
        cluster = [feature_names[idx] for idx, w in enumerate(center) if w > 0]
        print '--------------------------------'
        print cluster

if __name__ == '__main__':
    main()
