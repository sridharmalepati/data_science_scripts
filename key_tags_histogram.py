import sys
import itertools
from collections import Counter
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.mlab as mlab
from tools import open_osm_json

def plot_freq(counter, ax):
    ax.set_title('Frequence of key words')
    ax.set_xlabel('Key words')
    ax.set_ylabel('Frequence')

    data = counter.values()
    labels = counter.keys()

    ind = np.arange(len(data))
    rects = ax.bar(ind, data, width=0.5, color='blue',
        edgecolor='blue', log=False)

    ax.set_xticks(ind[::15])
    ax.set_xticklabels(labels, size=10, rotation=45)


def plot_hist(counter, ax):
    ax.set_title('Histogram')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Frequence')

    data = counter.values()

    sigma = np.std(data)
    mu = np.mean(data)

    n, bins, patches = ax.hist(data, 500, normed=1,
        facecolor='green', log=False)

    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, 'r--', linewidth=1)

if __name__ == '__main__':

    data = (obj.keys() for obj in open_osm_json(sys.argv[1]))
    counter = Counter(itertools.chain(*data))

    if 'user' in counter:
        del counter['user']

    figure = pl.figure(1)
    figure.clf()

    ax = figure.add_subplot(211)
    plot_freq(counter, ax)

    ax = figure.add_subplot(212)
    plot_hist(counter, ax)

    pl.show()
