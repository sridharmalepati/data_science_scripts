import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
import sys
sys.path.append("..")
import common_graphics


X = np.loadtxt("../traindata.tsv", dtype=float)
y = np.loadtxt("../trainyvalues.txt", dtype=int)
tes = np.loadtxt("../testa/testdata.tsv", dtype=float)


selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = map(lambda x: -np.log10(x) if x>0 else 0, selector.pvalues_)
scores /= scores.max()
ind =np.arange(100)
plt.bar(ind,scores, 0.35, color="blue")
plt.show()

nonzeroindices=[]

for i in range(100):
	if scores[i] > 0 : nonzeroindices.append(i)

smallerX =  X[,:nonzeroindices]
test_smaller = tes[,:nonzeroindices]

d = DecisionTreeClassifier(class_weight="auto")
e = AdaBoostClassifier(d, n_estimators=250)

e.fit(smallerX,y)o
predictedyvalues = e.predict(test_smaller)

cg.plot_ROC(e, X,y)
cg.print_f1_precision_recall_score(e, X,y)


et= ExtraTreesClassifier(class_weight="auto", bootstrap=True, oob_score=True, n_estimators=250)

print et.oob_score_

cg.plot_ROC(et, X,y)
cg.print_f1_precision_recall_score(et, X,y)

