from sklearn.externals.six.moves import xrange
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import  sklearn.cross_validation as cval


def plot_decision_boundary(model, X,y):
        plot_colors = "rb"
        cmap = plt.cm.RdBu
        plot_step = 0.02
        plot_step_coarser = 0.5
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
        plt.figure()
        for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)
        xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser), np.arange(y_min,y_max,plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")
        for i, c in zip(xrange(2), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=c, label=y[i],cmap=cmap)
        plt.show()

def plot_ROC(classifier, X,y):
	cv = StratifiedKFold(y, n_folds=2)
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	all_tpr = []

	for i, (train, test) in enumerate(cv):
    		probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    		fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    		mean_tpr += interp(mean_fpr, fpr, tpr)
    		mean_tpr[0] = 0.0
    		roc_auc = auc(fpr, tpr)
    		plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

	mean_tpr /= len(cv)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()
	

def print_f1_precision_recall_score(model, X,y):
	cv = cval.StratifiedKFold(y, 2)
	a = cval.cross_val_score(model, X, y, scoring="f1", cv=cv)
	b= cval.cross_val_score(model, X, y, scoring="precision", cv=cv)
	c = cval.cross_val_score(model, X, y, scoring="recall", cv=cv)
	print "f1", a[0]
	print "precision", b[0]
	print "recall", c[0]

def plot_oob(clf, X,y, min_estimators=1, max_estimator=250):
	error_rate=[]
	for i in range(min_estimators, max_estimators + 1):
        	clf.set_params(n_estimators=i)
        	clf.fit(X, y)
        	oob_error = 1 - clf.oob_score_
        	error_rate.append(oob_error)

    	plt.plot(error_rate)

	plt.xlim(min_estimators, max_estimators)
	plt.xlabel("n_estimators")
	plt.ylabel("OOB error rate")
	plt.show()
	
