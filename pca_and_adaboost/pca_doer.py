import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree   import DecisionTreeClassifier
import numpy as np
from sklearn.decomposition import PCA
import sys
sys.path.append("..")
import common_graphics

class pca_doer:
	def __init__(self):
		self.X = np.loadtxt("../train/traindata.tsv", dtype=float)
		self.y = np.loadtxt("../train/trainyvalues.txt", dtype=int)
		self.testdata = np.loadtxt("../testa/testdata.tsv", dtype=float)
		self.X_r = None
		self.pca = PCA(n_components=2)
		self.classifier=None
		self.testyvals=None
		self.test2dims=None

	def do_pca(self):
		self.X_r = self.pca.fit_transform(X)
		self.test2dims = pca.transform(self.testdata)
		print ("explained variance ration (first two components): %s" %str(pca.explained_variance_ratio_))

	def make_figure_after_pca(self):
		plt.figure()
		for c, i, target_name in zip("rb", [0, 1], ["zeroes", "ones"]):
    			plt.scatter(self.X_r[self.y == i, 0], self.X_r[self.y == i, 1], c=c, label=target_name)
		plt.show()

	def train_adaboost(self):
		d =DecisionTreeClassifier(class_weight="auto")
        	self.classifier = AdaBoostClassifier(d, n_estimators=250)
		self.classifier.fit(self.X_r, self.y)

	def train_extra_trees(self):
        	self.classifier = ExtraTreesClassifier(class_weight="auto", n_estimators=250)
		self.classifier.fit(self.X_r, self.y)

	def predict(self):
                self.testyvals = self.classifier.predict(self.test2dims)

	def decision_boundaries_2class_ada(self):
		plot_colors = "br"
                plot_step = 0.02
                class_names = "01"
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                x_min, x_max = self.X_r[:, 0].min() - 1, self.X_r[:, 0].max() + 1
                y_min, y_max = self.X_r[:, 1].min() - 1, self.X_r[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

                Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
                plt.axis("tight")

                for i, n, c in zip(range(2), class_names, plot_colors):
                	idx = np.where(self.y == i)
                	plt.scatter(self.X_r[idx, 0], self.X_r[idx, 1], c=c, cmap=plt.cm.Paired, label="Class %s" % n)
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.legend(loc='upper right')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Decision Boundary')

                twoclass_output = self.classifier.decision_function(X_r)
                plot_range = (twoclass_output.min(), twoclass_output.max())
                plt.subplot(122)
                for i, n, c in zip(range(2), class_names, plot_colors):
                        plt.hist(twoclass_output[self.y == i],bins=10,range=plot_range,facecolor=c,label='Class %s' % n, alpha=0.5)
                x1, x2, y1, y2 = plt.axis()
                plt.axis((x1, x2, y1, y2 * 1.2))
                plt.legend(loc='upper right')
                plt.ylabel('Samples')
                plt.xlabel('Score')
                plt.title('Decision Scores')

                plt.tight_layout()
                plt.subplots_adjust(wspace=0.35)
                plt.show()

	

	def draw_decision_boundary(self):
		cg.plot_decision_boundary(self.classifier, self.X_r, self.y)	

	def draw_ROC(self):
		cg.plot_ROC(self.classifier, self.X_r, self.y)

	def print_scores(self):
		cg.print_f1_precision_recall_score(self.classifier, self.X_r,self.y)

	def plot_oob():
		cg.plot_oob(self.classifier, self.X_r, self.y)

if __name__=="__main__":
	p = pca_doer()
	p.do_pca()
	p.make_figure_after_pca()	
	p.train_adaboost()
	p.predict()
	p.draw_decision_boundary()
	p.decision_boundaries_2class_ada()
	p.draw_ROC()
	p.draw_decision_boundary()
	p.train_extra_trees()
	p.plot_oob()
	p.print_scores()
