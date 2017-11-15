"""transfer.py
~~~~~~~~~~~~~~

Implement transfer learning for RMNIST, based on the features learnt
by ResNet-18.

"""

#### Libraries
# My libraries
import data_loader 

# Third-party libraries
import sklearn
import sklearn.svm
import sklearn.neighbors
import sklearn.tree
import sklearn.ensemble
import sklearn.neural_network

# Configuration: whether to use expanded training data or not
expanded = False

def all_transfers():
    if expanded: sizes = [1, 5, 10]
    else: sizes = [1, 5, 10, 0]
    for n in sizes:
        print "\n\nUsing RMNIST/{}".format(n)
        transfer(n)
        
def transfer(n):
    td, vd, ts = data_loader.load_data(n, abstract=True, expanded=expanded)
    classifiers = [
        #sklearn.svm.SVC(),
        #sklearn.svm.SVC(kernel="linear", C=0.1),
        #sklearn.neighbors.KNeighborsClassifier(1),
        #sklearn.tree.DecisionTreeClassifier(),
        #sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=500, max_features=1),
        sklearn.neural_network.MLPClassifier(alpha=1.0, hidden_layer_sizes=(300,), max_iter=500)
    ]
    for clf in classifiers:
        clf.fit(td[0], td[1])
        print "\n{}: {}".format(type(clf).__name__, round(clf.score(vd[0], vd[1])*100, 2))

if __name__ == "__main__":
    all_transfers()
    





