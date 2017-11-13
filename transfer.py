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

def all_baselines():
    for n in [1, 5, 10, 0]:
        print "\n\nUsing RMNIST/"+str(n)
        baselines(n)
        
def baselines(n):
    td, vd, ts = data_loader.load_data(n, abstract=True)
    classifiers = [
        sklearn.svm.SVC(),
        sklearn.svm.SVC(kernel="linear", C=0.1),
        sklearn.neighbors.KNeighborsClassifier(1),
        sklearn.tree.DecisionTreeClassifier(),
        sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=500, max_features=1),
        sklearn.neural_network.MLPClassifier(alpha=1, hidden_layer_sizes=(200,))
    ]
    for clf in classifiers:
        clf.fit(td[0], td[1])
        print "\n\n"+str(clf)
        print round(clf.score(vd[0], vd[1])*100, 2)

if __name__ == "__main__":
    all_baselines()
    

