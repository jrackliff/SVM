from sklearn.svm import SVC


def classify(features_train, labels_train):

    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier


    ### your code goes here!
    clf = SVC(gamma=1000.0)
    clf.fit(features_train, labels_train)
    clf.predict(features_train)

    print clf.score(features_train, labels_train)
    return clf



