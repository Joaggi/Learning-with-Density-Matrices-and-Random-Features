from sklearn.svm import LinearSVC



def linear_svc(X_train_features, y_train, C=2**1, tol=1e-05):
    model_lsvc = LinearSVC(C=2**1,tol=1e-05)
    model_lsvc.fit(X_train_features, y_train)
    return model_lsvc

