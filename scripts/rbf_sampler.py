from sklearn.kernel_approximation import RBFSampler

def rbf_sampler(*X_list, gamma, n_components, random_state):
    rbf_feature = RBFSampler(gamma=gamma, n_components = n_components, random_state=random_state)
    rbf_sampler_list = []
    for X in X_list:
        rbf_sampler_list.append(rbf_feature.fit_transform(X))

    return rbf_sampler_list
