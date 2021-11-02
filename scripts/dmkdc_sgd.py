import qmc.tf.layers as layers
import qmc.tf.models as models
import keras
import tensorflow as tf


def create_model(X_train, y_train, input_dim, num_classes, component_dim=100, gamma=1, lr=0.01, decay=0.,
                  random_state=None, eig_percentage=0, initialize_with_rff=False,
                  type_of_rff="rff", fix_rff=False, batch_size=32):
    '''This is a model generating function so that we can search over neural net
    parameters and architecture'''

    num_eig = round(eig_percentage * component_dim)

    opt = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=decay)

    if type_of_rff == 'rff':
        fm_x = layers.QFeatureMapRFF(input_dim, dim=component_dim, gamma=gamma, random_state=random_state)
    else:
        fm_x = layers.QFeatureMapORF(input_dim, dim=component_dim, gamma=gamma, random_state=random_state)

    if initialize_with_rff:
        qmkdc = models.DMKDClassifier(fm_x=fm_x, dim_x=component_dim, num_classes=num_classes)
        qmkdc.compile()
        qmkdc.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)

    qmkdc1 = models.DMKDClassifierSGD(input_dim=input_dim, dim_x=component_dim, num_eig=num_eig,
                                      num_classes=num_classes, gamma=gamma, random_state=random_state)

    if fix_rff:
        qmkdc1.layers[0].trainable = False

    qmkdc1.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    if initialize_with_rff:
        qmkdc1.set_rhos(qmkdc.get_rhos())

    # qmkdc1.fit(X_train, y_train_bin, epochs=epochs, batch_size=batch_size)

    return qmkdc1
