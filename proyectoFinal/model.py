# Aprendizaje Automático: Proyecto Final
# Clasificación de símbolos Devanagari
# Patricia Córdoba Hidalgo
# David Cabezas Berrido

# model.py
# Funciones para probar la eficacia de un modelo

from sklearn.metrics import accuracy_score, confusion_matrix

def modelAccuracy(estimator, train, train_label, test, test_label):
    estimator.fit(train,train_label)
    test_acc=estimator.score(test,test_label)
    train_acc=estimator.score(train,train_label)
    return test_acc, train_acc


def modelPerformance(estimator, train, train_label, test, test_label):
    estimator.fit(train, train_label)
    pred = estimator.predict(test)
    acc = accuracy_score(test_label, pred)
    conf_mat=confusion_matrix(test_label, pred, normalice='all')
    return acc, conf_mat

