from sklearn.model_selection import KFold
import numpy as np


def kfold_accuracy(create_model, X, y, epochs=15, k=10):
    kfold = KFold(n_splits=k, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []
    for i, (train, test) in enumerate(kfold.split(X, y), start=1):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {i} ...')

        model = create_model()
        _ = model.fit(X.iloc[train], y.iloc[train], epochs=epochs)

        scores = model.evaluate(X.iloc[test], y.iloc[test], verbose=0)
        print(
            f'Score for fold {i}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

    return np.mean(acc_per_fold)
