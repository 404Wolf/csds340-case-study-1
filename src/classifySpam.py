import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def get_l1_preserved_features(trainFeatures, trainLabels, C):
    """Return indices of non-zero coefficients after L1-regularized logistic regression."""
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",  # supports L1
        C=C,
        max_iter=1000,
        class_weight="balanced",
        random_state=40,
    )
    model.fit(trainFeatures, trainLabels)
    coef = model.coef_.ravel()
    idx = np.where(coef != 0)[0]  # indices of preserved (non-zero) features
    return idx.tolist()


def getModel(trainFeatures, trainLabels, testFeatures, C=50):
    imputer = SimpleImputer(missing_values=-1, strategy="median")
    train_imputed = imputer.fit_transform(trainFeatures)
    test_imputed = imputer.transform(testFeatures)

    preserved_idx = get_l1_preserved_features(train_imputed, trainLabels, C=C)
    n_total = train_imputed.shape[1]
    n_kept = len(preserved_idx)

    print(f"Features kept: {n_kept}/{n_total}")
    print(f"Indices kept: {preserved_idx}")

    train_selected = train_imputed[:, preserved_idx]
    test_selected = test_imputed[:, preserved_idx]

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        criterion="entropy",
        class_weight="balanced",
        bootstrap=True,
        n_jobs=-1,
    )
    model.fit(train_selected, trainLabels)

    return model, test_selected


def predictTest(trainFeatures, trainLabels, testFeatures, C=10):
    model, test_selected = getModel(trainFeatures, trainLabels, testFeatures, C=C)

    return model.predict_proba(test_selected)[:, 1]
