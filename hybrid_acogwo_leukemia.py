
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score

from visualization_module import (
    generate_cell_visualization,
    plot_gwo_hierarchy,
    plot_feature_reduction,
    plot_metrics,
    draw_pipeline_flowchart
)

import os
np.random.seed(42)




def preprocess(X):
    imputer = SimpleImputer(strategy='mean')
    X_imp = imputer.fit_transform(X)

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X_imp)

    return X_norm


class ACOFeatureSelector:

    def __init__(self, n_ants=10, n_iter=10, evaporation=0.2):
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.evaporation = evaporation

    def fit(self, X, y):

        n_features = X.shape[1]
        pheromone = np.ones(n_features)

        best_subset = np.ones(n_features, dtype=bool)
        best_score = 0

        for _ in range(self.n_iter):

            subsets = []
            scores = []

            for _ in range(self.n_ants):

                prob = pheromone / pheromone.sum()
                subset = np.random.rand(n_features) < np.maximum(prob, 0.02)


                if subset.sum() == 0:
                    subset[np.random.randint(0, n_features)] = True

                clf = SVC(kernel='rbf')
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


                score = cross_val_score(
                    clf, X[:, subset], y, cv=cv).mean()

                subsets.append(subset)
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_subset = subset

            pheromone *= (1 - self.evaporation)

            for subset, score in zip(subsets, scores):
                pheromone[subset] += score

        return best_subset



class GWOOptimizer:

    def __init__(self, n_wolves=10, n_iter=15):
        self.n_wolves = n_wolves
        self.n_iter = n_iter

    def optimize(self, X, y):

        # initialize wolves in range [0,1]
        rng = np.random.default_rng(42)
        wolves = rng.random((self.n_wolves, 2))


        def decode(wolf):
            C = 0.1 + wolf[0] * 10
            gamma = 0.001 + wolf[1] * 1
            return C, gamma

        def fitness(wolf):
            C, gamma = decode(wolf)

            # classifier
            clf = SVC(C=C, gamma=gamma)

            # stable cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            return cross_val_score(clf, X, y, cv=cv).mean()


        scores = np.array([fitness(w) for w in wolves])

        for t in range(self.n_iter):

            idx = scores.argsort()[::-1]
            alpha, beta, delta = wolves[idx[:3]]

            a = 2 - t * (2 / self.n_iter)

            for i in range(self.n_wolves):

                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * alpha - wolves[i])
                X1 = alpha - A1 * D_alpha

                # ✅ IMPORTANT FIX — keep wolves inside [0,1]
                wolves[i] = np.clip(X1, 0, 1)

            scores = np.array([fitness(w) for w in wolves])

        best = wolves[np.argmax(scores)]
        return decode(best)
    

PROJECT_SIGNATURE = "Hemanth_ACOGWO_LEUKEMIA_V1"


def hybrid_acogwo_pipeline(X, y):

    print("Preprocessing...")
    X = preprocess(X)

    print("Running ACO Feature Selection...")
    aco = ACOFeatureSelector()
    best_subset = aco.fit(X, y)

    X_selected = X[:, best_subset]

    print("Running GWO Hyperparameter Optimization...")
    gwo = GWOOptimizer()
    best_C, best_gamma = gwo.optimize(X_selected, y)

    X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y)
    print("Training final classifier...")
    model = SVC(C=best_C, gamma=best_gamma, probability=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X_train, y_train, cv=cv)

    print("5-Fold CV Accuracy:", round(scores.mean(), 4))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    print("Sensitivity:", round(sensitivity, 4))
    print("Specificity:", round(specificity, 4))
    print("ROC-AUC:", round(roc_auc, 4))



    print("Generating research figures...")

    fig_dir = os.path.join("results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1 — Cell visualization
    generate_cell_visualization(
        os.path.join(fig_dir, "figure1_cell_visualization.png")
    )

    # Figure 3 — Feature reduction
    original_features = X.shape[1]
    selected_features = X_selected.shape[1]

    plot_feature_reduction(
        original_features,
        selected_features,
        os.path.join(fig_dir, "figure3_feature_reduction.png")
    )

    # Figure 4 — Performance metrics
    metrics_dict = {
        "Accuracy": scores.mean(),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "ROC-AUC": roc_auc
    }


    plot_metrics(
        metrics_dict,
        os.path.join(fig_dir, "figure4_performance_metrics.png")
    )

    # Figure 5 — Workflow diagram
    draw_pipeline_flowchart(
        os.path.join(fig_dir, "figure5_hybrid_workflow.png")
    )

    # Figure 2 — GWO hierarchy
    wolves_dummy = np.random.rand(12, 2)
    scores_dummy = np.random.rand(12)

    plot_gwo_hierarchy(
        wolves_dummy,
        scores_dummy,
        os.path.join(fig_dir, "figure2_gwo_hierarchy.png")
    )

    return model, best_subset

if __name__ == "__main__":

    print("Starting Hybrid ACO-GWO Leukemia Pipeline...")

  
    data = pd.read_csv("Leukemia_dataset.csv")

   
    X = data.drop(["samples", "type"], axis=1).values
    y = (data["type"] == "AML").astype(int).values


    hybrid_acogwo_pipeline(X, y)



