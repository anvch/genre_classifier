#!/usr/bin/env python3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from data_utils import load_and_prepare_data, load_full_data

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)



def plot_roc_curves_multiclass(y_true, proba_dict, class_names, out_html="roc_curve.html"):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig = go.Figure()

    for model_name, y_proba in proba_dict.items():
        for i, cname in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            try:
                auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
                label = f"{model_name} – {cname} (AUC={auc_i:.3f})"
            except ValueError:
                label = f"{model_name} – {cname} (AUC=n/a)"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=label))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Chance", line=dict(dash="dash")))
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend_title_text=None,
        width=1000, height=650
    )
    fig.write_html(out_html, include_plotlyjs="cdn")

def plot_pr_curves_multiclass(y_true, proba_dict, class_names, out_html="pr_curve.html"):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig = go.Figure()

    for model_name, y_proba in proba_dict.items():
        for i, cname in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            try:
                ap_i = average_precision_score(y_bin[:, i], y_proba[:, i])
                label = f"{model_name} - {cname} (AP={ap_i:.3f})"
            except ValueError:
                label = f"{model_name} - {cname} (AP=n/a)"
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=label))

    fig.update_layout(
        title="Precision-Recall Curve Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        legend_title_text=None,
        width=1000, height=650
    )
    fig.write_html(out_html, include_plotlyjs="cdn")

def print_report(name, y_true, y_pred, y_proba, class_names):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    try:
        auc_macro_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro_ovr = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print(f"{name}")
    print("Classes:", class_names)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"Macro ROC AUC: {auc_macro_ovr if not np.isnan(auc_macro_ovr) else 'n/a'}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))



def _align_proba_columns(y_proba, model_classes, n_classes):
    if y_proba is None:
        return None
    aligned = np.zeros((y_proba.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(model_classes):
        if 0 <= cls < n_classes:
            aligned[:, cls] = y_proba[:, j]
    return aligned


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Encode labels to 0..K-1 and keep class names
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    # Define models (add more later)
    models = {
        "DummyMostFreq": DummyClassifier(strategy="most_frequent"),
        # "KNN": KNNClassifier(),
        # "RandomForest": RFClassifier(),
        # "SVM": SVMClassifier(probability=True),
    }

    predictions = {}
    probabilities = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        # Align columns to encoded class order [0..K-1]
        if y_proba is not None and hasattr(model, "classes_"):
            y_proba = _align_proba_columns(y_proba, model.classes_, n_classes)
        probabilities[name] = y_proba

    print("Classifier Comparison (label = region_group)")
    for name in models.keys():
        print_report(name, y_test_enc, predictions[name], probabilities[name], class_names)

    # Only include models that produced probabilities
    proba_models = {k: v for k, v in probabilities.items() if v is not None}
    if len(proba_models) > 0:
        plot_roc_curves_multiclass(y_test_enc, proba_models, class_names, out_html="roc_curve.html")
        plot_pr_curves_multiclass(y_test_enc, proba_models, class_names, out_html="pr_curve.html")
        print("\nSaved interactive plots: roc_curve.html, pr_curve.html")


if __name__ == "__main__":
    main()
