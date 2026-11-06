from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.config import THRESHOLD


def evaluate_probabilities(probabilities, y_test):
    """Score churn probabilities (one number from 0 to 1 per customer).

    Works for any model, including the Day 5 neural network, because all it
    needs is the list of probabilities. For churn we care most about RECALL:
    out of all customers who really left, how many did we catch?
    """
    # turn the probability into a yes/no guess: 1 = will churn, 0 = will stay
    guesses = (probabilities >= THRESHOLD).astype(int)

    stay_right, stay_wrong, churn_missed, churn_caught = confusion_matrix(y_test, guesses).ravel()

    return {
        "accuracy": round(float(accuracy_score(y_test, guesses)), 4),
        "precision": round(float(precision_score(y_test, guesses)), 4),
        "recall": round(float(recall_score(y_test, guesses)), 4),
        "f1": round(float(f1_score(y_test, guesses)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "decision_threshold": THRESHOLD,
        "confusion_matrix": {
            "true_negative": int(stay_right),
            "false_positive": int(stay_wrong),
            "false_negative": int(churn_missed),
            "true_positive": int(churn_caught),
        },
    }


def evaluate(pipeline, X_test, y_test):
    """Test a trained scikit-learn pipeline on customers it has never seen."""
    # churn probability for every test customer (a number from 0 to 1)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    return evaluate_probabilities(probabilities, y_test)
