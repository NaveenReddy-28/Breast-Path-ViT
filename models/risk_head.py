# models/risk_head.py

def risk_from_prob(cancer_prob: float) -> str:
    """
    Convert cancer probability (0.0 to 1.0) into a risk label.
    This is a simple rule-based version for now.
    """
    if cancer_prob < 0.33:
        return "Low risk"
    elif cancer_prob < 0.66:
        return "Medium risk"
    else:
        return "High risk"
