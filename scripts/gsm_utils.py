def normalize_gsm(gsm, min_gsm, max_gsm):
    """
    Convert GSM to normalized density score (0–1)
    """
    if max_gsm == min_gsm:
        return 0.5
    value = (gsm - min_gsm) / (max_gsm - min_gsm)
    return max(0.0, min(1.0, value))


def denormalize_gsm(density, min_gsm, max_gsm):
    """
    Convert normalized density (0–1) back to GSM
    """
    density = max(0.0, min(1.0, density))
    return min_gsm + density * (max_gsm - min_gsm)
