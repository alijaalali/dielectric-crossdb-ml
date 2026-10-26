# Tests the constant-offset correction with a held-out calibration split: the offset
# is estimated on one half of the JARVIS test set and applied to the other half, so
# the evaluation is not contaminated by the same data used to fit the correction.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
OUT = PROJ / "models" / "crossdb"
RNG = 42


def main():
    pred = pd.read_csv(OUT / "predictions_mp_to_jarvis_etotal.csv")
    y_true = pred["true_log10"].to_numpy(dtype=float)
    y_pred = pred["pred_log10"].to_numpy(dtype=float)
    n = len(y_true)

    rng = np.random.RandomState(RNG)
    idx = rng.permutation(n)
    half = n // 2
    cal_idx, hold_idx = idx[:half], idx[half:]

    offset_cal = float(np.mean(y_pred[cal_idx] - y_true[cal_idx]))
    offset_cal_median = float(np.median(y_pred[cal_idx] - y_true[cal_idx]))

    mae_hold_uncorrected = mean_absolute_error(y_true[hold_idx], y_pred[hold_idx])
    rmse_hold_uncorrected = np.sqrt(mean_squared_error(y_true[hold_idx], y_pred[hold_idx]))

    y_pred_corrected_mean = y_pred[hold_idx] - offset_cal
    mae_hold_corrected_mean = mean_absolute_error(y_true[hold_idx], y_pred_corrected_mean)
    rmse_hold_corrected_mean = np.sqrt(mean_squared_error(y_true[hold_idx], y_pred_corrected_mean))

    y_pred_corrected_median = y_pred[hold_idx] - offset_cal_median
    mae_hold_corrected_median = mean_absolute_error(y_true[hold_idx], y_pred_corrected_median)
    rmse_hold_corrected_median = np.sqrt(mean_squared_error(y_true[hold_idx], y_pred_corrected_median))

    mult_factor = 10 ** (-offset_cal)

    out = {
        "n_total": n, "n_calibration": len(cal_idx), "n_heldout": len(hold_idx),
        "offset_mean_log10_estimated_on_calibration_half": offset_cal,
        "offset_median_log10_estimated_on_calibration_half": offset_cal_median,
        "raw_unit_multiplicative_factor_from_mean_offset": mult_factor,
        "heldout_uncorrected": {"mae": float(mae_hold_uncorrected), "rmse": float(rmse_hold_uncorrected)},
        "heldout_corrected_mean_offset": {"mae": float(mae_hold_corrected_mean), "rmse": float(rmse_hold_corrected_mean)},
        "heldout_corrected_median_offset": {"mae": float(mae_hold_corrected_median), "rmse": float(rmse_hold_corrected_median)},
    }
    with open(OUT / "offset_calibration_split.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
