"Predict one delivery, from the command line."
from pathlib import Path
import pandas as pd

from delivery import load_model


def main():
    model_path = Path(__file__).parent / "model.joblib"
    model = load_model(model_path)
    order = pd.DataFrame([{
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0
    }])
    minutes = model.predict(order)[0]
    print(f"PREDICTION: {minutes:.1f}")
    #       traffic 3 / no rain order, and print "PREDICTION: <minutes>"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
