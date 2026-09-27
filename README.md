import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

def generate_synthetic_radar_data(samples=1000, contamination=0.05):
    """Generates synthetic EW Radar telemetry signals with simulated anomalies."""
    np.random.seed(42)
    
    # Nominal Radar Telemetry Data Range
    frequency = np.random.normal(loc=3000, scale=150, size=samples)      # MHz
    pulse_width = np.random.normal(loc=1.5, scale=0.2, size=samples)     # us
    pri = np.random.normal(loc=1000, scale=50, size=samples)             # us
    amplitude = np.random.normal(loc=-40, scale=5, size=samples)         # dBm

    data = pd.DataFrame({
        'frequency': frequency,
        'pulse_width': pulse_width,
        'pri': pri,
        'amplitude': amplitude
    })

    # Injecting synthetic anomalies
    num_anomalies = int(samples * contamination)
    anomaly_indices = np.random.choice(samples, size=num_anomalies, replace=False)

    data.loc[anomaly_indices, 'frequency'] += np.random.choice([1500, -1500], size=num_anomalies)
    data.loc[anomaly_indices, 'amplitude'] += np.random.choice([30, -30], size=num_anomalies)

    return data

def train_and_save_pipeline():
    print("Generating radar telemetry dataset...")
    df = generate_synthetic_radar_data()

    print("Preprocessing telemetry features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    print("Training Isolation Forest Anomaly Detection Model...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(scaled_features)

    # Exporting model artifacts
    print("Saving pipeline artifacts (.pkl)...")
    joblib.dump(model, "radar_anomaly_model.pkl")
    joblib.dump(scaler, "radar_scaler.pkl")

    print("Pipeline training completed successfully!")

if __name__ == "__main__":
    train_and_save_pipeline()
