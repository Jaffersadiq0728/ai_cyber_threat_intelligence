"""
preprocessing/cleaner.py
------------------------
Loads, cleans and feature-engineers the CIC-IDS-2017 dataset.

Handles:
 - Infinite / NaN values
 - Whitespace in column names
 - Label normalisation
 - Feature scaling
 - Train / test split
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Features to drop (non-numeric or identifier columns)
DROP_COLS = ["Flow ID", "Source IP", "Source Port",
             "Destination IP", "Destination Port", "Timestamp"]

# Numeric feature columns (all except identifiers and label)
FEATURE_COLS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def _find_csv_files(data_dir):
    """Return all CSV files under data_dir."""
    patterns = [
        os.path.join(data_dir, "*.csv"),
        os.path.join(data_dir, "**", "*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return list(set(files))


def load_dataset(data_dir):
    """
    Load one or more CIC-IDS-2017 CSV files from data_dir.
    Returns raw concatenated DataFrame.
    """
    csv_files = _find_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in csv_files:
        print(f"  Loading: {os.path.basename(f)}")
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    raw = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows loaded: {len(raw):,}")
    return raw


def clean(df):
    """
    Cleans the raw CIC-IDS-2017 DataFrame:
      1. Strip whitespace from column names
      2. Drop identifier columns
      3. Replace inf values with NaN then fill
      4. Drop all-NaN columns
      5. Normalise label column name
    """
    # 1. Strip column name whitespace (common CICFlowMeter artefact)
    df.columns = df.columns.str.strip()

    # 2. Normalise label column — may be named 'Label' or ' Label'
    label_candidates = [c for c in df.columns if c.lower().strip() == "label"]
    if not label_candidates:
        raise KeyError("No 'Label' column found in dataset.")
    df = df.rename(columns={label_candidates[0]: "Label"})

    # 3. Drop identifier columns that exist
    existing_drops = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drops)

    # 4. Replace inf/-inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 5. Fill NaN with column median (robust to outliers)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 6. Clip extreme values (1st–99th percentile) to reduce noise
    for col in num_cols:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        if hi > lo:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # 7. Normalise labels to title-case
    df["Label"] = df["Label"].str.strip()

    # Keep a copy with identifiers BEFORE dropping them (for NLP enrichment)
    id_cols = [c for c in ["Source IP", "Destination IP",
                            "Source Port", "Destination Port",
                            "Timestamp", "Protocol"] if c in df.columns]
    df._meta_cols = df[id_cols].copy() if id_cols else None

    print(f"  After cleaning: {len(df):,} rows, {df['Label'].nunique()} classes")
    return df


def get_feature_matrix(df):
    """
    Return X (feature matrix), y (encoded labels),
    feature names list, label encoder, scaler.
    """
    # Use only columns that are present AND numeric
    available = [c for c in FEATURE_COLS if c in df.columns]
    if not available:
        # Fallback: use all numeric except Label
        available = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c != "Label"]

    X = df[available].values.astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(df["Label"].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, available, le, scaler


def split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def preprocess_pipeline(data_dir):
    """
    Full preprocessing pipeline:
      load → clean → feature matrix → split
    Returns dict with all artefacts needed by model training.
    """
    print("[PREPROCESSING] Loading dataset...")
    raw = load_dataset(data_dir)

    print("[PREPROCESSING] Cleaning...")
    clean_df = clean(raw)

    print("[PREPROCESSING] Building feature matrix...")
    X, y, feat_names, le, scaler = get_feature_matrix(clean_df)

    print("[PREPROCESSING] Splitting train/test...")
    X_train, X_test, y_train, y_test = split(X, y)

    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"  Classes: {list(le.classes_)}")

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_names": feat_names,
        "label_encoder": le,
        "scaler": scaler,
        "clean_df": clean_df,
    }
