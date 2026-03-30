import argparse
import os
import pandas as pd
import numpy as np


# Columns we need from the raw Google trace
RAW_COLS = {
    'time': 0,
    'job_id': 2,
    'event_type': 5,
    'scheduling_class': 7,
    'priority': 8,
    'cpu_request': 9,
    'mem_request': 10,
}

# event_type == 0 means SUBMIT
SUBMIT_EVENT = 0


def preprocess_google_traces(
    input_path: str,
    output_path: str,
    max_tasks: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    print(f"[Google Traces] Loading {input_path} ...")

    # Try to detect if file has a header row
    sample = pd.read_csv(input_path, nrows=3, header=None)
    has_header = not pd.to_numeric(sample.iloc[0], errors='coerce').notna().any()

    df = pd.read_csv(
        input_path,
        header=0 if has_header else None,
        nrows=max_tasks * 3,  # over-sample before filtering
    )
    df.columns = [str(c).strip().lower() for c in df.columns]

    # If raw format (no header), assign column names by index
    if not has_header:
        col_map = {v: k for k, v in RAW_COLS.items()}
        rename = {df.columns[i]: col_map[i] for i in col_map if i < len(df.columns)}
        df = df.rename(columns=rename)

    print(f"[Google Traces] Raw rows: {len(df):,}")

    # Keep only SUBMIT events if event_type column exists
    if 'event_type' in df.columns:
        df = df[df['event_type'] == SUBMIT_EVENT].copy()
        print(f"[Google Traces] After SUBMIT filter: {len(df):,}")

    # Required columns with fallbacks
    for col in ['time', 'cpu_request', 'mem_request']:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Columns: {list(df.columns)}")

    # Drop missing values in critical columns
    df = df.dropna(subset=['time', 'cpu_request', 'mem_request'])
    df = df[df['cpu_request'] > 0].copy()

    # Sort by arrival time
    df = df.sort_values('time').reset_index(drop=True)

    # Cap to max_tasks
    df = df.head(max_tasks).copy()
    print(f"[Google Traces] Tasks after cleaning: {len(df):,}")

    # ----------------------------------------------------------------
    # Normalise CPU → MI range [1000, 10000]
    # ----------------------------------------------------------------
    cpu_max = df['cpu_request'].quantile(0.99)   # clip outliers
    df['cpu_mi'] = (
        df['cpu_request'].clip(upper=cpu_max) / max(cpu_max, 1e-9)
    ) * 9000 + 1000

    # ----------------------------------------------------------------
    # Normalise memory → MB range [100, 512]
    # ----------------------------------------------------------------
    mem_max = df['mem_request'].quantile(0.99)
    df['mem_mb'] = (
        df['mem_request'].clip(upper=mem_max) / max(mem_max, 1e-9)
    ) * 412 + 100

    # ----------------------------------------------------------------
    # Priority mapping: Google [0-11] → {low, medium, high}
    # ----------------------------------------------------------------
    if 'priority' in df.columns:
        df['priority'] = pd.to_numeric(df['priority'], errors='coerce').fillna(0)
        df['pri_label'] = pd.cut(
            df['priority'],
            bins=[-1, 3, 7, 11],
            labels=['low', 'medium', 'high'],
        ).astype(str)
    else:
        df['pri_label'] = rng.choice(['low', 'medium', 'high'], size=len(df))

    # ----------------------------------------------------------------
    # Arrival time: microseconds → seconds (relative to trace start)
    # ----------------------------------------------------------------
    t0 = df['time'].min()
    df['arrival_s'] = (df['time'] - t0) / 1_000_000.0

    # ----------------------------------------------------------------
    # Synthetic deadline: arrival + U[1, 10] seconds
    # ----------------------------------------------------------------
    df['deadline_offset_s'] = rng.uniform(1, 10, size=len(df))
    df['deadline_s'] = df['arrival_s'] + df['deadline_offset_s']

    # ----------------------------------------------------------------
    # Data size for communication time (proportional to CPU demand)
    # ----------------------------------------------------------------
    df['data_size_mb'] = df['cpu_mi'] * 0.001

    # ----------------------------------------------------------------
    # Select and save
    # ----------------------------------------------------------------
    output_cols = [
        'arrival_s', 'cpu_mi', 'mem_mb', 'pri_label',
        'deadline_s', 'data_size_mb',
    ]
    if 'scheduling_class' in df.columns:
        output_cols.append('scheduling_class')
    if 'job_id' in df.columns:
        output_cols.append('job_id')

    out_df = df[output_cols].copy()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[Google Traces] Saved {len(out_df):,} tasks → {output_path}")

    # ----------------------------------------------------------------
    # Summary statistics
    # ----------------------------------------------------------------
    print("\n[Google Traces] Summary:")
    print(f"  CPU MI:   mean={out_df['cpu_mi'].mean():.0f}, "
          f"std={out_df['cpu_mi'].std():.0f}")
    print(f"  Mem MB:   mean={out_df['mem_mb'].mean():.0f}, "
          f"std={out_df['mem_mb'].std():.0f}")
    print(f"  Priority: {out_df['pri_label'].value_counts().to_dict()}")
    print(f"  Duration: {out_df['arrival_s'].max():.1f}s")

    return out_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Google Cluster Traces for DRL scheduler"
    )
    parser.add_argument("--input",  required=True,
                        help="Path to raw Google Cluster CSV")
    parser.add_argument("--output", default="data/processed/google_processed.csv",
                        help="Output CSV path")
    parser.add_argument("--max_tasks", type=int, default=100_000,
                        help="Maximum tasks to retain (default: 100000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocess_google_traces(
        input_path=args.input,
        output_path=args.output,
        max_tasks=args.max_tasks,
        seed=args.seed,
    )