import argparse
import os
import pandas as pd
import numpy as np


def preprocess_azure(
    input_path: str,
    output_path: str,
    max_tasks: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load, clean, and normalise Azure Functions dataset.

    Steps:
      1. Load raw CSV.
      2. Drop missing rows.
      3. Scale duration → CPU MI range [1000, 10000].
      4. Scale memory → MB range [100, 512].
      5. Assign priority (Azure serverless = mostly high-priority).
      6. Normalise timestamps → seconds relative to trace start.
      7. Assign tight deadlines (1-5s, reflecting serverless SLA).
      8. Save processed CSV.

    Args:
        input_path:  Path to raw Azure Functions CSV.
        output_path: Path to save processed CSV.
        max_tasks:   Maximum number of invocations to retain.
        seed:        RNG seed for reproducibility.

    Returns:
        Processed DataFrame.
    """
    rng = np.random.default_rng(seed)

    print(f"[Azure Functions] Loading {input_path} ...")

    df = pd.read_csv(input_path, nrows=max_tasks * 2)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    print(f"[Azure Functions] Raw rows: {len(df):,}")
    print(f"[Azure Functions] Columns:  {list(df.columns)}")

    # Drop rows with all-NaN
    df = df.dropna(how='all').reset_index(drop=True)


    time_col = None
    for candidate in ['trigger_time', 'invocation_time', 'timestamp', 'time']:
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        # Fallback: use first column
        time_col = df.columns[0]
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)


    dur_col = None
    for candidate in ['duration', 'execution_time', 'duration_ms', 'exec_time']:
        if candidate in df.columns:
            dur_col = candidate
            break

    mem_col = None
    for candidate in ['memory', 'mem', 'memory_mb', 'mem_usage']:
        if candidate in df.columns:
            mem_col = candidate
            break

    df = df.head(max_tasks).copy()
    print(f"[Azure Functions] Tasks after cleaning: {len(df):,}")

    if dur_col is not None:
        df[dur_col] = pd.to_numeric(df[dur_col], errors='coerce').fillna(1.0)
        dur_max = df[dur_col].quantile(0.99)
        df['cpu_mi'] = (
            df[dur_col].clip(upper=dur_max) / max(dur_max, 1e-9)
        ) * 9000 + 1000
    else:
        df['cpu_mi'] = rng.uniform(1000, 10000, size=len(df))


    if mem_col is not None:
        df[mem_col] = pd.to_numeric(df[mem_col], errors='coerce').fillna(100.0)
        mem_max = df[mem_col].quantile(0.99)
        df['mem_mb'] = (
            df[mem_col].clip(upper=mem_max) / max(mem_max, 1e-9)
        ) * 412 + 100
    else:
        df['mem_mb'] = rng.uniform(100, 512, size=len(df))

    df['pri_label'] = rng.choice(
        ['low', 'medium', 'high'],
        p=[0.10, 0.30, 0.60],
        size=len(df),
    )

    t0 = df[time_col].min()
    scale = 1000.0 if df[time_col].max() > 1e9 else 1.0  # ms → s if needed
    df['arrival_s'] = (df[time_col] - t0) / scale

    df['deadline_offset_s'] = rng.uniform(1.0, 5.0, size=len(df))
    df['deadline_s'] = df['arrival_s'] + df['deadline_offset_s']

    df['data_size_mb'] = df['cpu_mi'] * 0.001

    output_cols = [
        'arrival_s', 'cpu_mi', 'mem_mb', 'pri_label',
        'deadline_s', 'data_size_mb',
    ]
    if 'function_id' in df.columns:
        output_cols.append('function_id')

    out_df = df[output_cols].copy()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[Azure Functions] Saved {len(out_df):,} tasks → {output_path}")


    print("\n[Azure Functions] Summary:")
    print(f"  CPU MI:   mean={out_df['cpu_mi'].mean():.0f}, "
          f"std={out_df['cpu_mi'].std():.0f}")
    print(f"  Mem MB:   mean={out_df['mem_mb'].mean():.0f}, "
          f"std={out_df['mem_mb'].std():.0f}")
    print(f"  Priority: {out_df['pri_label'].value_counts().to_dict()}")
    print(f"  Duration: {out_df['arrival_s'].max():.1f}s")

    return out_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Azure Functions Dataset for DRL scheduler"
    )
    parser.add_argument("--input",  required=True,
                        help="Path to raw Azure Functions CSV")
    parser.add_argument("--output", default="data/processed/azure_processed.csv",
                        help="Output CSV path")
    parser.add_argument("--max_tasks", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocess_azure(
        input_path=args.input,
        output_path=args.output,
        max_tasks=args.max_tasks,
        seed=args.seed,
    )