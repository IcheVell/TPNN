import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def corr_heatmap(df, title, save_path):
    corr = df.corr(numeric_only=True, method="pearson")
    plt.figure(figsize=(9, 7))
    plt.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return corr

def train_test_split_np(X, y, test_size = 0.2, random_state = 0):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    test_n = int(round(n * test_size))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def standardize_transform(X, mu, sigma):
    return (X - mu) / sigma

def iqr_clip_df(df, cols, k=1.5):
    out = df.copy()
    for c in cols:
        q1 = out[c].quantile(0.25)
        q3 = out[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out

def drop_highly_correlated_features(df, threshold=0.85):
    corr = df.corr(numeric_only=True).abs()
    cols = list(corr.columns)
    to_drop = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] >= threshold:
                to_drop.add(cols[j])
    return df.drop(columns=list(to_drop)), sorted(to_drop)

def entropy(y):
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def split_info(sizes):
    sizes = np.asarray(sizes, dtype=float)
    total = sizes.sum()
    if total == 0:
        return 0.0
    w = sizes / total
    return -np.sum(w * np.log2(w + 1e-12))

def info_gain(parent_y, children_ys):
    H_parent = entropy(parent_y)
    n = len(parent_y)
    weighted = 0.0
    for cy in children_ys:
        if len(cy) == 0:
            continue
        weighted += (len(cy) / n) * entropy(cy)
    return H_parent - weighted

def gain_ratio(parent_y, children_ys):
    ig = info_gain(parent_y, children_ys)
    si = split_info([len(cy) for cy in children_ys])
    if si <= 1e-12:
        return 0.0
    return ig / si

def best_gain_ratio_for_continuous_feature(x_col, y, max_candidates=2000):
    x_col = np.asarray(x_col, dtype=float)
    y = np.asarray(y)

    order = np.argsort(x_col, kind="mergesort")
    xs = x_col[order]

    diffs = xs[1:] != xs[:-1]
    if not np.any(diffs):
        return 0.0, None

    mids = (xs[1:][diffs] + xs[:-1][diffs]) / 2.0

    if len(mids) > max_candidates:
        idxs = np.linspace(0, len(mids) - 1, max_candidates, dtype=int)
        mids = mids[idxs]

    best_gr = -1.0
    best_t = None
    for t in mids:
        left = y[x_col <= t]
        right = y[x_col > t]
        gr = gain_ratio(y, [left, right])
        if gr > best_gr:
            best_gr = gr
            best_t = t

    return float(best_gr), float(best_t)

def gain_ratio_for_categorical_feature(x_col, y):
    x_col = np.asarray(x_col)
    y = np.asarray(y)
    children = [y[x_col == v] for v in np.unique(x_col)]
    
    return float(gain_ratio(y, children))

def rank_features_by_gain_ratio(X, y, feature_names, categorical_unique_threshold=20):
    results = []

    for j, name in enumerate(feature_names):
        col = X[:, j]
        uniq = np.unique(col)
        if len(uniq) <= categorical_unique_threshold:
            gr = gain_ratio_for_categorical_feature(col, y)
            t = None
        else:
            gr, t = best_gain_ratio_for_continuous_feature(col, y)
        results.append((name, gr, t))
    results.sort(key=lambda z: z[1], reverse=True)
    return results

def load_and_merge_wines(red_path, white_path):
    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")
    red.columns = red.columns.str.strip()
    white.columns = white.columns.str.strip()
    red["color"] = 1
    white["color"] = 0
    df = pd.concat([red, white], ignore_index=True)
    df.columns = df.columns.str.strip()
    return df

def top_features_with_color(red_path="winequality-red.csv", white_path="winequality-white.csv", test_size=0.2, random_state=0, top_k=12, corr_threshold=0.85, heatmap_path="corr_merged_after.png"):
    df = load_and_merge_wines(red_path, white_path)

    y = df["quality"].astype(int).to_numpy()
    X_df = df.drop(columns=["quality"])

    for c in X_df.columns:
        if X_df[c].isna().any():
            X_df[c] = X_df[c].fillna(X_df[c].mean())

    df2 = pd.concat([X_df, pd.Series(y, name="quality")], axis=1).drop_duplicates()
    y = df2["quality"].to_numpy()
    X_df = df2.drop(columns=["quality"])

    X_df = iqr_clip_df(X_df, cols=X_df.columns.tolist(), k=1.5)

    X_df_reduced, dropped = drop_highly_correlated_features(X_df, threshold=corr_threshold)
    print(f"Удалены по корреляции (|corr| >= {corr_threshold}): {dropped}")

    corr_heatmap(X_df_reduced, "MERGED wine - Correlation heatmap", heatmap_path)

    X = X_df_reduced.to_numpy(dtype=float)
    feature_names = X_df_reduced.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=test_size, random_state=random_state)

    mu, sigma = standardize_fit(X_train)
    X_train_s = standardize_transform(X_train, mu, sigma)

    ranking = rank_features_by_gain_ratio(X_train_s, y_train, feature_names, categorical_unique_threshold=20)

    print(f"ТОП-{top_k} признаков по Gain Ratio:")
    for i, (fname, gr, thr) in enumerate(ranking[:top_k], start=1):
        if thr is None:
            print(f"{i:2d}) {fname:25s} GainRatio={gr:.6f}  split=categorical")
        else:
            print(f"{i:2d}) {fname:25s} GainRatio={gr:.6f}  best_threshold={thr:.6f}")

if __name__ == "__main__":
    top_features_with_color(top_k = 12)