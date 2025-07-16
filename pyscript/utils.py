import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def get_reduced_correlated_features(df, threshold=0.8):
    """
    Identifies correlated feature pairs and selects a reduced set of features
    by keeping only one feature from each highly correlated group.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): Correlation coefficient threshold (default: 0.8).

    Returns:
        kept_features (List[str]): Features to keep (non-redundant).
        dropped_features (List[str]): Features identified as redundant and dropped.
        correlated_pairs (List[Tuple[str, str, float]]): Correlated feature pairs above threshold.
    """
    corr_matrix = df.corr(numeric_only=True)
    correlated_pairs = []
    to_drop = set()
    already_in_group = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                correlated_pairs.append((col1, col2, corr_value))
                # Drop col2 if col1 hasn't already been marked to drop
                if col1 not in to_drop and col2 not in already_in_group:
                    to_drop.add(col2)
                    already_in_group.add(col1)
                    already_in_group.add(col2)

    all_features = set(df.select_dtypes(include="number").columns)
    kept_features = sorted(list(all_features - to_drop))
    dropped_features = sorted(list(to_drop))

    return kept_features, dropped_features, correlated_pairs


def calculate_vif(df):
    df = df.select_dtypes(include="number")
    X = add_constant(df)
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i + 1) for i in range(len(df.columns))
    ]
    return vif_data.sort_values(by="VIF", ascending=False)
