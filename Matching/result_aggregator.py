import pandas as pd

def aggregate_results(matched_df):
    
    if matched_df.empty:
        print("No results to aggregate.")
        return pd.DataFrame()

    summary = matched_df.groupby(["category", "sub_category"]).agg({
        "keyword": "count",
        "similarity": ["mean", "min", "max"]
    }).reset_index()

    summary.columns = ["category", "sub_category", "matched_keywords_count", "avg_similarity", "min_similarity", "max_similarity"]
    summary = summary.sort_values(by=["category", "sub_category"])

    return summary
