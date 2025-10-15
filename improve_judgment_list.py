#!/usr/bin/env python3
"""
Script to improve the judgment list CSV for better LTR model performance.

Based on analysis findings from ANALYSIS_SUMMARY.md, this script:
1. Fixes CTR calculation inconsistencies
2. Applies consistent relevance grade binning
3. Removes keywords with no relevant results
4. Removes keywords with too few products (<5)
5. Removes duplicate keyword-product pairs
6. Balances relevance grade distribution
7. Handles zero-click products with high impressions
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data(input_path: str) -> pd.DataFrame:
    """Load the judgment list CSV."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Initial dataset: {len(df):,} records")
    print(f"Columns: {df.columns.tolist()}")
    return df

def fix_ctr_calculation(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate CTR consistently."""
    print("\n1. Fixing CTR calculations...")

    # Recalculate CTR properly
    df['calculated_ctr'] = (df['total_clicks'] / df['total_impressions'] * 100).round(2)

    # Count mismatches
    df['ctr_mismatch'] = ~np.isclose(df['ctr'], df['calculated_ctr'], atol=0.01)
    mismatches = df['ctr_mismatch'].sum()
    print(f"   - Found {mismatches:,} CTR mismatches ({mismatches/len(df)*100:.2f}%)")

    # Replace with calculated CTR
    df['ctr'] = df['calculated_ctr']
    df = df.drop(columns=['calculated_ctr', 'ctr_mismatch'])

    return df

def apply_consistent_relevance_grading(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent CTR-based relevance grading."""
    print("\n2. Applying consistent relevance grading...")

    # Store original grades for comparison
    df['original_grade'] = df['relevance_grade']

    # Apply consistent binning:
    # 0: CTR < 1%
    # 1: 1% <= CTR < 3%
    # 2: 3% <= CTR < 6%
    # 3: 6% <= CTR < 20%
    # 4: CTR >= 20%
    df['relevance_grade'] = pd.cut(
        df['ctr'],
        bins=[-np.inf, 1, 3, 6, 20, np.inf],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(int)

    # Count changes
    grade_changes = (df['original_grade'] != df['relevance_grade']).sum()
    print(f"   - Changed {grade_changes:,} relevance grades ({grade_changes/len(df)*100:.2f}%)")

    # Show new distribution
    print("\n   New relevance grade distribution:")
    grade_dist = df['relevance_grade'].value_counts(normalize=True).sort_index() * 100
    for grade, pct in grade_dist.items():
        print(f"   Grade {grade}: {pct:5.2f}%")

    df = df.drop(columns=['original_grade'])
    return df

def handle_zero_click_products(df: pd.DataFrame, min_impressions: int = 50) -> pd.DataFrame:
    """Mark products with many impressions but zero clicks as irrelevant."""
    print(f"\n3. Handling zero-click products (>={min_impressions} impressions)...")

    zero_click_mask = (df['total_clicks'] == 0) & (df['total_impressions'] >= min_impressions)
    zero_click_count = zero_click_mask.sum()

    print(f"   - Found {zero_click_count:,} zero-click products with >={min_impressions} impressions")

    # These should definitely be grade 0
    df.loc[zero_click_mask, 'relevance_grade'] = 0

    return df

def remove_keywords_with_no_relevant_results(df: pd.DataFrame) -> pd.DataFrame:
    """Remove keywords where ALL products are irrelevant (grade 0)."""
    print("\n4. Removing keywords with no relevant results...")

    # Find keywords with at least one relevant product (grade > 0)
    keyword_has_relevant = df.groupby('keyword')['relevance_grade'].apply(lambda x: (x > 0).any())
    valid_keywords = keyword_has_relevant[keyword_has_relevant].index

    initial_count = len(df)
    df = df[df['keyword'].isin(valid_keywords)].copy()
    removed = initial_count - len(df)

    print(f"   - Removed {removed:,} records from {len(keyword_has_relevant) - len(valid_keywords)} keywords")
    print(f"   - Remaining keywords: {len(valid_keywords):,}")

    return df

def remove_keywords_with_few_products(df: pd.DataFrame, min_products: int = 5) -> pd.DataFrame:
    """Remove keywords with too few products to learn meaningful ranking."""
    print(f"\n5. Removing keywords with <{min_products} products...")

    keyword_product_count = df.groupby('keyword').size()
    valid_keywords = keyword_product_count[keyword_product_count >= min_products].index

    initial_count = len(df)
    df = df[df['keyword'].isin(valid_keywords)].copy()
    removed = initial_count - len(df)

    print(f"   - Removed {removed:,} records from {len(keyword_product_count) - len(valid_keywords)} keywords")
    print(f"   - Remaining keywords: {len(valid_keywords):,}")

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate keyword-product pairs."""
    print("\n6. Removing duplicate keyword-product pairs...")

    initial_count = len(df)

    # Keep the row with highest total_clicks (most activity)
    df = df.sort_values('total_clicks', ascending=False)
    df = df.drop_duplicates(subset=['keyword', 'product_id'], keep='first')

    removed = initial_count - len(df)
    print(f"   - Removed {removed:,} duplicate records ({removed/initial_count*100:.2f}%)")

    return df

def balance_grade_distribution(df: pd.DataFrame, target_grade_1_pct: float = 0.40) -> pd.DataFrame:
    """Downsample grade 1 to reduce noise (optional but recommended)."""
    print(f"\n7. Balancing grade distribution (target grade 1: {target_grade_1_pct*100}%)...")

    grade_1_count = (df['relevance_grade'] == 1).sum()
    current_pct = grade_1_count / len(df)

    print(f"   - Current grade 1 percentage: {current_pct*100:.2f}%")

    if current_pct > target_grade_1_pct:
        # Calculate how many grade 1 records to keep
        target_count = int(len(df) * target_grade_1_pct / (1 - current_pct + target_grade_1_pct))

        # Separate grade 1 from others
        grade_1_df = df[df['relevance_grade'] == 1].copy()
        other_grades_df = df[df['relevance_grade'] != 1].copy()

        # Sample grade 1
        grade_1_sampled = grade_1_df.sample(n=min(target_count, len(grade_1_df)), random_state=42)

        # Combine back
        df = pd.concat([other_grades_df, grade_1_sampled], ignore_index=True)

        removed = grade_1_count - len(grade_1_sampled)
        print(f"   - Removed {removed:,} grade 1 records")
        print(f"   - New grade 1 percentage: {len(grade_1_sampled)/len(df)*100:.2f}%")
    else:
        print(f"   - Grade 1 percentage is already acceptable, skipping downsampling")

    return df

def print_final_statistics(df: pd.DataFrame):
    """Print final dataset statistics."""
    print("\n" + "="*70)
    print("FINAL DATASET STATISTICS")
    print("="*70)

    print(f"\nTotal records: {len(df):,}")
    print(f"Unique keywords: {df['keyword'].nunique():,}")
    print(f"Unique products: {df['product_id'].nunique():,}")

    print("\nRelevance grade distribution:")
    grade_dist = df['relevance_grade'].value_counts(normalize=True).sort_index() * 100
    for grade, pct in grade_dist.items():
        count = (df['relevance_grade'] == grade).sum()
        print(f"  Grade {grade}: {count:>8,} ({pct:5.2f}%)")

    print("\nCTR statistics:")
    print(f"  Mean CTR: {df['ctr'].mean():.2f}%")
    print(f"  Median CTR: {df['ctr'].median():.2f}%")
    print(f"  Std Dev: {df['ctr'].std():.2f}%")
    print(f"  Min CTR: {df['ctr'].min():.2f}%")
    print(f"  Max CTR: {df['ctr'].max():.2f}%")

    print("\nKeyword statistics:")
    products_per_keyword = df.groupby('keyword').size()
    print(f"  Mean products per keyword: {products_per_keyword.mean():.1f}")
    print(f"  Median products per keyword: {products_per_keyword.median():.1f}")
    print(f"  Keywords with <5 products: {(products_per_keyword < 5).sum()}")
    print(f"  Keywords with >100 products: {(products_per_keyword > 100).sum()}")

    print("\nImpression statistics:")
    print(f"  Mean impressions: {df['total_impressions'].mean():.1f}")
    print(f"  Median impressions: {df['total_impressions'].median():.1f}")
    print(f"  Min impressions: {df['total_impressions'].min()}")
    print(f"  Max impressions: {df['total_impressions'].max()}")

    print("\n" + "="*70)

def main():
    """Main execution function."""
    input_path = "outputs/OLAPBasalam_search_product_keyword_agg.csv"
    output_path = "outputs/OLAPBasalam_search_product_keyword_agg_IMPROVED.csv"

    print("="*70)
    print("JUDGMENT LIST IMPROVEMENT SCRIPT")
    print("="*70)

    # Load data
    df = load_data(input_path)
    initial_count = len(df)

    # Apply improvements
    df = fix_ctr_calculation(df)
    df = apply_consistent_relevance_grading(df)
    df = handle_zero_click_products(df, min_impressions=50)
    df = remove_keywords_with_no_relevant_results(df)
    df = remove_keywords_with_few_products(df, min_products=5)
    df = remove_duplicates(df)

    # Optional: balance grade distribution (uncomment if you want to reduce grade 1 noise)
    # df = balance_grade_distribution(df, target_grade_1_pct=0.40)

    # Print statistics
    print_final_statistics(df)

    # Save improved dataset
    print(f"\nSaving improved dataset to {output_path}...")
    df.to_csv(output_path, index=False)

    final_count = len(df)
    removed_total = initial_count - final_count
    print(f"\n✅ DONE!")
    print(f"   Initial records: {initial_count:,}")
    print(f"   Final records: {final_count:,}")
    print(f"   Removed: {removed_total:,} ({removed_total/initial_count*100:.2f}%)")
    print(f"   Retention: {final_count/initial_count*100:.2f}%")
    print(f"\n   Output file: {output_path}")

    # Create a comparison report
    print("\n📊 Creating comparison report...")
    create_comparison_report(input_path, output_path)

def create_comparison_report(original_path: str, improved_path: str):
    """Create a comparison report between original and improved datasets."""
    df_orig = pd.read_csv(original_path)
    df_improved = pd.read_csv(improved_path)

    report_path = "outputs/IMPROVEMENT_REPORT.md"

    with open(report_path, 'w') as f:
        f.write("# Judgment List Improvement Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Original records:** {len(df_orig):,}\n")
        f.write(f"- **Improved records:** {len(df_improved):,}\n")
        f.write(f"- **Records removed:** {len(df_orig) - len(df_improved):,} ({(len(df_orig) - len(df_improved))/len(df_orig)*100:.2f}%)\n")
        f.write(f"- **Retention rate:** {len(df_improved)/len(df_orig)*100:.2f}%\n\n")

        f.write("## Key Metrics Comparison\n\n")
        f.write("| Metric | Original | Improved | Change |\n")
        f.write("|--------|----------|----------|--------|\n")
        f.write(f"| Total Records | {len(df_orig):,} | {len(df_improved):,} | {len(df_improved) - len(df_orig):,} |\n")
        f.write(f"| Unique Keywords | {df_orig['keyword'].nunique():,} | {df_improved['keyword'].nunique():,} | {df_improved['keyword'].nunique() - df_orig['keyword'].nunique():,} |\n")
        f.write(f"| Unique Products | {df_orig['product_id'].nunique():,} | {df_improved['product_id'].nunique():,} | {df_improved['product_id'].nunique() - df_orig['product_id'].nunique():,} |\n")
        f.write(f"| Mean CTR | {df_orig['ctr'].mean():.2f}% | {df_improved['ctr'].mean():.2f}% | {df_improved['ctr'].mean() - df_orig['ctr'].mean():+.2f}% |\n")
        f.write(f"| Median CTR | {df_orig['ctr'].median():.2f}% | {df_improved['ctr'].median():.2f}% | {df_improved['ctr'].median() - df_orig['ctr'].median():+.2f}% |\n\n")

        f.write("## Relevance Grade Distribution\n\n")
        f.write("| Grade | Original Count | Original % | Improved Count | Improved % | Change |\n")
        f.write("|-------|----------------|------------|----------------|------------|--------|\n")

        for grade in range(5):
            orig_count = (df_orig['relevance_grade'] == grade).sum()
            orig_pct = orig_count / len(df_orig) * 100
            imp_count = (df_improved['relevance_grade'] == grade).sum()
            imp_pct = imp_count / len(df_improved) * 100
            f.write(f"| {grade} | {orig_count:,} | {orig_pct:.2f}% | {imp_count:,} | {imp_pct:.2f}% | {imp_pct - orig_pct:+.2f}% |\n")

        f.write("\n## Improvements Applied\n\n")
        f.write("1. ✅ Fixed CTR calculation inconsistencies\n")
        f.write("2. ✅ Applied consistent relevance grade binning\n")
        f.write("3. ✅ Marked zero-click products (≥50 impressions) as irrelevant\n")
        f.write("4. ✅ Removed keywords with no relevant results\n")
        f.write("5. ✅ Removed keywords with <5 products\n")
        f.write("6. ✅ Removed duplicate keyword-product pairs\n")
        f.write("7. ⏭️  Grade 1 downsampling (optional, commented out)\n\n")

        f.write("## Expected Impact\n\n")
        f.write("Based on these improvements, expected NDCG@10 improvement: **+5-10%**\n\n")
        f.write("- Better data consistency → Better learning signals\n")
        f.write("- Removed noise → More accurate rankings\n")
        f.write("- Removed duplicates → No data leakage\n")
        f.write("- Removed sparse keywords → More robust model\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Use `outputs/OLAPBasalam_search_product_keyword_agg_IMPROVED.csv` for training\n")
        f.write("2. Update XGBoost hyperparameters (see ANALYSIS_SUMMARY.md)\n")
        f.write("3. Re-run training pipeline\n")
        f.write("4. Monitor NDCG@10 improvement\n")
        f.write("5. Consider adding conversion rate and other signals (Phase 3)\n")

    print(f"   Report saved to: {report_path}")

if __name__ == "__main__":
    main()
