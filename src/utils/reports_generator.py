from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import json
import logging
from typing import Dict, Any
from src.utils.correlation_utils import prepare_corr_frame as _prepare_corr_frame
logger = logging.getLogger(__name__)


def generate_enhanced_reports(uploads: pd.DataFrame, output_dir: pathlib.Path):
    """Generate enhanced reports with comprehensive visualizations showing ML value-add."""
    # Determine rule severity threshold for coverage comparison
    if 'label' in uploads.columns:
        if (uploads['label'] >= 2).any():
            rule_severity_threshold = 2          # use only critical (creates separation)
        else:
            rule_severity_threshold = 1          # fallback if no level 2 present
        rule_flag_series = (uploads['label'] >= rule_severity_threshold).astype(int)
    else:
        rule_severity_threshold = None
        rule_flag_series = pd.Series(0, index=uploads.index)

    risk_correlations = {}

    # --- Dynamic ML score thresholds---
    ml_score_col = 'ml_risk_score' if 'ml_risk_score' in uploads.columns else 'risk_score'
    ml_scores = uploads[ml_score_col] if ml_score_col in uploads.columns else pd.Series(0.0, index=uploads.index)
    p80, p95 = ml_scores.quantile([0.80, 0.95]).values
    uploads['ml_high_flag'] = (ml_scores >= p95).astype(int)
    uploads['ml_med_flag']  = ((ml_scores >= p80) & (ml_scores < p95)).astype(int)
    rule_flag_series = (uploads.get('label', uploads.get('risk_label', 0)) >= 1).astype(int)

    # Pre‑define placeholders to avoid NameError later if comparison block skipped
    ml_high_risk   = uploads[uploads['ml_high_flag'] == 1]
    ml_med_risk    = uploads[uploads['ml_med_flag'] == 1]
    rule_high_risk = uploads[rule_flag_series == 1]
    ml_only_flags  = ml_high_risk.iloc[0:0]
    rule_only_flags = rule_high_risk.iloc[0:0]
    both_flags      = ml_high_risk.iloc[0:0]

    report_dir = output_dir / "enhanced_reports"
    report_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Generating enhanced reports with visualizations in {report_dir}")
    print(f"📊 Generating enhanced ML analysis reports...")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # ========================================================================
    # 1. ML vs Rules Comparison Analysis
    # ========================================================================
    if 'risk_score' in uploads.columns and 'label' in uploads.columns:
        print("   📈 Analyzing ML vs Rules performance...")

        # Quantile-based thresholds
        ml_high_risk = uploads[uploads['ml_high_flag'] == 1]
        ml_med_risk  = uploads[uploads['ml_med_flag'] == 1]
        rule_high_risk = uploads[rule_flag_series == 1]

        # Overlap categories
        ml_only_flags   = ml_high_risk[~ml_high_risk.index.isin(rule_high_risk.index)]
        rule_only_flags = rule_high_risk[~rule_high_risk.index.isin(ml_high_risk.index)]
        both_flags      = ml_high_risk[ml_high_risk.index.isin(rule_high_risk.index)]

        # Save ML-only discoveries
        if len(ml_only_flags) > 0:
            ml_only_flags.to_csv(report_dir / "ml_discovered_risks.csv", index=False)
            print(f"   💡 ML discovered {len(ml_only_flags)} high-risk events missed by rules (p95 threshold {p95:.3f})")

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ML vs Rules: Risk Detection Comparison Analysis', fontsize=16, fontweight='bold')

        # 1.1 Coverage pie
        # Coverage pie
        categories = [
            'ML High Only',
            f'Rule (label≥{rule_severity_threshold}) Only',
            'Both Systems',
            'Neither'
        ]
        counts = [
            len(ml_only_flags),
            len(rule_only_flags),
            len(both_flags),
            len(uploads) - (len(ml_only_flags) + len(rule_only_flags) + len(both_flags))
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ax1.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Coverage (Rule label≥{rule_severity_threshold})', fontweight='bold')

        # 1.2 Risk Score Distribution (with quantile lines)
        ax2.hist(ml_scores, bins=50, alpha=0.7, color='skyblue', label='All Events')
        ax2.hist(ml_high_risk[ml_score_col], bins=20, alpha=0.75, color='red', label='High (>=p95)')
        if len(ml_med_risk) > 0:
            ax2.hist(ml_med_risk[ml_score_col], bins=20, alpha=0.6, color='orange', label='Medium (p80–p95)')
        ax2.axvline(p80, color='orange', linestyle='--', linewidth=2, label=f'p80 {p80:.2f}')
        ax2.axvline(p95, color='red', linestyle='--', linewidth=2, label=f'p95 {p95:.2f}')
        ax2.set_xlabel(f'ML Risk Score ({ml_score_col})')
        # ax2.set_ylabel('Count')
        # ax2.set_title('ML Risk Score Distribution')
        # ax2.legend()
        # ax2.grid(True, alpha=0.3)

        # 1.3 Rules label distribution
        label_counts = uploads['label'].value_counts().sort_index()
        bars = ax3.bar(label_counts.index, label_counts.values,
                       color=['green', 'orange', 'red', 'darkred'][:len(label_counts)])
        ax3.set_xlabel('Risk Label (Rules-Based)')
        ax3.set_ylabel('Count')
        ax3.set_title('Rules-Based Risk Label Distribution')
        ax3.grid(True, alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}', ha='center', va='bottom')

        # 1.4 Scatter correlation
        scatter = ax4.scatter(uploads['label'], ml_scores, alpha=0.6,
                              c=ml_scores, cmap='Reds', s=30)
        ax4.set_xlabel('Rules-Based Label')
        ax4.set_ylabel('ML Risk Score')
        ax4.set_title('ML Score vs Rules Label Correlation')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Risk Score')

        plt.tight_layout()
        plt.savefig(report_dir / "ml_vs_rules_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Pattern analysis summary uses quantile threshold counts
        pattern_analysis = {
            "total_events": len(uploads),
            "ml_high_risk_events": len(ml_high_risk),
            "ml_medium_risk_events": len(ml_med_risk),
            "rule_high_risk_events": len(rule_high_risk),
            "ml_only_discoveries": len(ml_only_flags),
            "rule_only_flags": len(rule_only_flags),
            "common_flags": len(both_flags),
            "ml_precision_with_rules": len(both_flags) / len(ml_high_risk) if len(ml_high_risk) else 0.0,
            "rule_coverage": len(rule_high_risk) / len(uploads),
            "ml_high_coverage": len(ml_high_risk) / len(uploads),
            "ml_unique_contribution": len(ml_only_flags) / len(uploads),
            "p80_threshold": float(p80),
            "p95_threshold": float(p95),
            "rule_severity_threshold": rule_severity_threshold
        }
        with open(report_dir / "ml_value_analysis.json", "w") as f:
            json.dump(pattern_analysis, f, indent=2)


    # ========================================================================
    # 2. Anomaly Detection Analysis
    # ========================================================================
    
    anomaly_cols = [col for col in uploads.columns if 'anomaly' in col.lower()]
    # Drop zero-variance or all-NaN anomaly columns to avoid correlation warnings
    dropped_anomaly = [c for c in anomaly_cols if uploads[c].std(ddof=0) == 0 or uploads[c].notna().sum() == 0]
    anomaly_cols = [c for c in anomaly_cols if c not in dropped_anomaly]
    if dropped_anomaly:
        logger.info(f"Dropped {len(dropped_anomaly)} constant/empty anomaly features: {dropped_anomaly[:10]}{'...' if len(dropped_anomaly)>10 else ''}")
    
    if anomaly_cols:
        print(f"   🔍 Analyzing {len(anomaly_cols)} anomaly detection features...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anomaly Detection Feature Analysis', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, col in enumerate(anomaly_cols[:6]):  # Show up to 6 anomaly features
            ax = axes[i]
            ax.hist(uploads[col], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax.axvline(uploads[col].mean(), color='blue', linestyle='--', 
                       label=f'Mean: {uploads[col].mean():.3f}')
            ax.axvline(uploads[col].quantile(0.95), color='red', linestyle='--', 
                       label=f'95th pct: {uploads[col].quantile(0.95):.3f}')
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_ylabel('Count')
            ax.set_title(f'{col.replace("_", " ").title()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for j in range(len(anomaly_cols), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(report_dir / "anomaly_detection_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Summary stats
        uploads[anomaly_cols].describe().to_csv(report_dir / "anomaly_detection_summary.csv")
        
        # Correlations vs ML risk score (use ml_scores to align target choice)
        if ml_scores.std(ddof=0) > 0:
            anomaly_correlations = {}
            for col in anomaly_cols:
                if uploads[col].std(ddof=0) == 0:  # extra safety
                    continue
                corr_val = uploads[col].corr(ml_scores)
                if pd.notna(corr_val):
                    anomaly_correlations[col] = corr_val
            
            if anomaly_correlations:
                plt.figure(figsize=(12, 8))
                sorted_corrs = sorted(anomaly_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                features, correlations = zip(*sorted_corrs)
                colors = ['red' if c > 0 else 'blue' for c in correlations]
                bars = plt.barh(range(len(features)), correlations, color=colors, alpha=0.7)
                plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
                plt.xlabel(f'Correlation with {ml_score_col}')
                plt.title('Anomaly Features Correlation with ML Risk Score', fontweight='bold')
                plt.grid(True, alpha=0.3, axis='x')
                for i,(bar,corr_val) in enumerate(zip(bars, correlations)):
                    plt.text(corr_val + (0.01 if corr_val>0 else -0.01), i, f'{corr_val:.3f}',
                             va='center', ha='left' if corr_val>0 else 'right')
                plt.tight_layout()
                plt.savefig(report_dir / "anomaly_correlations.png", dpi=300, bbox_inches='tight')
                plt.close()
    else:
        logger.info("No anomaly features available after filtering.")

        
    # ========================================================================
    # 3. Temporal Pattern Analysis
    # ========================================================================
    
    print("   ⏰ Analyzing temporal patterns...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temporal Risk Pattern Analysis', fontsize=16, fontweight='bold')
    
    # 3.1 Risk by Hour of Day
    if 'hour' in uploads.columns:
        hourly_risk = uploads.groupby('hour').agg({
            'risk_score': ['mean', 'count'] if 'risk_score' in uploads.columns else ['count'],
            'label': 'mean' if 'label' in uploads.columns else 'count'
        }).round(3)
        
        if 'risk_score' in uploads.columns:
            ax1.plot(hourly_risk.index, hourly_risk[('risk_score', 'mean')], 
                    marker='o', linewidth=2, color='red', label='Avg Risk Score')
            ax1_twin = ax1.twinx()
            ax1_twin.bar(hourly_risk.index, hourly_risk[('risk_score', 'count')], 
                        alpha=0.3, color='blue', label='Event Count')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Average Risk Score', color='red')
            ax1_twin.set_ylabel('Event Count', color='blue')
            ax1.set_title('Risk Score by Hour of Day')
            ax1.grid(True, alpha=0.3)
            
            # Highlight high-risk hours
            high_risk_hours = hourly_risk[hourly_risk[('risk_score', 'mean')] > 
                                       hourly_risk[('risk_score', 'mean')].quantile(0.8)]
            for hour in high_risk_hours.index:
                ax1.axvline(hour, color='orange', alpha=0.5, linestyle='--')
    
    # 3.2 Risk by Day of Week
    if 'day_of_week' in uploads.columns or uploads['date'].dtype.name.startswith('datetime'):
        if 'day_of_week' not in uploads.columns:
            uploads['day_of_week'] = uploads['date'].dt.dayofweek
        
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_risk = uploads.groupby('day_of_week').agg({
            'risk_score': 'mean' if 'risk_score' in uploads.columns else 'count',
            'label': 'mean' if 'label' in uploads.columns else 'count'
        }).round(3)
        
        if 'risk_score' in uploads.columns:
            bars = ax2.bar(range(7), daily_risk['risk_score'], 
                          color=['lightblue' if i < 5 else 'lightcoral' for i in range(7)])
            ax2.set_xticks(range(7))
            ax2.set_xticklabels(dow_names)
            ax2.set_ylabel('Average Risk Score')
            ax2.set_title('Risk Score by Day of Week')
            ax2.grid(True, alpha=0.3)
            
            # Highlight weekend
            ax2.axvspan(4.5, 6.5, alpha=0.2, color='red', label='Weekend')
            ax2.legend()
    
    # 3.3 Upload Volume Over Time
    daily_stats = uploads.groupby(uploads['date'].dt.date).agg({
        'risk_score': ['mean', 'count'] if 'risk_score' in uploads.columns else ['count'],
        'label': ['mean', 'sum'] if 'label' in uploads.columns else ['count']
    })
    
    ax3.plot(daily_stats.index, daily_stats[('risk_score', 'count')] if 'risk_score' in uploads.columns else daily_stats.iloc[:, 0], 
            color='blue', alpha=0.7, label='Daily Upload Count')
    if 'risk_score' in uploads.columns:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(daily_stats.index, daily_stats[('risk_score', 'mean')], 
                     color='red', alpha=0.7, label='Daily Avg Risk Score')
        ax3_twin.set_ylabel('Average Risk Score', color='red')
    
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Upload Count', color='blue')
    ax3.set_title('Upload Volume and Risk Over Time')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 3.4 Channel Analysis
    if 'channel' in uploads.columns:
        channel_risk = uploads.groupby('channel').agg({
            'risk_score': ['mean', 'count'] if 'risk_score' in uploads.columns else ['count'],
            'label': ['mean', 'sum'] if 'label' in uploads.columns else ['count']
        }).round(3)
        
        if 'risk_score' in uploads.columns:
            bars = ax4.bar(channel_risk.index, channel_risk[('risk_score', 'mean')], 
                          color=['green', 'orange', 'red'][:len(channel_risk)])
            ax4.set_ylabel('Average Risk Score')
            ax4.set_title('Risk Score by Channel')
            ax4.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for i, (channel, bar) in enumerate(zip(channel_risk.index, bars)):
                count = channel_risk.loc[channel, ('risk_score', 'count')]
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'n={int(count)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(report_dir / "temporal_risk_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 4. Feature Importance Analysis
    # ========================================================================

    print("   🎯 Analyzing feature importance...")

    # --- Sanitized correlation frame (drop leakage & zero-variance) ---
    leakage_cols = [
        'ml_risk_score', 'ml_high_flag', 'ml_med_flag',
        'risk_label', 'composite_anomaly_score'  # add others if clearly target-derived
    ]

    # Decide target
    target_col = 'risk_score' if 'risk_score' in uploads.columns else 'label'
    # Do not treat the target itself as leakage for building correlation list
    corr_df = _prepare_corr_frame(uploads.drop(columns=['label']) if target_col == 'risk_score' and 'label' in uploads.columns else uploads,
                                  leakage_cols=leakage_cols)

    if target_col not in corr_df.columns:
        # Add target back if it was dropped (ensure numeric & non-constant)
        corr_df[target_col] = uploads[target_col]

    # Recompute after potential insertion
    corr_df = corr_df.loc[:, corr_df.std(ddof=0) > 0]

    risk_correlations = {}
    if target_col in corr_df.columns and corr_df[target_col].std(ddof=0) > 0:
        feature_cols = [c for c in corr_df.columns if c != target_col]
        for col in feature_cols:
            # Skip IDs or purely index-like
            if col.endswith('_id'):
                continue
            corr = corr_df[col].corr(corr_df[target_col])
            if pd.notna(corr):
                risk_correlations[col] = float(corr)

        if risk_correlations:
            sorted_corrs = sorted(risk_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
            features, correlations = zip(*sorted_corrs)

            plt.figure(figsize=(14, 10))
            colors = ['red' if c > 0 else 'blue' for c in correlations]
            bars = plt.barh(range(len(features)), correlations, color=colors, alpha=0.75)
            plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
            plt.xlabel(f'Correlation with {target_col.replace("_", " ").title()}')
            plt.title('Top 20 Feature Correlations (Sanitized)', fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')

            for i, (bar, corr) in enumerate(zip(bars, correlations)):
                xoff = 0.01 if corr > 0 else -0.01
                plt.text(corr + xoff, i, f'{corr:.3f}',
                         va='center', ha='left' if corr > 0 else 'right')

            plt.tight_layout()
            plt.savefig(report_dir / "feature_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()

            with open(report_dir / "feature_correlations.json", "w") as f:
                json.dump(dict(sorted_corrs), f, indent=2)
    else:
        risk_correlations = {}

    # ========================================================================
    # 5. Geographic and Network Analysis 
    # ========================================================================
    
    if 'destination_domain' in uploads.columns:
        print("   🌐 Analyzing network destinations...")
        
        # Top destinations analysis
        top_destinations = uploads['destination_domain'].value_counts().head(15)
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(top_destinations)), top_destinations.values, 
                      color='skyblue', alpha=0.7)
        plt.xticks(range(len(top_destinations)), top_destinations.index, rotation=45, ha='right')
        plt.ylabel('Upload Count')
        plt.title('Top 15 Upload Destinations', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, top_destinations.values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(top_destinations.values) * 0.01,
                    f'{int(count)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(report_dir / "top_destinations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Risk by destination
        if 'risk_score' in uploads.columns:
            dest_risk = uploads.groupby('destination_domain').agg({
                'risk_score': ['mean', 'count'],
                'label': 'mean' if 'label' in uploads.columns else 'count'
            }).round(3)
            
            # Filter destinations with sufficient samples
            dest_risk_filtered = dest_risk[dest_risk[('risk_score', 'count')] >= 5]
            dest_risk_sorted = dest_risk_filtered.sort_values(('risk_score', 'mean'), ascending=False).head(10)
            
            if len(dest_risk_sorted) > 0:
                plt.figure(figsize=(14, 8))
                bars = plt.bar(range(len(dest_risk_sorted)), 
                              dest_risk_sorted[('risk_score', 'mean')].values,
                              color='lightcoral', alpha=0.7)
                plt.xticks(range(len(dest_risk_sorted)), dest_risk_sorted.index, rotation=45, ha='right')
                plt.ylabel('Average Risk Score')
                plt.title('Top 10 Highest Risk Destinations (min 5 uploads)', fontweight='bold')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add count labels
                for i, (bar, dest) in enumerate(zip(bars, dest_risk_sorted.index)):
                    count = dest_risk_sorted.loc[dest, ('risk_score', 'count')]
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                            f'n={int(count)}', ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(report_dir / "highest_risk_destinations.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    # ========================================================================
    # 6. Executive Summary Dashboard
    # ========================================================================
    
    print("   📋 Creating executive summary dashboard...")
    
    # Calculate key metrics
    total_events = len(uploads)
    high_risk_events = int((uploads['ml_high_flag'] == 1).sum())#len(uploads[uploads['risk_score'] > 0.6]) if 'risk_score' in uploads.columns else 0
    flagged_events = len(uploads[uploads['label'] >= 1]) if 'label' in uploads.columns else 0

    high_risk_mask = (uploads['ml_high_flag'] == 1)
    rule_flag_mask = (uploads['label'] >= 1) if 'label' in uploads.columns else pd.Series(False, index=uploads.index)
    union_flags = (high_risk_mask | rule_flag_mask).sum()
    clean_events = total_events - union_flags
    
    # Create summary visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Executive Risk Dashboard', fontsize=18, fontweight='bold')
    
    # Key metrics
    metrics = ['Total Events', 'ML High Risk', 'Rule Flagged', 'Clean Events']
    values = [total_events, high_risk_events, flagged_events, clean_events]
    colors = ['lightblue', 'orange', 'red', 'lightgreen']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
    ax1.set_ylabel('Count')
    ax1.set_title('Event Classification Summary')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values) * 0.01,
               f'{int(value):,}', ha='center', va='bottom', fontweight='bold')
    
    # Risk distribution over time
    if 'risk_score' in uploads.columns:
        weekly_risk = uploads.set_index('date').resample('W')['risk_score'].mean()
        ax2.plot(weekly_risk.index, weekly_risk.values, marker='o', linewidth=2, color='red')
        ax2.set_ylabel('Average Risk Score')
        ax2.set_title('Weekly Risk Trend')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # Channel risk comparison
    if 'channel' in uploads.columns and 'risk_score' in uploads.columns:
        channel_stats = uploads.groupby('channel').agg({
            'risk_score': ['mean', 'count'],
            'label': 'sum' if 'label' in uploads.columns else 'count'
        })
        
        bars = ax3.bar(channel_stats.index, channel_stats[('risk_score', 'mean')], 
                      color=['green', 'yellow', 'red'][:len(channel_stats)], alpha=0.7)
        ax3.set_ylabel('Average Risk Score')
        ax3.set_title('Risk by Channel')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Top risk factors
    if risk_correlations:
        top_risk_factors = dict(list(sorted(risk_correlations.items(), 
                                          key=lambda x: abs(x[1]), reverse=True))[:5])
        factors = list(top_risk_factors.keys())
        importance = list(top_risk_factors.values())
        
        bars = ax4.barh(factors, importance, color=['red' if x > 0 else 'blue' for x in importance], alpha=0.7)
        ax4.set_xlabel('Risk Correlation')
        ax4.set_title('Top 5 Risk Factors')
        ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(report_dir / "executive_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 7. Generate Summary Report
    # ========================================================================
    
    summary_stats = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "rule_severity_threshold": rule_severity_threshold,
        "ml_score_column": ml_score_col,
        "p80_threshold": float(p80),
        "p95_threshold": float(p95),
        "total_events_analyzed": int(total_events),
        "high_risk_events_ml": int(high_risk_events),
        "flagged_events_rules": int(flagged_events),
        "clean_events": int(clean_events),
        "ml_detection_rate": float(high_risk_events / total_events) if total_events > 0 else 0,
        "rule_detection_rate": float(flagged_events / total_events) if total_events > 0 else 0,
        "union_flag_coverage": float((high_risk_mask | rule_flag_mask).mean()) if total_events else 0,
        "temporal_range": {
            "start": uploads['date'].min().isoformat(),
            "end": uploads['date'].max().isoformat(),
            "duration_days": (uploads['date'].max() - uploads['date'].min()).days
        },
        "channels_analyzed": uploads['channel'].unique().tolist() if 'channel' in uploads.columns else [],
        "anomaly_features_count": len(anomaly_cols),
        "key_insights": {
            "highest_risk_hour": int(uploads.groupby('hour')[ml_score_col].mean().idxmax()) if 'hour' in uploads.columns else None,
            "riskiest_channel": uploads.groupby('channel')[ml_score_col].mean().idxmax() if 'channel' in uploads.columns else None,
            "peak_activity_day": uploads['date'].dt.day_name().mode().iloc[0] if len(uploads) > 0 else None
        }
    }

    with open(report_dir / "executive_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    # Generate markdown report
    markdown_report = f"""
# Enhanced ML Risk Analysis Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Events Analyzed:** {total_events:,}
- **ML High-Risk Events (>=p95 {p95:.2f}, column {ml_score_col}):** {high_risk_events:,} ({high_risk_events/len(uploads)*100:.1f}%)
- **Rule-Flagged Events:** {flagged_events:,} ({(flagged_events/total_events*100):.1f}%)
- **Clean Events:** {clean_events:,} ({clean_events/total_events*100:.1f}%)
- **Analysis Period:** {uploads['date'].min().strftime('%Y-%m-%d')} to {uploads['date'].max().strftime('%Y-%m-%d')}

## Key Findings
### ML Value-Add
- ML discovered **{len(ml_only_flags)}** high-risk events not in rule tier (label ≥ {rule_severity_threshold})
- Overlap (both systems): **{len(both_flags)}**
- Pattern correlation analysis identified **{len(risk_correlations)}** candidate risk indicators
- Anomaly features analyzed: **{len(anomaly_cols)}**

### Risk Patterns
- **Highest Risk Hour:** {summary_stats['key_insights']['highest_risk_hour'] if summary_stats['key_insights']['highest_risk_hour'] is not None else 'N/A'}:00
- **Riskiest Channel:** {summary_stats['key_insights']['riskiest_channel'] or 'N/A'}
- **Peak Activity Day:** {summary_stats['key_insights']['peak_activity_day'] or 'N/A'}

## Visualizations Generated
1. ml_vs_rules_comparison.png
2. anomaly_detection_analysis.png
3. temporal_risk_analysis.png
4. feature_correlations.png
5. executive_dashboard.png

## Recommendations
- Focus monitoring on high-risk hours/channels
- Review ML-only discoveries for emerging patterns
- Tune rules using quantile-derived thresholds (p80/p95)
- Continue feature calibration to increase score resolution

---
*Report generated by Enhanced ML Risk Analysis Pipeline*
"""

    with open(report_dir / "analysis_report.md", "w") as f:
        f.write(markdown_report)
    
    print(f"✅ Enhanced reports generated successfully!")
    print(f"   📊 {len([f for f in report_dir.glob('*.png')])} visualizations created")
    print(f"   📄 {len([f for f in report_dir.glob('*.csv')])} data files exported")
    print(f"   📋 Executive summary and detailed reports available")
    
    logger.info(f"Enhanced reports with visualizations generated in {report_dir}")
    
    return {
        "report_directory": str(report_dir),
        "visualizations_created": len([f for f in report_dir.glob('*.png')]),
        "data_files_created": len([f for f in report_dir.glob('*.csv')]),
        "summary_stats": summary_stats
    }