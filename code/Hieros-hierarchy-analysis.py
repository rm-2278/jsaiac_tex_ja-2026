#!/usr/bin/env python3
"""
Hieros階層性の変更実験の解析スクリプト
max_hierarchyパラメータの影響をepisode/scoreとheatmapで可視化
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def setup_matplotlib():
    """Matplotlibの設定を他のグラフと統一"""
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

def create_media_dir():
    """出力ディレクトリの作成"""
    media_dir = Path("media/hierarchy")
    media_dir.mkdir(parents=True, exist_ok=True)
    return media_dir

def fetch_hierarchy_sweep_data():
    """階層性実験のスイープデータを取得"""
    api = wandb.Api()
    
    # スイープの情報
    project = "rm2278-university-of-cambridge/Hieros-hieros" 
    sweep_id = "myeqsabh"
    
    print(f"Fetching sweep: {sweep_id}")
    sweep = api.sweep(f"{project}/sweeps/{sweep_id}")
    
    runs_data = []
    
    print("Processing runs...")
    for run in sweep.runs:
        if run.state != "finished":
            print(f"Skipping run {run.name}: state={run.state}")
            continue
            
        # max_hierarchyパラメータを取得
        max_hierarchy = run.config.get('max_hierarchy', None)
        if max_hierarchy is None:
            print(f"Skipping run {run.name}: no max_hierarchy config")
            continue
            
        # max_hierarchyを数値に変換
        try:
            max_hierarchy = int(max_hierarchy)
        except (ValueError, TypeError):
            print(f"Skipping run {run.name}: invalid max_hierarchy value: {max_hierarchy}")
            continue
            
        # ヒストリーを取得
        history = run.scan_history(keys=["episode/score", "_step"])
        df = pd.DataFrame(history)
        
        if df.empty or "episode/score" not in df.columns:
            print(f"Skipping run {run.name}: no episode/score data")
            continue
            
        df = df.dropna(subset=["episode/score"])
        if df.empty:
            continue
            
        # max_hierarchyを追加
        df['max_hierarchy'] = max_hierarchy
        df['run_name'] = run.name
        df['run_id'] = run.id
        
        runs_data.append(df)
        print(f"✓ Added run {run.name}: max_hierarchy={max_hierarchy}, {len(df)} data points")
    
    if not runs_data:
        raise ValueError("No valid runs found in sweep")
        
    # 全データを結合
    all_data = pd.concat(runs_data, ignore_index=True)
    print(f"Total data points: {len(all_data)}")
    print(f"Max hierarchy values: {sorted(all_data['max_hierarchy'].unique())}")
    
    return all_data

def create_episode_score_plot(data, output_dir):
    """episode/scoreの学習曲線を作成"""
    plt.figure(figsize=(12, 8))
    
    # max_hierarchyごとに色分け（他のグラフと同じカラーパレット使用）
    hierarchy_values = sorted(data['max_hierarchy'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # matplotlib default colors
    
    for i, max_hier in enumerate(hierarchy_values):
        hier_data = data[data['max_hierarchy'] == max_hier]
        color = colors[i % len(colors)]
        
        # ステップごとにグループ化して平均を計算
        mean_data = hier_data.groupby('_step')['episode/score'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        # X軸を1000で割って表示（他のグラフと統一）
        x = mean_data['_step'] / 1000
        
        # 平均線をプロット
        plt.plot(x, mean_data['mean'], 
                color=color, linewidth=1.4, 
                label=f'max_hierarchy={max_hier} (n={len(hier_data["run_id"].unique())})', alpha=0.8)
        
        # min-maxの帯を追加
        plt.fill_between(x, mean_data['min'], mean_data['max'], color=color, alpha=0.2)
    
    plt.xlabel('Env. Steps (×10³)', fontsize=9)
    plt.ylabel('Episode Return', fontsize=9)
    plt.title('Learning Curves by Max Hierarchy Level', fontsize=10, fontweight='bold')
    plt.legend(fontsize=8, loc='best')
    
    # 他のグラフと同じスタイリング
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "hierarchy_episode_scores.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return output_path

def create_performance_heatmap(data, output_dir):
    """最終性能のヒートマップを作成"""
    # 各ランの最終スコアを取得
    final_scores = []
    
    for run_id in data['run_id'].unique():
        run_data = data[data['run_id'] == run_id].sort_values('_step')
        if len(run_data) > 0:
            final_score = run_data['episode/score'].iloc[-1]
            max_hierarchy = run_data['max_hierarchy'].iloc[0]
            final_scores.append({
                'run_id': run_id,
                'max_hierarchy': max_hierarchy,
                'final_score': final_score,
                'run_name': run_data['run_name'].iloc[0]
            })
    
    final_df = pd.DataFrame(final_scores)
    
    # ヒートマップ用のピボットテーブルを作成
    # max_hierarchyごとの統計値を計算
    hierarchy_stats = final_df.groupby('max_hierarchy')['final_score'].agg([
        'mean', 'std', 'count', 'min', 'max'
    ]).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左側：平均スコアのバープロット
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bar_colors = [colors[i % len(colors)] for i in range(len(hierarchy_stats))]
    bars = ax1.bar(hierarchy_stats['max_hierarchy'], hierarchy_stats['mean'], 
                   yerr=hierarchy_stats['std'], capsize=5, color=bar_colors, alpha=0.8)
    ax1.set_xlabel('Max Hierarchy Level', fontsize=9)
    ax1.set_ylabel('Mean Final Episode Score', fontsize=9)
    ax1.set_title('Final Performance by Max Hierarchy', fontsize=10, fontweight='bold')
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="both", labelsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 各バーに値を表示
    for i, (bar, mean_val, count) in enumerate(zip(bars, hierarchy_stats['mean'], hierarchy_stats['count'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + hierarchy_stats['std'].iloc[i],
                f'{mean_val:.1f}\n(n={count})', ha='center', va='bottom', fontsize=9)
    
    # 右側：個別ランのスコア散布図
    hierarchy_values = sorted(final_df['max_hierarchy'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, max_hier in enumerate(hierarchy_values):
        hier_data = final_df[final_df['max_hierarchy'] == max_hier]
        color = colors[i % len(colors)]
        # x軸に少しジッターを加えて重複を避ける
        x_pos = [max_hier + np.random.normal(0, 0.1) for _ in range(len(hier_data))]
        ax2.scatter(x_pos, hier_data['final_score'], 
                   color=color, alpha=0.7, s=50, label=f'max_hierarchy={max_hier}')
    
    ax2.set_xlabel('Max Hierarchy Level', fontsize=9)
    ax2.set_ylabel('Final Episode Score', fontsize=9)
    ax2.set_title('Individual Run Performance Distribution', fontsize=10, fontweight='bold')
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", labelsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    
    output_path = output_dir / "hierarchy_performance_analysis.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # 統計サマリーを出力
    print("\n=== Performance Summary ===")
    for _, row in hierarchy_stats.iterrows():
        print(f"max_hierarchy={row['max_hierarchy']}: "
              f"mean={row['mean']:.2f}±{row['std']:.2f} "
              f"(n={row['count']}, range=[{row['min']:.1f}, {row['max']:.1f}])")
    
    return output_path

def main():
    """メイン実行関数"""
    setup_matplotlib()
    output_dir = create_media_dir()
    
    try:
        # データ取得
        print("Fetching hierarchy sweep data...")
        data = fetch_hierarchy_sweep_data()
        
        # 学習曲線の作成
        print("\nCreating episode score plot...")
        create_episode_score_plot(data, output_dir)
        
        # ヒートマップ/性能解析の作成
        print("\nCreating performance analysis...")
        create_performance_heatmap(data, output_dir)
        
        print(f"\n✓ All hierarchy analysis visualizations complete!")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()