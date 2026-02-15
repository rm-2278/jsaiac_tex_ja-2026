#!/usr/bin/env python3
"""
Hieros階層性の変更実験の解析スクリプト（図8スタイル）
max_hierarchyパラメータの影響をepisode/scoreで可視化
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def create_media_dir():
    """出力ディレクトリの作成"""
    output_dir = Path("media/hierarchy")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def fetch_hierarchy_sweep_data():
    """階層性実験のスイープデータを取得"""
    api = wandb.Api()
    
    # スイープの情報
    project = "rm2278-university-of-cambridge/Hieros-hieros" 
    sweep_id = "myeqsabh"
    
    print(f"Fetching sweep: {sweep_id}")
    sweep = api.sweep(f"{project}/sweeps/{sweep_id}")
    
    print(f"Found {len(sweep.runs)} runs")
    
    # 図8と同じデータ収集方式
    valid_runs = []
    
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
            
        # 図8と同じhistory取得方式
        history = run.history(keys=["episode/score", "_step"])
        
        if history.empty or "episode/score" not in history.columns:
            print(f"Skipping run {run.name}: no episode/score data")
            continue
        
        # 図8と同じデータクリーニング
        df = history.dropna(subset=["episode/score"]).sort_values("_step")
        
        if df.empty:
            print(f"Skipping run {run.name}: all NaN")
            continue
            
        # max_hierarchyとrun情報を追加
        df['max_hierarchy'] = max_hierarchy
        df['run_name'] = run.name
        df['run_id'] = run.id
        
        valid_runs.append(df)
        print(f"✓ Added run {run.name}: max_hierarchy={max_hierarchy}, {len(df)} data points")
    
    if not valid_runs:
        raise ValueError("No valid runs found in sweep")
        
    # 全データを結合
    all_data = pd.concat(valid_runs, ignore_index=True)
    print(f"Total data points: {len(all_data)}")
    print(f"Max hierarchy values: {sorted(all_data['max_hierarchy'].unique())}")
    
    return all_data

def create_episode_score_plot(data, output_dir):
    """episode/scoreの学習曲線を作成（図8のスタイルを完全に模倣）"""
    
    # 図8と全く同じ設定
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
    
    # max_hierarchyごとにプロット
    hierarchy_values = sorted(data['max_hierarchy'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, max_hier in enumerate(hierarchy_values):
        hier_data = data[data['max_hierarchy'] == max_hier]
        color = colors[i % len(colors)]
        
        # 各runのデータを個別に取得
        for run_id in hier_data['run_id'].unique():
            run_data = hier_data[hier_data['run_id'] == run_id].sort_values("_step")
            
            if len(run_data) == 0:
                continue
                
            x = run_data["_step"] / 1000  # thousands of steps  
            y = run_data["episode/score"]
            
            # 図8と同じスムージング
            window = 20
            y_smooth = y.rolling(window=window, min_periods=1).mean()
            
            # 最初のrunだけlabelを付ける
            if run_id == hier_data['run_id'].unique()[0]:
                label = f'max_hierarchy={max_hier}'
            else:
                label = None
            
            # 図8と全く同じスタイリング
            ax.plot(x, y_smooth, linewidth=1.4, label=label, alpha=0.8, color=color)
    
    # 図8と全く同じ軸とスタイリング設定
    ax.set_xlabel("Env. Steps (×10³)", fontsize=9)
    ax.set_ylabel("Episode Return", fontsize=9)
    ax.legend(fontsize=6, loc="best")  # 図8と同じlegend fontsize
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 図8と同じ保存設定
    output_path = output_dir / "hierarchy_episode_scores.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"✓ Saved: {output_path}")
    return output_path

def main():
    """メイン実行関数"""
    print("Fetching hierarchy sweep data...")
    data = fetch_hierarchy_sweep_data()
    
    print("\nCreating episode score plot...")
    output_dir = create_media_dir()
    create_episode_score_plot(data, output_dir)
    
    print(f"\n✓ All hierarchy analysis visualizations complete!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()