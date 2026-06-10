# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

歩行者軌道予測（Social-GAN）と Frenet 座標系経路計画を統合した自動運転シミュレーション研究プロジェクト。AVEC 論文向けの実験コードベースであり、README・ドキュメントは日本語。開発は `dev` ブランチで行い、PR は `main` に向ける。

## 環境とコマンド

仮想環境は `.venv`（Python 3.12、pysocialforce / torch インストール済み）。README には `venv` と書かれているが実体は `.venv`。

```bash
# テスト（pytest 9.x）
.venv/bin/python -m pytest tests/
.venv/bin/python -m pytest tests/test_frenet_planner.py -k <name>   # 単一テスト

# シミュレーション実行（--method cv|lstm|sgan、--animate でGIF/MP4生成）
.venv/bin/python examples/run_simulation.py --scenario scenarios/scenario_01.yaml --method sgan

# ベンチマーク
.venv/bin/python examples/benchmark_prediction.py --scenario scenarios/scenario_01.yaml  # 3手法比較 → output/benchmark/
.venv/bin/python examples/run_statistical_benchmark.py   # 複数シード統計（mean±std、乗り心地指標含む）
.venv/bin/python examples/run_da_poc.py                   # chance-constrained 計画の PoC

# 学習済みモデルのダウンロード（SGAN実行に必須。未指定/欠損だと RuntimeError）
python scripts/download_sgan_models.py --pooling
```

`tests/manual_test_headless.py` と `tests/benchmark_collision.py` は pytest 用ではなく手動実行スクリプト。

論文図版スクリプト `examples/plot_simulation_figs.py` / `plot_lateral_analysis.py` は `output/scenario_0X/trajectory.npz` を読み、リポジトリ外の `~/Research/AVEC_FullPaper/figs/` に出力する。

## アーキテクチャ

1ステップのデータフロー（`src/simulation/integrated_simulator.py` の `IntegratedSimulator.step()` が統合点）:

```
PedestrianSimulator (pysocialforce; Ego車両を斥力源として歩行者が回避)
  → PedestrianObserver (src/pedestrian/observer.py; シミュレーション dt に依らず
     SGAN想定の 0.4s 間隔へダウンサンプリングして観測履歴を保持)
  → TrajectoryPredictor (src/prediction/trajectory_predictor.py; SGAN本体は
     src/prediction/sgan_vendor/ にベンダリング)
  → 予測後処理: SGAN出力(12ステップ@0.4s)をプランナ dt(0.1s) に線形補間し、
     計画ホライゾン max_t(5.0s) まで等速外挿。予測失敗時は等速直線モデルで代替
  → FrenetPlanner (src/planning/frenet_planner.py; cubic_spline の参照経路上で
     quintic polynomial 候補をベクトル化生成し、コスト最小の衝突フリー経路を選択)
  → FailSafeStateMachine (src/core/state_machine.py; NORMAL→CAUTION→EMERGENCY で
     加速度・曲率制約を段階的に緩和、再計画は3回/ステップ上限、最終的に緊急停止)
```

設計上の重要な約束事:

- **衝突判定は「同時刻位置」のみ評価**する（将来軌道を平坦化しない）。動的障害物の時間次元を保つことが前提なので、予測フォールバックも必ず計画ホライゾン分の時系列を生成する。
- **予測モード3種**: `cv`（等速）/ `lstm`（SGAN w/o Pooling = `models/sgan-models/`）/ `sgan`（Pooling あり = `models/sgan-p-models/`）。ベンチマークスクリプトの `resolve_model_path()` がモード名からモデルディレクトリを切替える。
- **分布対応計画（PoC）**: `distribution_aware_planning: true` で SGAN の全 `num_samples` サンプル（PoC シナリオでは 20）に対する chance-constrained 衝突判定（`chance_epsilon` = 衝突許容サンプル割合、0.0 = worst-case）に切替わる。デフォルトは単一サンプル（`predict_single_best`）。
- **設定**: `src/config/__init__.py` の `SimulationConfig` dataclass がすべての設定の単一情報源。`load_config()` が `default_config.yaml` をベースにシナリオ YAML をマージし、`validate_config()` で読み込み時に検証する。設定キーを追加する際は dataclass・`default_config.yaml`・バリデーションの3箇所を揃える。
- **可視化は完全分離**: `visualization_enabled: false`（ベンチマークでは必須）で `src/visualization/` をスキップしてヘッドレス実行できる。結果は各シナリオ YAML の `output_path`（例: `output/scenario_01/`）に保存（dashboard.png、trajectory.npz など）。
- **ウォームアップ**: t=0 で観測履歴をプリロール生成するため、開始直後から SGAN 予測が有効。シミュレーションはゴール 2m 以内到達で `total_time` を待たず自動終了する。

## シナリオ

`scenarios/scenario_0{1,2,3}.yaml`（交差・狭路すれ違い・交差点右折）。`_cv` / `_lstm` サフィックス版は予測モード別の派生。S2/S3 のプランナコスト重みはデフォルト（全1.0）に統一済み — 変更時はシナリオ間の公平性に注意。
