# Integrated Path Planning with Pedestrian Trajectory Prediction

統合経路計画システム - Social ForceモデルとSocial-GANによる歩行者軌道予測、Frenet座標系を用いた自動運転車の安全な経路計画を実現するシミュレーション環境

## 概要

このプロジェクトは、以下の3つのコンポーネントを統合します：

1. **Social Force Model**: 歩行者の動きをシミュレート（Ground Truth生成）
2. **Social-GAN**: 歩行者の未来軌道を予測
3. **Frenet Optimal Trajectory**: 予測された歩行者を回避する安全な経路を計画

## システムアーキテクチャ

```
[Social Force Simulator] → [Pedestrian Observer] → [Social-GAN Predictor]
                                                            ↓
                                                    [Predicted Trajectories]
                                                            ↓
[Ego Vehicle State] ← [Frenet Planner] ← [Coordinate Converter]
```

### 時間整合性と衝突判定の挙動
- 観測はシミュレーション `dt` に依らず SGAN の想定サンプリング 0.4s 間隔でダウンサンプリングされます。
- SGAN 出力はプランナ/シミュレーション `dt`（デフォルト 0.1s）に線形補間され、5s の計画ホライゾンまで等速外挿して時間幅を揃えます。
- 衝突判定は動的障害物の「同時刻位置」のみを評価し、将来軌道を平坦化しません（過剰な停止・回避を防止）。

### パフォーマンスとロバスト性 (v1.1 Update)
- **高速化**: 衝突判定のベクトル化（NumPy Broadcasting）により、数百の障害物が存在しても 0.06ms 程度で判定可能です。
- **効率化**: 参照経路上の座標探索にキャッシュ付き局所探索を導入し、計算コストを O(1) に削減しました。
- **安定化**: SGANの予測が途切れた後（12ステップ以降）の外挿処理に速度制限（max 2.5m/s）を設け、非現実的な挙動を抑制しました。

### シミュレーションエンジン (v1.2 Update)
- **PySocialForce統合**: 簡易的な等速直線運動モデルを廃止し、`pysocialforce` による Social Force Model を標準採用しました。歩行者同士の回避行動に加え、**Ego車両を動的な障害物として認識することで、歩行者が車両を能動的に回避する相互作用**を実装しました。

### 評価と可視化 (v1.3 Update)
- **拡張メトリクス**: 従来の安全性指標に加え、**ADE/FDE** (予測精度), **Jerk** (乗り心地), **TTC** (衝突リスク) を評価指標に追加しました。
- **Dashboard**: シミュレーション結果を包括的に可視化する静的ダッシュボード生成機能 (`dashboard.png`) を実装しました。
- **Map Visualization**: 道路境界線、レーン、横断歩道などの地図情報をアニメーションとダッシュボードに描画し、状況把握を容易にしました。
- **Headless対応**: `visualization_enabled` フラグにより、可視化処理を完全にスキップして高速実行やサーバーサイド実行が可能になりました。

### 比較研究機能 (v1.4 Update - Prediction Modes)
- **予測モード比較**: 歩行者予測の影響を検証するために、3つのモードを切り替え可能です。
  - `cv`: 等速直線運動（Constant Velocity） - ベースライン
  - `lstm`: 相互作用を考慮しない単純なLSTM予測（SGAN w/o Pooling）
  - `sgan`: 相互作用を考慮したSocial-GAN予測（推奨）
- **ベンチマーク**: シナリオごとの安全性・効率性を一括比較するスクリプトを提供します。

## インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd integrated_path_planning

# 仮想環境の作成と有効化
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt

# パッケージのインストール（開発モード）
pip install -e .
```

## 学習済みモデルのダウンロード（必須）

Social-GAN 予測には学習済みモデルが必須です（定速フォールバックはありません）。以下で入手してください：

### 方法1: Pythonスクリプト（推奨）

```bash
# 基本モデルのみダウンロード
python scripts/download_sgan_models.py

# プーリングモデルも含めてダウンロード
python scripts/download_sgan_models.py --pooling
```

### 方法2: Bashスクリプト

```bash
bash scripts/download_sgan_models.sh
```

ダウンロードされるモデル例：
- `models/sgan-models/eth_8_model.pt` / `*_12_model.pt`
- `models/sgan-models/hotel_8.pt`
- `models/sgan-models/univ_8.pt`
- `models/sgan-models/zara1_8.pt`
- `models/sgan-models/zara2_8.pt`

モデルサイズ: 各モデル約5-10MB（合計数十MB）

## 使用方法

### 基本的な使い方

```python
from src.simulation.integrated_simulator import IntegratedSimulator
from src.config import load_config

# 設定ファイルの読み込み
config = load_config('scenarios/scenario_01.yaml')

# シミュレータの初期化
simulator = IntegratedSimulator(config)

# シミュレーションの実行
results = simulator.run(n_steps=100)

# 結果の保存と可視化（自動的にdashboard.pngが生成されます）
simulator.save_results()
```

### コマンドラインからの実行

#### 基本実行
```bash
python examples/run_simulation.py --scenario scenarios/scenario_01.yaml
```

#### 予測モードの切り替え (v1.4 Update)
```bash
# 等速直線運動 (ベースライン)
python examples/run_simulation.py --scenario scenarios/scenario_01.yaml --method cv

# 単純LSTM (相互作用なし)
python examples/run_simulation.py --scenario scenarios/scenario_01.yaml --method lstm

# Social-GAN (デフォルト)
python examples/run_simulation.py --scenario scenarios/scenario_01.yaml --method sgan
```

#### アニメーション生成（NEW! 🆕）
```bash
# GIFアニメーション生成
python examples/run_simulation.py \
    --scenario scenarios/scenario_01.yaml \
    --animate \
    --animation-format gif \
    --fps 10

# MP4アニメーション生成（高品質）
python examples/run_simulation.py \
    --scenario scenarios/scenario_02.yaml \
    --animate \
    --animation-format mp4 \
    --fps 20
```

#### アニメーションデモ
```bash
# 両フォーマット（GIF + MP4）を生成するデモ
python examples/demo_animation.py
```

### Pythonコードでアニメーション作成

```python
from src.simulation.integrated_simulator import IntegratedSimulator
from src.visualization import create_simple_animation
from src.config import load_config

# シミュレーション実行
config = load_config('scenarios/scenario_01.yaml')
simulator = IntegratedSimulator(config)
results = simulator.run(n_steps=150)

# アニメーション作成
create_simple_animation(
    results=results,
    output_path='output/my_animation.gif',
    show=True,  # 表示する
    show_predictions=True,  # 予測軌道を表示
    show_metrics=True,  # メトリクスを表示
    fps=10
)
```

### 予測モデルのベンチマーク (v1.4 Update)

3つの予測モード（CV, LSTM, SGAN）を同一シナリオで実行し、安全性指標（最小距離、衝突回数、TTC）と効率性指標を比較します。

```bash
python examples/benchmark_prediction.py --scenario scenarios/scenario_01.yaml
```

レポートは `output/benchmark/benchmark_report.md` に保存されます。

### 学習済みモデルの指定（必須）

シナリオYAMLでモデルパスを指定してください。未指定のまま実行すると `RuntimeError` で停止します。

```yaml
# scenarios/my_scenario.yaml
sgan_model_path: "models/sgan-models/eth_8_model.pt"
```

## プロジェクト構成

```
integrated_path_planning/
├── src/
│   ├── config/          # 設定管理
│   ├── core/            # 基本データ構造と座標変換
│   ├── pedestrian/      # Social Force統合と観測
│   ├── prediction/      # Social-GAN統合
│   ├── planning/        # Frenet経路計画
│   ├── simulation/      # 統合シミュレータ
│   └── visualization/   # 可視化
├── scenarios/           # シミュレーションシナリオ
├── models/              # 学習済みモデル
├── tests/               # ユニットテスト
└── examples/            # 使用例
```

## シナリオ

複数のシナリオが用意されています：

1. **scenario_01.yaml**: 歩行者との交差シナリオ
2. **scenario_02.yaml**: 狭い通路でのすれ違いシナリオ
3. **scenario_03.yaml**: 曲線路合流＋歩行者すれ違い
4. **scenario_04.yaml**: 多波交差の混雑シナリオ
5. **scenario_05.yaml**: 静的障害で狭窄した通路を通過
6. **scenario_06.yaml**: 斜め横断歩行者
7. **scenario_07.yaml**: 高速車両とまばらな歩行者
8. **scenario_08.yaml**: 複数の静的障害物と双方向流
9. **scenario_09.yaml**: 混雑した狭い通路
10. **scenario_10.yaml**: 交通量の多い交差点

## 主な設定項目（YAML）

- 時間: `dt`, `total_time`, 観測/予測長 `obs_len`, `pred_len`
- Ego: `ego_initial_state`, `ego_target_speed`, `ego_max_speed`, `ego_max_accel`, `ego_max_curvature`
- 安全パラメータ: `ego_radius`, `ped_radius`, `obstacle_radius`, `safety_buffer`
- プランナ重み（任意上書き）: `k_j`, `k_t`, `k_d`, `k_s_dot`, `k_lat`, `k_lon`
- 経路: `reference_waypoints_x`, `reference_waypoints_y`
- 歩行者: `ped_initial_states`, `ped_groups`
- 障害物: `static_obstacles`（矩形: `[x_min, x_max, y_min, y_max]`）
- 予測モデル: `sgan_model_path`（必須。未設定の場合はエラー）
- デバイス/出力: `device`, `output_path`, `visualization_enabled`

## 保存される出力

`simulator.save_results()` は以下を `trajectory.npz` に保存します（object配列含む）:
- 時系列: `times`
- Ego: `ego_x`, `ego_y`, `ego_v`
- 安全指標: `min_distances`, `ttc`
- 歩行者: `ped_positions`, `ped_velocities`, `ped_goals`
- 予測: `predicted_trajectories`
- 計画軌跡: `planned_x`, `planned_y`, `planned_v`, `planned_a`, `planned_yaw`, `planned_cost`

## テスト

```bash
pytest tests/
```

## 評価指標

- **安全性**: 最小距離（歩行者との最短距離）
- **効率性**: 目標到達時間
- **快適性**: 最大加速度、最大ジャーク

## ライセンス

MIT License

## 参考文献

1. Helbing, D., & Molnár, P. (1995). Social force model for pedestrian dynamics.
2. Gupta, A., et al. (2018). Social GAN: Socially Acceptable Trajectories with GANs.
3. Werling, M., et al. (2010). Optimal trajectory generation for dynamic street scenarios in a Frenet Frame.
