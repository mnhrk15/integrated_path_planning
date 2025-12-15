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
config = load_config('scenarios/scenario_01_crossing.yaml')

# シミュレータの初期化
simulator = IntegratedSimulator(config)

# シミュレーションの実行
results = simulator.run(n_steps=100)

# 結果の保存と可視化
simulator.save_results()
simulator.visualize()
```

### コマンドラインからの実行

#### 基本実行
```bash
python examples/run_simulation.py --scenario scenarios/scenario_01_crossing.yaml
```

#### アニメーション生成（NEW! 🆕）
```bash
# GIFアニメーション生成
python examples/run_simulation.py \
    --scenario scenarios/scenario_01_crossing.yaml \
    --animate \
    --animation-format gif \
    --fps 10

# MP4アニメーション生成（高品質）
python examples/run_simulation.py \
    --scenario scenarios/scenario_02_corridor.yaml \
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
config = load_config('scenarios/scenario_01_crossing.yaml')
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

1. **scenario_01_crossing.yaml**: 歩行者との交差シナリオ
2. **scenario_02_corridor.yaml**: 狭い通路でのすれ違いシナリオ
3. **scenario_03_curved_merge.yaml**: 曲線路合流＋歩行者すれ違い
4. **scenario_04_multi_crossing.yaml**: 多波交差の混雑シナリオ
5. **scenario_05_blocked_corridor.yaml**: 静的障害で狭窄した通路を通過

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
