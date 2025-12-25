# Quick Start Guide

このガイドでは、統合経路計画システムの基本的な使い方を説明します。

## インストール

### 1. リポジトリのクローン

```bash
cd integrated_path_planning
```

### 2. 仮想環境の作成

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 依存パッケージのインストール

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**追加パッケージ（オプション）:**

```bash
# PySocialForce（高精度な歩行者シミュレーション）
pip install pysocialforce

# アニメーション用
pip install pillow  # GIF生成
pip install ffmpeg-python  # MP4生成（要ffmpegバイナリ）
```

**ffmpegのインストール（MP4生成用）:**

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# https://ffmpeg.org/download.html からダウンロード
```

### 4. パッケージのインストール（開発モード）

```bash
pip install -e .
```

### 5. Social-GAN学習済みモデルのダウンロード（必須）

**推奨方法:**

```bash
# 基本モデルをダウンロード
python scripts/download_sgan_models.py

# プーリングモデルも含める場合
python scripts/download_sgan_models.py --pooling
```

**または、Bashスクリプトで:**

```bash
bash scripts/download_sgan_models.sh
```

**ダウンロードされるモデル:**
- ETH, HOTEL, UNIV, ZARA1, ZARA2データセット
- 予測長: 8ステップ (推奨)
- 合計サイズ: 約50-100MB

**注意:** `sgan_model_path` に学習済みモデルを指定することが必須です。未指定のまま実行すると `RuntimeError` で停止します。

## 基本的な使い方

### シナリオ1: 歩行者との交差

```bash
python examples/run_simulation.py --scenario scenarios/scenario_01.yaml
```

### シナリオ2: 狭い通路

```bash
python examples/run_simulation.py --scenario scenarios/scenario_02.yaml
```

### カスタムオプション

```bash
# ステップ数を指定
python examples/run_simulation.py --scenario scenarios/scenario_01.yaml --steps 100

# 出力ディレクトリを指定
python examples/run_simulation.py --scenario scenarios/scenario_01.yaml --output my_results

# ログレベルを変更
python examples/run_simulation.py --scenario scenarios/scenario_01.yaml --log-level DEBUG

# プランナ重み・安全パラメータを上書き（例: YAMLを複製して編集）
# k_j, k_t, k_d, k_s_dot, k_lat, k_lon, ego_radius, ped_radius, obstacle_radius
# min_t, max_t, d_t_s, n_s_sample (プランナ時間ホライゾン)
# state_machine_* (状態マシンパラメータ)
```

### アニメーション生成（NEW! 🆕）

**GIFアニメーション:**

```bash
python examples/run_simulation.py \
    --scenario scenarios/scenario_01.yaml \
    --animate \
    --animation-format gif \
    --fps 10
```

**MP4アニメーション（高品質）:**

```bash
python examples/run_simulation.py \
    --scenario scenarios/scenario_02.yaml \
    --animate \
    --animation-format mp4 \
    --fps 20
```

> 保存が失敗した場合は自動で1回リトライし、失敗時は後処理を行います。`pillow`(GIF) または `ffmpeg`(MP4) のインストールを確認してください。

**デモスクリプト（GIF + MP4両方生成）:**

```bash
python examples/demo_animation.py
```

**生成される内容:**
- 自車と歩行者の軌跡アニメーション
- 予測軌道の可視化
- リアルタイムメトリクス（速度、距離）
- カスタマイズ可能な描画設定

## 出力

シミュレーション実行後、以下のファイルが生成されます：

```
output/
├── trajectory.npz        # 軌道データ（NumPy形式）
├── simulation.png        # 静的可視化結果
└── simulation.gif        # アニメーション（--animateオプション使用時）
    # or simulation.mp4
```

**アニメーション機能の特徴:**
- 自車の軌跡（青色のトレイル）
- 歩行者の動き（赤色の円）
- 予測軌道の可視化（オレンジ色の線）
- リアルタイムメトリクス（速度、最小距離）
- 高品質エクスポート（GIF/MP4）

## 時間スケールと衝突判定の注意点
- 観測はシミュレーション `dt` に依らず SGAN 想定の 0.4s 間隔で蓄積されます（内部でダウンサンプリング）。
- SGAN 出力はシミュレーション `dt`（例: 0.1s）に線形補間され、設定可能な計画ホライゾン（デフォルト: `max_t`=5.0s）まで等速外挿されます。
- 衝突判定は動的障害物の同時刻位置を参照し、未来軌道を平坦化せず時間整合でチェックします。

### データの読み込み

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results
data = np.load('output/trajectory.npz')
times = data['times']
ego_x = data['ego_x']
ego_y = data['ego_y']
ego_v = data['ego_v']
min_distances = data['min_distances']

# Plot velocity over time
plt.figure()
plt.plot(times, ego_v)
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Ego Vehicle Velocity')
plt.grid(True)
plt.show()
```

## カスタムシナリオの作成

新しいシナリオを作成するには、YAMLファイルを作成します：

```yaml
# my_scenario.yaml
dt: 0.1
total_time: 20.0

obs_len: 8
pred_len: 8
num_samples: 20

ego_initial_state: [0.0, 0.0, 0.0, 5.0, 0.0]
ego_target_speed: 6.0
ego_max_speed: 10.0
ego_max_accel: 2.0
ego_max_curvature: 1.0

reference_waypoints_x: [0.0, 20.0, 40.0, 60.0]
reference_waypoints_y: [0.0, 5.0, 5.0, 0.0]

ped_initial_states:
  - [30.0, 3.0, -0.5, 0.0, 30.0, -3.0]

ped_groups: [[0]]

static_obstacles: []

# Planner parameters
d_road_w: 0.3
max_road_width: 7.0
min_t: 4.0  # Minimum prediction time [s]
max_t: 5.0  # Maximum prediction time [s]
d_t_s: 1.39  # Target speed sampling width [m/s]
n_s_sample: 1  # Sampling number of target speed

# State machine parameters (optional)
state_machine_safe_distance_caution: 0.5  # Safe distance for CAUTION->NORMAL [m]
state_machine_safe_distance_emergency: 1.0  # Safe distance for EMERGENCY->CAUTION [m]
state_machine_caution_accel_multiplier: 1.5  # Acceleration multiplier in CAUTION
state_machine_caution_curvature_multiplier: 1.2  # Curvature multiplier in CAUTION
state_machine_caution_speed_multiplier: 0.8  # Speed multiplier in CAUTION
state_machine_emergency_accel_multiplier: 3.0  # Acceleration multiplier in EMERGENCY
state_machine_emergency_curvature_multiplier: 2.0  # Curvature multiplier in EMERGENCY

sgan_model_path: "models/sgan-p-models/zara1_8_model.pt"  # 必須: 学習済みモデルへのパス
device: "cpu"
visualization_enabled: true
output_path: "output/scenario_99"

# (Optional) Social Force Parameters Tuning
social_force_params:
  ped_repulsion.sigma: 0.7  # Interaction range
  ped_repulsion.v0: 3.5     # Interaction strength

# 注意: 設定ファイル読み込み時に自動的に検証が実行されます
# 不正な値（負の値、整合性のない設定など）があると ConfigValidationError が発生します

```

実行：

```bash
python examples/run_simulation.py --scenario my_scenario.yaml
```

## Pythonスクリプトでの使用

### 基本的な使用方法

```python
from src.config import load_config
from src.simulation.integrated_simulator import IntegratedSimulator

# Load configuration
config = load_config('scenarios/scenario_01.yaml')

# Create simulator
simulator = IntegratedSimulator(config)

# Run simulation
results = simulator.run(n_steps=200)

# Access results
for result in results:
    print(f"Time: {result.time:.2f}s")
    print(f"Ego position: ({result.ego_state.x:.2f}, {result.ego_state.y:.2f})")
    print(f"Minimum distance: {result.metrics['min_distance']:.2f}m")
    print()

# Save and visualize
simulator.save_results()
simulator.visualize()
```

### アニメーション作成（NEW! 🆕）

```python
from src.config import load_config
from src.simulation.integrated_simulator import IntegratedSimulator
from src.visualization import create_simple_animation

# Run simulation
config = load_config('scenarios/scenario_01.yaml')
simulator = IntegratedSimulator(config)
results = simulator.run(n_steps=150)

# Create animated GIF
create_simple_animation(
    results=results,
    output_path='output/my_animation.gif',
    show=True,  # Display animation
    show_predictions=True,  # Show predicted trajectories
    show_metrics=True,  # Show velocity/distance plots
    trail_length=30,  # Length of trajectory trail
    fps=10  # Frames per second
)

# Or create MP4 video
create_simple_animation(
    results=results,
    output_path='output/my_animation.mp4',
    show=False,
    show_predictions=True,
    show_metrics=True,
    fps=20
)
```

### 高度なアニメーションカスタマイズ

```python
from src.visualization import SimulationAnimator

# Create animator
animator = SimulationAnimator(
    results=results,
    figsize=(16, 10),  # Figure size
    dpi=100,  # Resolution
    interval=50  # ms between frames
)

# Create animation with custom settings
animator.create_animation(
    show_predictions=True,
    show_metrics=True,
    trail_length=50,  # Longer trail
    ego_color='blue',
    ped_color='red',
    pred_color='orange',
    save_path='output/custom_animation.gif',
    writer='pillow',
    fps=15
)

# Display
animator.show()
```

## 出力ファイル

- `output/<scenario>/trajectory.npz`  
  - `times`, `ego_x`, `ego_y`, `ego_v`  
  - `min_distances`, `ttc`  
  - `ped_positions`, `ped_velocities`, `ped_goals`  
  - `predicted_trajectories`  
  - `planned_x`, `planned_y`, `planned_v`, `planned_a`, `planned_yaw`, `planned_cost`  
- `output/<scenario>/simulation.png`: 軌跡・メトリクスの静止画像  
- `simulation.gif` / `simulation.mp4`（`--animate`指定時）: 失敗時は自動で1回リトライし、後処理を実施します。`pillow` (GIF) または `ffmpeg` (MP4) をインストールしてください。

## システムの堅牢性とエラーハンドリング

### 予測失敗時の動作

システムは予測失敗時にも安全に動作するよう設計されています：

- **自動フォールバック**: Social-GAN予測が失敗した場合、システムは自動的に等速直線運動モデルを使用して計画ホライゾン分の軌道を生成します。
- **時間次元の保持**: フォールバック時でも適切な時間次元を保持するため、将来の衝突判定が正常に機能します。
- **ログ出力**: 予測失敗時は警告メッセージが出力され、フォールバック処理が実行されたことが記録されます。

### 再計画の動作

状態マシンによる再計画処理は以下のように動作します：

- **試行回数制限**: 各ステップで最大3回まで再計画を試行します（設定可能）。
- **段階的な制約緩和**: NORMAL → CAUTION → EMERGENCY の順で制約を緩和しながら再計画を試行します。
- **緊急停止**: 最大試行回数に達しても経路が見つからない場合、安全に緊急停止します。
- **次のステップでのリセット**: 各ステップの終了時に再計画カウンターがリセットされ、次のステップで再度試行可能になります。

### エラーハンドリング

システムは以下のエラーに対して堅牢に動作します：

- **座標変換エラー**: 参照経路上の座標計算が失敗した場合、グローバル検索に自動的にフォールバックします。
- **配列インデックス範囲外**: メトリクス計算時に配列インデックスが範囲外にならないよう、事前にチェックを行います。
- **None値チェック**: 座標変換や経路計算で`None`が返された場合、適切なエラーメッセージと共に処理を中断します。

## トラブルシューティング

### 設定ファイルの検証エラー

**症状:** `ConfigValidationError`が発生する

**原因:** 設定ファイルに不正な値や整合性のない設定が含まれている

**解決策:**
エラーメッセージを確認して、指摘された問題を修正してください。よくある問題：

```yaml
# ❌ 間違い: 負の値
dt: -0.1

# ✅ 正しい: 正の値
dt: 0.1
```

```yaml
# ❌ 間違い: 速度の整合性が取れていない
ego_max_speed: 5.0
ego_target_speed: 10.0  # max_speed < target_speed

# ✅ 正しい: max_speed >= target_speed
ego_max_speed: 10.0
ego_target_speed: 5.0
```

```yaml
# ❌ 間違い: 時間ホライゾンの整合性が取れていない
min_t: 5.0
max_t: 4.0  # min_t > max_t

# ✅ 正しい: min_t < max_t
min_t: 4.0
max_t: 5.0
```

検証エラーの例：
```
ConfigValidationError: Configuration validation failed:
  - dt must be positive, got -0.1
  - ego_max_speed (5.0) must be >= ego_target_speed (10.0)
  - min_t (5.0) must be < max_t (4.0)
```

### Social-GANモデルが見つからない

**症状:** モデルファイルがないというエラー

**解決策（必須）:**
```bash
python scripts/download_sgan_models.py
```
```yaml
sgan_model_path: "models/sgan-p-models/zara1_8_model.pt"
```

### PySocialForceがインストールされていない

**症状:** "pysocialforce package not found" 警告

**解決策:**
```bash
pip install pysocialforce
```

**注意:** PySocialForceなしでも実行可能（簡易ダイナミクスにフォールバック）

### アニメーション生成エラー

**GIF生成エラー:**
```bash
pip install pillow
```

**MP4生成エラー:**
```bash
# 1. ffmpegバイナリをインストール
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# 2. Pythonパッケージをインストール
pip install ffmpeg-python
```

**代替方法:** GIF形式を使用
```bash
python examples/run_simulation.py --animate --animation-format gif
```

### GPU/MPSの使用

**CUDA（NVIDIA GPU）:**
```yaml
device: "cuda"
```

**Apple Silicon（MPS）:**
```yaml
device: "mps"
```

**CPU（デフォルト）:**
```yaml
device: "cpu"
```

### メモリ不足

観測長や予測長を減らす：

```yaml
obs_len: 4    # Default: 8
pred_len: 8   # Default: 12
```

ステップ数を減らす：
```bash
python examples/run_simulation.py --steps 50
```

### アニメーションが重い

**FPSを下げる:**
```bash
python examples/run_simulation.py --animate --fps 5
```

**trail_lengthを短くする:**
```python
create_simple_animation(results, trail_length=10)  # Default: 50
```

**解像度を下げる:**
```python
animator = SimulationAnimator(results, dpi=72)  # Default: 100
```

### 予測失敗時の動作

**症状:** ログに「Prediction failed」という警告が表示される

**原因:** Social-GANモデルの読み込みエラー、観測データの不足、またはモデルの推論エラー

**動作:** システムは自動的に等速直線運動モデルにフォールバックし、計画ホライゾン分の軌道を生成します。シミュレーションは継続され、安全な衝突判定が可能です。

**対処法:**
- モデルファイルのパスが正しいか確認: `sgan_model_path` が正しく設定されているか
- モデルファイルが存在するか確認: `models/sgan-p-models/` ディレクトリにモデルファイルがあるか
- 観測データが十分か確認: `obs_len` が適切に設定されているか（デフォルト: 8）

### 再計画が頻繁に発生する

**症状:** ログに「Planning failed」や「Re-planning」というメッセージが頻繁に表示される

**原因:** 障害物が多く、通常の制約では経路が見つからない状況

**動作:** 状態マシンが自動的に制約を緩和しながら再計画を試行します。各ステップで最大3回まで再試行し、それでも見つからない場合は緊急停止します。

**対処法:**
- プランナーのパラメータを調整: `d_road_w`（横方向サンプリング間隔）を小さくする、`max_road_width`を大きくする
- 状態マシンのパラメータを調整: `state_machine_caution_*_multiplier` や `state_machine_emergency_*_multiplier` を調整
- シナリオの難易度を下げる: 歩行者数や障害物を減らす

## テストの実行

```bash
pytest tests/ -v
```

## より詳しい情報

- プロジェクト構造: [README.md](README.md)
- API リファレンス: コード内のdocstring参照
- 設定オプション: [src/config/default_config.yaml](src/config/default_config.yaml)
