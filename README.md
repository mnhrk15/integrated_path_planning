# Integrated Path Planning with Pedestrian Trajectory Prediction

歩行者軌道予測（Social-GAN）と Frenet 座標系経路計画を統合した自動運転シミュレーション環境。以下の3コンポーネントを統合しています:

1. **Social Force Model**（pysocialforce）: 歩行者挙動のシミュレーション。Ego 車両を斥力源として歩行者が能動的に回避する相互作用を含む
2. **Social-GAN**: 社会的相互作用を考慮した歩行者軌道予測（`cv` / `lstm` / `sgan` の3モード切替）
3. **Frenet Optimal Trajectory**: 参照経路に沿った候補経路のベクトル化生成と、予測歩行者を回避する衝突フリー経路の選択

## アーキテクチャ

1ステップのデータフロー（統合点は `src/simulation/integrated_simulator.py` の `IntegratedSimulator.step()`）:

```
PedestrianSimulator (pysocialforce; Ego車両を斥力源として歩行者が回避)
  → PedestrianObserver (シミュレーション dt に依らず SGAN 想定の 0.4s 間隔で観測履歴を保持)
  → TrajectoryPredictor (Social-GAN; 12ステップ@0.4s の軌道を予測)
  → 予測後処理 (プランナ dt=0.1s へ線形補間し、計画ホライゾン max_t まで等速外挿)
  → FrenetPlanner (cubic spline 参照経路上で quintic polynomial 候補を生成、コスト最小の衝突フリー経路を選択)
  → FailSafeStateMachine (NORMAL→CAUTION→EMERGENCY で制約を段階的に緩和、最終的に緊急停止)
```

設計上の要点:

- **時間整合**: 観測はシミュレーション `dt` に依らず 0.4s 間隔にダウンサンプリングされ、予測出力はプランナ `dt`（0.1s）へ線形補間・`max_t`（デフォルト 5.0s）まで等速外挿されます。
- **衝突判定は「同時刻位置」のみ評価**します（将来軌道を平坦化しない）。予測失敗時のフォールバックも必ず計画ホライゾン分の時系列を生成します。
- **予測失敗時のフォールバック**: Social-GAN 予測が失敗した場合は等速直線モデルの軌道で継続します（5回連続で失敗すると `RuntimeError` で停止）。
- **フェイルセーフ**: 計画失敗時は状態マシンが加速度等の制約を段階的に緩和しながら同一ステップ内で再計画し、それでも経路が見つからなければ安全に緊急停止します。
- **分布対応計画（オプション）**: `distribution_aware_planning: true` で SGAN の全 `num_samples` サンプルに対する chance-constrained 衝突判定に切替わります（`chance_epsilon` = 許容衝突サンプル割合、0.0 = worst-case）。デフォルトは単一代表サンプルです。
- **ウォームアップ**: t=0 で観測履歴をプリロール生成するため、開始直後から SGAN 予測が有効です。ゴール 2m 以内に到達すると `total_time` を待たず自動終了します。

## セットアップ

Python 3.12 で動作確認済み。

```bash
git clone https://github.com/mnhrk15/integrated_path_planning.git
cd integrated_path_planning

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt   # pysocialforce・torch 等の必須依存を含む
pip install -e .
```

MP4 アニメーションを生成する場合のみ ffmpeg バイナリが必要です（macOS: `brew install ffmpeg` / Ubuntu: `sudo apt-get install ffmpeg`。GIF のみなら不要）。

### 学習済みモデルのダウンロード（必須）

`lstm` / `sgan` モードには学習済み Social-GAN モデルが必須です。`--pooling` 付きで両モデル群を取得してください:

```bash
python scripts/download_sgan_models.py --pooling
```

- `models/sgan-models/` — Pooling なし（`--method lstm` 用）
- `models/sgan-p-models/` — Pooling あり（`--method sgan` 用）

各ディレクトリに `{eth,hotel,univ,zara1,zara2}_{8,12}_model.pt` の10ファイル（各 5〜10MB）が入ります。使用モデルはシナリオ YAML の `sgan_model_path` で指定します。未指定・パス不在の場合は設定読み込み時に `ConfigValidationError` で停止します。

## 使い方

### シミュレーション実行

```bash
python examples/run_simulation.py --scenario scenarios/scenario_01.yaml --method sgan
```

| オプション | 既定値 | 説明 |
|---|---|---|
| `--scenario` | `scenarios/scenario_01.yaml` | シナリオ設定ファイル |
| `--method` | YAML の `prediction_method` | 予測モード `cv` / `lstm` / `sgan`（モデルディレクトリも自動切替） |
| `--steps` | YAML の `total_time / dt` | シミュレーションステップ数の上書き |
| `--output` | YAML の `output_path` | 出力ディレクトリの上書き |
| `--seed` | なし | 乱数シード（再現性のため `metrics_report.txt` / `metrics_summary.csv` に記録） |
| `--animate` | off | アニメーション（GIF/MP4）を生成 |
| `--animation-format` | `gif` | `gif` または `mp4` |
| `--fps` | `10` | アニメーションのフレームレート |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### 予測モード

| モード | モデル | 説明 |
|---|---|---|
| `cv` | 不要 | 等速直線運動（ベースライン） |
| `lstm` | `models/sgan-models/` | 相互作用を考慮しない予測（SGAN w/o Pooling） |
| `sgan` | `models/sgan-p-models/` | 相互作用を考慮した Social-GAN 予測 |

`--method` を指定すると `sgan_model_path` のファイル名を保ったままディレクトリが自動で切り替わります（切替先が存在しない場合は `FileNotFoundError`）。

### アニメーション生成

```bash
python examples/run_simulation.py --scenario scenarios/scenario_01.yaml \
    --animate --animation-format gif --fps 10
```

`simulation.{gif,mp4}`（メトリクス付き）と `simulation_simple.{gif,mp4}`（マップのみ）の2本が出力されます。予測の不確実性は `num_samples > 1`（推奨: 20）のとき複数サンプルの予測軌道が半透明の「雲」として描画されます（アニメーションのみ。静的ダッシュボードでは非表示）。

### Python API

```python
from src.config import load_config
from src.simulation.integrated_simulator import IntegratedSimulator

config = load_config('scenarios/scenario_01.yaml')
simulator = IntegratedSimulator(config)
results = simulator.run()          # n_steps を渡さなければ total_time / dt 分実行
simulator.save_results()           # config.output_path に成果物一式を保存
```

### ベンチマーク

```bash
# 3手法（cv/lstm/sgan）を同一シナリオ・同一シードで比較
python examples/benchmark_prediction.py --scenario scenarios/scenario_01.yaml
# → output/benchmark/<シナリオ名>/benchmark_report.md

# 複数シードの統計比較（mean±std、乗り心地指標含む）
python examples/run_statistical_benchmark.py
# → output/statistical_benchmark/{all_runs.csv, summary_stats.csv, latex_table.txt}
```

### テスト

```bash
.venv/bin/python -m pytest tests/
```

`tests/manual_test_headless.py` と `tests/benchmark_collision.py` は pytest 対象外の手動実行スクリプトです。

## シナリオ

| ファイル | 内容 |
|---|---|
| `scenarios/scenario_01.yaml` | 歩行者との交差（横断歩道） |
| `scenarios/scenario_02.yaml` | 狭い通路でのすれ違い |
| `scenarios/scenario_03.yaml` | 交差点での右折（Yielding） |

各シナリオには `_cv` / `_lstm` サフィックスの派生版があります（base との差分は `prediction_method` と `output_path` のみ。base 自体が `sgan`）。`scenarios/rq1b/` は研究用スクリプトが使う機械生成コピーです。

## 主要設定（YAML）

全設定項目とデフォルト値の**単一情報源は `src/config/__init__.py` の `SimulationConfig` dataclass** です。読み込み時に `validate_config()` が自動検証し、不正な値や未知のキーはエラーになります。主要カテゴリ:

- **時間・予測**: `dt`（0.1）、`total_time`、`obs_len`（8）、`pred_len`（8。同梱シナリオは 12）、`num_samples`、`single_select`（`medoid` / `draw`）
- **Ego 車両**: `ego_initial_state`（`[x, y, yaw, v, a]`）、`ego_target_speed`、`ego_max_speed`、`ego_max_accel`、`ego_max_curvature`（0.2）、`ego_footprint`（`circle` / `multi_circle`）、`vehicle_length` / `vehicle_width`
- **安全半径**: `ego_radius`、`ped_radius`（プランナ用マージン）、`obstacle_radius`
- **経路・環境**: `reference_waypoints_x` / `reference_waypoints_y`（2点以上）、`ped_initial_states`（`[x, y, vx, vy, gx, gy]`）、`ped_groups`、`static_obstacles`（矩形 `[x_min, x_max, y_min, y_max]`）、`map_config`
- **プランナ**: 横方向サンプリング `d_road_w` / `max_road_width`、時間ホライゾン `min_t` / `max_t`、速度サンプリング `d_t_s` / `n_s_sample`、コスト重み `k_j` / `k_t` / `k_d` / `k_s_dot` / `k_lat` / `k_lon`
- **分布対応計画**: `distribution_aware_planning`、`chance_epsilon`、`collision_margin_inflation`
- **状態マシン**: 予防トリガ `state_machine_trigger_clearance_caution` / `state_machine_trigger_time_headway`、復帰ゲート `state_machine_recover_clearance_{caution,emergency}`、速度エンベロープ `state_machine_envelope_decel` / `state_machine_envelope_standoff`、制約緩和倍率 `state_machine_{caution,emergency}_accel_multiplier` / `state_machine_caution_speed_multiplier` / `state_machine_emergency_lat_accel_multiplier`
- **Social Force**: `social_force_params`（ドット記法の辞書）。主要キーは `ego_repulsion.sigma`（Ego→歩行者斥力の減衰距離）、`ego_repulsion.v0`（同・強度）、`agent_radius`。その他のキーは pysocialforce の設定にそのまま渡されます
- **予測・実行**: `prediction_method`、`sgan_model_path`（`lstm`/`sgan` で必須）、`device`（`cpu` / `cuda` / `mps`）、`visualization_enabled`、`output_path`

## 出力ファイル

`save_results()` は `output_path`（例: `output/scenario_01/`）に以下を保存します:

| ファイル | 条件 | 内容 |
|---|---|---|
| `trajectory.npz` | 常時 | 時系列データ（Ego 状態・歩行者位置・予測・計画経路・安全指標）。object 配列を含むため読み込みは `np.load(..., allow_pickle=True)` |
| `metrics_summary.csv` | 常時 | メトリクス集計 1 行（実行ごとに上書き） |
| `metrics_report.txt` | 常時 | 全設定値と詳細メトリクスの可読レポート |
| `dashboard.png` / `simulation.png` | `visualization_enabled: true` | 統合ダッシュボード / 軌跡静止画 |
| `simulation.{gif,mp4}` / `simulation_simple.{gif,mp4}` | `--animate` | アニメーション2種 |

ベンチマーク実行時は `visualization_enabled: false` でヘッドレス実行するのが前提です（`src/visualization/` を完全スキップ）。

主な評価指標: 安全性（最小距離・衝突・TTC）、予測精度（標準 ADE/FDE = 0.4s 間隔のシーン単位 best-of-N、計画用 `planning_ade`/`planning_fde` = プランナ入力軌道のローリング評価）、効率性（到達時間・平均速度）、快適性（最大加速度・ジャーク）。

## カスタムシナリオ

最小構成の例:

```yaml
# my_scenario.yaml
dt: 0.1
total_time: 20.0
pred_len: 12
num_samples: 20

ego_initial_state: [0.0, 0.0, 0.0, 5.0, 0.0]   # [x, y, yaw, v, a]
ego_target_speed: 6.0
ego_max_speed: 10.0

reference_waypoints_x: [0.0, 20.0, 40.0, 60.0]
reference_waypoints_y: [0.0, 5.0, 5.0, 0.0]

ped_initial_states:
  - [30.0, 3.0, -0.5, 0.0, 30.0, -3.0]          # [x, y, vx, vy, gx, gy]
ped_groups: [[0]]

social_force_params:
  ego_repulsion.sigma: 0.7
  ego_repulsion.v0: 3.5

sgan_model_path: models/sgan-p-models/zara1_12_model.pt
output_path: output/my_scenario
```

未指定のキーは `SimulationConfig` のデフォルトが使われます。不正な値（負の `dt`、`min_t >= max_t` など）や未知のキーは読み込み時に `ConfigValidationError` / `ValueError` になり、問題箇所がメッセージに列挙されます。

## トラブルシューティング

- **`ConfigValidationError` が出る**: メッセージに列挙された項目を修正してください（例: `sgan_model_path is required when prediction_method is 'sgan'` → モデルをダウンロードしてパスを設定）。
- **モデルファイルがない**: `python scripts/download_sgan_models.py --pooling` を実行。`--method lstm` は `models/sgan-models/`、`--method sgan` は `models/sgan-p-models/` を参照します。
- **MP4 生成に失敗する**: ffmpeg バイナリをインストールするか、`--animation-format gif` を使用してください（GIF は pillow のみで生成可）。
- **GPU を使いたい**: YAML で `device: "cuda"`（NVIDIA）または `device: "mps"`（Apple Silicon）を指定。デフォルトは `cpu`。
- **ログに "Prediction failed" が出る**: 等速直線モデルに自動フォールバックして継続します。頻発する場合は `sgan_model_path` と `obs_len` の設定を確認してください。
- **再計画が頻発する**: 状態マシンが制約を緩和して再計画している状態です。`d_road_w` を小さくする・`max_road_width` を大きくする・状態マシンの倍率パラメータを調整するなどで探索を広げられます。

## プロジェクト構成

```
integrated_path_planning/
├── src/
│   ├── config/          # 設定管理（SimulationConfig = 設定の単一情報源）
│   ├── core/            # 基本データ構造・座標変換・状態マシン
│   ├── pedestrian/      # 観測履歴の管理（PedestrianObserver）
│   ├── prediction/      # Social-GAN 統合（本体は sgan_vendor/ にベンダリング）
│   ├── planning/        # Frenet 経路計画
│   ├── simulation/      # 統合シミュレータ・歩行者シミュレータ
│   ├── visualization/   # 可視化（ダッシュボード・アニメーション）
│   ├── calibration/     # SFM パラメータ較正（研究用）
│   └── datasets/        # 実データローダ ETH/UCY・VCI（研究用）
├── scenarios/           # シミュレーションシナリオ
├── models/              # 学習済み Social-GAN モデル（ダウンロードで取得）
├── examples/            # 実行スクリプト（run_rq* / plot_* 等は修論研究用）
├── scripts/             # モデル・データ取得等のユーティリティ
├── tests/               # pytest テストスイート
└── docs/                # 研究記録・レビュー文書
```

## ライセンス

MIT License（[LICENSE](LICENSE) を参照）。`src/prediction/sgan_vendor/` は [Social-GAN](https://github.com/agrimgupta92/sgan)（MIT License）のベンダリングです。

## 参考文献

1. Helbing, D., & Molnár, P. (1995). Social force model for pedestrian dynamics.
2. Gupta, A., et al. (2018). Social GAN: Socially Acceptable Trajectories with GANs.
3. Werling, M., et al. (2010). Optimal trajectory generation for dynamic street scenarios in a Frenet Frame.
