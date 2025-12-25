# 追加統合機能ガイド

このドキュメントでは、プロジェクトに追加された3つの主要機能について詳しく説明します。

## 目次

1. [Social-GAN学習済みモデル統合](#1-social-gan学習済みモデル統合)
2. [PySocialForce統合](#2-pysocialforce統合)  
3. [高度なアニメーション可視化](#3-高度なアニメーション可視化)
4. [静的障害物と安全パラメータ](#4-静的障害物と安全パラメータ)
5. [プランナコスト重みの上書き](#5-プランナコスト重みの上書き)
6. [時間整合した予測・衝突判定](#6-時間整合した予測衝突判定)
7. [設定可能な状態マシン](#7-設定可能な状態マシン)
8. [設定可能なプランナ時間ホライゾン](#8-設定可能なプランナ時間ホライゾン)
9. [設定値の自動検証](#9-設定値の自動検証)
10. [堅牢性とエラーハンドリングの改善 (v3.6)](#10-堅牢性とエラーハンドリングの改善-v36)

---

## 概要

PySocialForce/可視化など多くの機能はオプションですが、軌道予測には学習済みSocial-GANモデルが必須です（フォールバックなし）。

詳細は完全版ドキュメントを参照してください。

## 4. 静的障害物と安全パラメータ

- シナリオYAMLで`static_obstacles`（矩形: `[x_min, x_max, y_min, y_max]`）を指定すると、プランナの衝突判定に組み込まれます。
- 安全パラメータ（デフォルトは`default_config.yaml`参照）:
  - `ego_radius`, `ped_radius`, `obstacle_radius`, `safety_buffer`
- これらは膨張半径として衝突判定・安全メトリクス（`min_distance`, `ttc`）に利用されます。

## 5. プランナコスト重みの上書き

- Frenetプランナのコスト重みを設定で上書き可能です。
  - `k_j`, `k_t`, `k_d`, `k_s_dot`, `k_lat`, `k_lon`
- 未指定の場合は従来の定数が使われます。シナリオごとに調整してチューニングが可能です。

## 6. 時間整合した予測・衝突判定

- 観測: シミュレーション `dt` に依らず、SGAN 想定の 0.4s 間隔にダウンサンプリングして蓄積します。
- 予測: SGAN 出力（0.4s 間隔）をシミュレーション `dt`（例: 0.1s）に補間し、設定可能な計画ホライゾン（デフォルト: `max_t`=5.0s）まで等速外挿します。
- **予測失敗時のフォールバック (v3.6)**: Social-GAN予測が失敗した場合、システムは自動的に等速直線運動モデルを使用して計画ホライゾン分の軌道を生成します。現在の歩行者速度を使用して外挿し、適切な時間次元を保持するため、予測失敗時でも安全な衝突判定が可能です。
- 衝突判定: 動的障害物は時間軸を維持したまま評価し、パス時刻 `t` に対応する位置との距離のみを判定します（未来軌道の平坦化なし）。過剰なブレーキや回避を防ぎ、時間整合した安全判定を行います。

## 7. 設定可能な状態マシン (v3.4 & v3.6 Update)

Fail-Safe State Machineの動作をYAML設定で細かく調整できます。

### 再計画の試行回数制限 (v3.6)

状態マシンによる再計画処理には、無限ループを防止するための試行回数制限が設けられています：

- **デフォルト最大試行回数**: 各ステップで最大3回まで再計画を試行
- **動作**: 計画が失敗し、状態がより重要な状態（NORMAL → CAUTION → EMERGENCY）に遷移した場合、新しい制約で自動的に再計画を試行
- **緊急停止**: 最大試行回数に達しても経路が見つからない場合、安全に緊急停止
- **リセット**: 各ステップの終了時に再計画カウンターがリセットされ、次のステップで再度試行可能

### 設定パラメータ

```yaml
# 安全距離（状態遷移の閾値）
state_machine_safe_distance_caution: 0.5  # CAUTION→NORMAL遷移の安全距離 [m]
state_machine_safe_distance_emergency: 1.0  # EMERGENCY→CAUTION遷移の安全距離 [m]

# CAUTION状態の制約緩和係数
state_machine_caution_accel_multiplier: 1.5  # 加速度倍率（通常の1.5倍まで許容）
state_machine_caution_curvature_multiplier: 1.2  # 曲率倍率（通常の1.2倍まで許容）
state_machine_caution_speed_multiplier: 0.8  # 速度倍率（通常の80%に減速）

# EMERGENCY状態の制約緩和係数
state_machine_emergency_accel_multiplier: 3.0  # 加速度倍率（緊急停止のため大幅緩和）
state_machine_emergency_curvature_multiplier: 2.0  # 曲率倍率（緊急回避のため大幅緩和）
```

### 使用例

**保守的な設定（より安全重視）:**
```yaml
state_machine_safe_distance_caution: 1.0  # より長い安全距離を要求
state_machine_safe_distance_emergency: 2.0
state_machine_caution_speed_multiplier: 0.6  # より大幅に減速
```

**積極的な設定（より効率重視）:**
```yaml
state_machine_safe_distance_caution: 0.3  # より短い安全距離で回復
state_machine_caution_speed_multiplier: 0.9  # 減速を最小限に
state_machine_caution_accel_multiplier: 2.0  # より積極的な回避を許容
```

### 動作の流れ

1. **NORMAL状態**: 通常の制約で経路計画を実行
2. **計画失敗時 → CAUTION状態**: 制約を緩和（加速度・曲率を増加、速度を減少）して再計画
3. **CAUTIONでも失敗 → EMERGENCY状態**: さらに制約を緩和し、目標速度を0に設定して停止を試みる
4. **安全距離を確保できた場合**: より安全な状態に戻る

## 8. 設定可能なプランナ時間ホライゾン (v3.4 Update)

Frenet Plannerの予測時間範囲と速度サンプリングを設定ファイルで制御できます。

### 設定パラメータ

```yaml
# 時間ホライゾン
min_t: 4.0  # 最小予測時間 [s]（短いと反応が速いが、長期的な計画ができない）
max_t: 5.0  # 最大予測時間 [s]（長いと先読みできるが、計算コストが増加）

# 速度サンプリング
d_t_s: 1.39  # 目標速度サンプリング幅 [m/s]（5.0 km/h相当）
n_s_sample: 1  # 目標速度のサンプリング数（1の場合は目標速度のみ）
```

### 使用例

**短期的な計画（反応重視）:**
```yaml
min_t: 2.0
max_t: 3.0
```

**長期的な計画（先読み重視）:**
```yaml
min_t: 5.0
max_t: 8.0
```

**速度バリエーションを増やす（より多様な経路を探索）:**
```yaml
d_t_s: 2.78  # 10 km/h幅
n_s_sample: 3  # 目標速度±10km/hの範囲で3パターン
```

### パフォーマンスへの影響

- `max_t`を大きくすると、生成される候補経路数が増加し、計算時間が長くなります
- `n_s_sample`を増やすと、速度バリエーションが増え、経路探索の精度が向上しますが、計算コストも増加します
- デフォルト値（`min_t=4.0`, `max_t=5.0`, `n_s_sample=1`）は、精度と速度のバランスが取れた設定です

## 9. 設定値の自動検証 (v3.5 Update)

設定ファイル読み込み時に自動的に設定値の検証が実行され、不正な設定を早期に検出できます。

### 検証項目

**値の範囲チェック:**
- 正の値が必要なパラメータ（`dt`, `ego_max_accel`, `min_t`, `max_t`など）
- 非負の値が必要なパラメータ（`ego_target_speed`, `ego_max_speed`など）
- 特定の範囲内の値（`state_machine_caution_speed_multiplier`は0より大きく1以下）

**整合性チェック:**
- `ego_max_speed >= ego_target_speed`（最大速度は目標速度以上）
- `min_t < max_t`（最小予測時間は最大予測時間より小さい）
- `max_road_width >= d_road_w`（最大道路幅はサンプリング間隔以上）
- `state_machine_safe_distance_emergency >= state_machine_safe_distance_caution`（推奨）

**形式チェック:**
- `ego_initial_state`は5要素の配列（`[x, y, yaw, v, a]`）
- `ped_initial_states`の各要素は6要素の配列（`[x, y, vx, vy, gx, gy]`）
- `reference_waypoints_x`と`reference_waypoints_y`の長さが一致
- `reference_waypoints_x`は少なくとも2点必要
- `static_obstacles`の各要素は4要素の配列（`[x_min, x_max, y_min, y_max]`）
- `ped_groups`のインデックスが有効な範囲内

**存在確認:**
- `sgan_model_path`で指定されたファイルが存在する（`sgan`または`lstm`モードの場合）
- `device`が有効な値（`'cpu'`, `'cuda'`, `'mps'`のいずれか）

### 使用例

**正常な設定:**
```python
from src.config import load_config

# 設定ファイルを読み込むと自動的に検証が実行される
config = load_config('scenarios/scenario_01.yaml')
# 検証が成功すると、設定が読み込まれる
```

**不正な設定の場合:**
```python
from src.config import load_config, ConfigValidationError

try:
    config = load_config('scenarios/invalid_scenario.yaml')
except ConfigValidationError as e:
    print("設定検証エラー:")
    print(e)
    # 出力例:
    # Configuration validation failed:
    #   - dt must be positive, got -0.1
    #   - ego_max_speed (5.0) must be >= ego_target_speed (10.0)
    #   - min_t (5.0) must be < max_t (4.0)
```

### エラーメッセージの例

検証に失敗すると、`ConfigValidationError`が発生し、すべての問題点がリストアップされます：

```
ConfigValidationError: Configuration validation failed:
  - dt must be positive, got -0.1
  - total_time must be positive, got -30.0
  - ego_max_speed (5.0) must be >= ego_target_speed (10.0)
  - min_t (5.0) must be < max_t (4.0)
  - reference_waypoints_x (1) must have at least 2 points
  - reference_waypoints_x (2) and reference_waypoints_y (3) must have the same length
  - ped_initial_states[0] must have 6 elements [x, y, vx, vy, gx, gy], got 4
  - Pedestrian group index 5 is out of range [0, 3]
  - static_obstacles[0]: x_min (10.0) must be < x_max (5.0)
  - sgan_model_path does not exist: models/nonexistent_model.pt
  - device must be one of ['cpu', 'cuda', 'mps'], got 'gpu'
```

### 利点

1. **早期エラー検出**: 実行前に設定の問題を発見できる
2. **詳細なエラーメッセージ**: 問題箇所を特定しやすい
3. **デバッグ時間の短縮**: 実行時の予期しないエラーを防止
4. **設定の品質向上**: 不正な設定による予期しない動作を防止

## 10. 堅牢性とエラーハンドリングの改善 (v3.6 Update)

システムの堅牢性とエラーハンドリングを大幅に改善しました。

### 予測失敗時のフォールバック

Social-GAN予測が失敗した場合でも、システムは安全に動作し続けます：

- **自動フォールバック**: 予測失敗時に等速直線運動モデルを使用して計画ホライゾン分の軌道を自動生成
- **時間次元の保持**: フォールバック時でも適切な時間次元を保持するため、将来の衝突判定が正常に機能
- **現在速度の使用**: 現在の歩行者速度を使用して外挿し、現実的な軌道を生成
- **ログ出力**: 予測失敗時は警告メッセージが出力され、フォールバック処理が実行されたことが記録

**実装詳細:**
```python
# 予測失敗時、計画ホライゾン（max_t）に基づいて軌道を生成
plan_horizon = getattr(config, 'max_t', 5.0)  # デフォルト: 5.0秒
plan_horizon_steps = max(1, int(plan_horizon / dt))
dynamic_obstacles = np.zeros((n_peds, plan_horizon_steps, 2))
for step in range(plan_horizon_steps):
    t = (step + 1) * dt
    dynamic_obstacles[:, step, :] = current_positions + current_velocities * t
```

### 再計画の試行回数制限

状態マシンによる再計画処理に試行回数の上限を設け、無限ループを防止：

- **最大試行回数**: 各ステップで最大3回まで再計画を試行（デフォルト）
- **段階的な制約緩和**: NORMAL → CAUTION → EMERGENCY の順で制約を緩和しながら再計画
- **緊急停止**: 最大試行回数に達しても経路が見つからない場合、安全に緊急停止
- **自動リセット**: 各ステップの終了時に再計画カウンターがリセットされ、次のステップで再度試行可能

**動作フロー:**
1. 通常の経路計画を試行
2. 計画失敗 → 状態マシンが状態を更新（例: NORMAL → CAUTION）
3. 新しい制約で再計画を試行（試行回数: 1）
4. 再失敗 → さらに制約を緩和して再試行（試行回数: 2）
5. 最大試行回数に達した場合 → 緊急停止

### エラーハンドリングの強化

以下のエラーに対して堅牢に動作します：

**配列インデックス範囲外エラー:**
- `calculate_ade_fde()`で配列インデックスが範囲外にならないよう、事前にチェック
- 範囲外の場合は`valid_gt = False`として処理をスキップ

**座標変換エラー:**
- `find_nearest_point_on_path()`で`None`値が返された場合、グローバル検索に自動フォールバック
- それでも失敗した場合は明確なエラーメッセージと共に`ValueError`を発生

**パフォーマンス最適化:**
- 経路検証処理でジェネレータ式を使用することで、早期終了によるメモリ使用量とCPU時間を削減
- `any([...])`から`any(...)`への変更により、不要なリスト作成を回避

### 利点

1. **システムの安定性向上**: 予測失敗や異常な状況でもシステムがクラッシュしない
2. **無限ループの防止**: 再計画の試行回数制限により、システムが停止状態に陥ることを防止
3. **デバッグの容易さ**: 明確なエラーメッセージとログ出力により、問題の特定が容易
4. **パフォーマンス向上**: ジェネレータ式の使用により、メモリ使用量とCPU時間を削減
