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
- 衝突判定: 動的障害物は時間軸を維持したまま評価し、パス時刻 `t` に対応する位置との距離のみを判定します（未来軌道の平坦化なし）。過剰なブレーキや回避を防ぎ、時間整合した安全判定を行います。

## 7. 設定可能な状態マシン (v3.4 Update)

Fail-Safe State Machineの動作をYAML設定で細かく調整できます。

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
