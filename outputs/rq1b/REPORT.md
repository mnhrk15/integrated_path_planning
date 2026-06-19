# RQ1b 感度分析レポート

較正済み反応モデル下での robust/inflation/single 計画の再ベンチ（感度分析、外的検証ではない）。
全ラン cruise=3.0 m/s（RQ2 較正有効域 ~[0.4, 4.0] m/s 内＝5-6 m/s 外挿を回避。RQ2 limitation #2）。AVEC の 6/5/5 m/s 結果とは直接非比較で、同一 ~3 m/s の `avec` アームが域内再ベースライン。

## GT 反応モデル設定（σ/v0）

| gt_label | sigma | v0 | meaning |
|---|---|---|---|
| avec | per-scenario | per-scenario | AVEC per-scenario default (re-baseline) |
| calib | 1.156 | 1.681 | LOCO mean (canonical calibration) |
| calib_lo | 1.04 | 1.542 | -1SD (weakest avoidance) |
| calib_hi | 1.272 | 1.82 | +1SD (strongest avoidance) |

（avec の σ/v0 は各シナリオ YAML 値: S1/S3 σ0.7/v0 3.5, S2 σ0.3/v0 2.1。実効値は means.csv の min_dist 等とともに all_runs.csv の ego_repulsion_sigma/v0 列に記録。）

## 判定サマリ

- robust_gain_holds: 全シナリオ同時に robust を支配する inflation が無い（＝主張①保持）
- cv_danger_holds: CV single の衝突 > SGAN robust の衝突（＝主張②保持）
- lstm_danger_holds: LSTM single の衝突 > LSTM robust の衝突

| gt_label | robust_gain_holds | dominating_inflations | robust_collisions | cv_danger_holds | lstm_danger_holds |
|---|---|---|---|---|---|
| avec | True |  | 0 | True | True |
| calib | True |  | 0 | True | True |
| calib_lo | True |  | 0 | True | True |
| calib_hi | True |  | 0 | False | True |

> 注意: 集計 `cv_danger_holds`/`lstm_danger_holds` は衝突をシナリオ横断で合算するため、シナリオごとの内訳を隠す。主張②は必ず下記の per-scenario 分類で読むこと。主張①（`robust_gain_holds`）はシナリオ横断の集計でも頑健。

## 感度（GT 間で判定が反転するか）

- **robust_gain_holds**: 全 GT で不変（頑健）
- **cv_danger_holds**: 反転あり（較正に感度あり）
- **lstm_danger_holds**: 全 GT で不変（頑健）

## 主張② シナリオ別（per-scenario・汚染なし）

各 (GT, シナリオ) の rand 衝突数と分類（single-danger=分布なし single が衝突しつつ robust 2種は無衝突＝真の主張②信号／mixed=single≫robust>0＝主張②方向は残るが robust も非ゼロ／GT-artifact=robust≧single＝弁別でなく GT 自体の衝突／no-conflict=無衝突）:

| scenario | gt_label | cv_single | lstm_single | sgan_single_inf1.00 | lstm_robust_eps0.0 | sgan_robust_eps0.0 | class |
|---|---|---|---|---|---|---|---|
| scenario_01 | avec | 1 | 0 | 0 | 0 | 0 | single-danger |
| scenario_01 | calib | 1 | 1 | 0 | 0 | 0 | single-danger |
| scenario_01 | calib_lo | 0 | 1 | 0 | 0 | 0 | single-danger |
| scenario_01 | calib_hi | 0 | 1 | 1 | 0 | 0 | single-danger |
| scenario_02 | avec | 0 | 3 | 6 | 0 | 0 | single-danger |
| scenario_02 | calib | 0 | 0 | 0 | 0 | 0 | no-conflict |
| scenario_02 | calib_lo | 0 | 0 | 1 | 0 | 0 | single-danger |
| scenario_02 | calib_hi | 0 | 0 | 0 | 0 | 0 | no-conflict |
| scenario_03 | avec | 0 | 2 | 1 | 0 | 0 | single-danger |
| scenario_03 | calib | 1 | 3 | 3 | 1 | 1 | mixed |
| scenario_03 | calib_lo | 2 | 2 | 1 | 0 | 0 | single-danger |
| scenario_03 | calib_hi | 0 | 2 | 1 | 1 | 0 | mixed |

**読み筋（per-scenario・3シナリオとも干渉成立する修正シナリオ）**:
- **S1（密交差）**: 両 GT で single-danger が残る（主に cv＝盲目予測の本質的危険。協調的な実較正歩行者でも、ego が誤った単一軌道に commit すると回避できない）。
- **S2（狭路すれ違い）**: AVEC GT では single 衝突・robust 0（single-danger）→ **較正 GT では single も 0（no-conflict）＝主張②が消失**。狭路では協調的な実較正歩行者が single 計画の危険を解消する。
- **S3（右折 yield）**: AVEC GT では single-danger（single 衝突・robust 0）→ **較正 GT（弱い v0≈1.68）では交錯が厳しくなり single が大幅悪化する一方 robust も完全には守れない（mixed: single≫robust>0）**。single≫robust の方向（主張②）は残るが、弱い斥力が右折交錯を困難化し robust の安全余裕も低下する。

**主張②の結論（scenario-dependent）**: AVEC 反応モデルでは主張②（分布なし計画は危険）は3シナリオすべてで成立。**実較正（より協調的）反応モデル下では効果はシナリオ依存**＝狭路 S2 では消失（協調回避）、密交差 S1（盲目 cv）と右折 S3（弱斥力で交錯困難）では残る。「分布なし計画の危険度は反応モデルに依存する」が正直な結論で、AVEC sim の一律な危険性主張は手調整反応モデルに一部依存していた。robust 利得（主張①）は全 GT・全シナリオで頑健に生き残る。

## 付録: 平均指標（means.csv 抜粋）

### margin キャンペーン

| gt_label | scenario | condition | n | collisions | min_dist_mean | time_mean_cf | rms_jerk_mean | mean_accel_mean |
|---|---|---|---|---|---|---|---|---|
| avec | scenario_01 | sgan_robust_eps0.0 | 20 | 0 | 2.222 | 16.2 | 0.102 | 0.089 |
| avec | scenario_01 | sgan_single_inf1.00 | 20 | 0 | 1.853 | 15.31 | 0.124 | 0.145 |
| avec | scenario_01 | sgan_single_inf1.10 | 20 | 0 | 1.9 | 15.29 | 0.123 | 0.145 |
| avec | scenario_01 | sgan_single_inf1.20 | 20 | 0 | 1.942 | 15.33 | 0.121 | 0.144 |
| avec | scenario_01 | sgan_single_inf1.35 | 20 | 0 | 2.113 | 16.0 | 0.101 | 0.098 |
| avec | scenario_01 | sgan_single_inf1.50 | 20 | 0 | 2.177 | 15.94 | 0.108 | 0.108 |
| avec | scenario_02 | sgan_robust_eps0.0 | 20 | 0 | 1.861 | 17.68 | 0.087 | 0.072 |
| avec | scenario_02 | sgan_single_inf1.00 | 20 | 10 | 1.195 | 25.1 | 2.818 | 0.209 |
| avec | scenario_02 | sgan_single_inf1.10 | 20 | 0 | 1.325 | 26.39 | 2.262 | 0.167 |
| avec | scenario_02 | sgan_single_inf1.20 | 20 | 0 | 1.604 | 26.81 | 0.989 | 0.128 |
| avec | scenario_02 | sgan_single_inf1.35 | 20 | 0 | 1.803 | 23.74 | 0.393 | 0.121 |
| avec | scenario_02 | sgan_single_inf1.50 | 20 | 0 | 2.062 | 23.56 | 0.096 | 0.11 |
| avec | scenario_03 | sgan_robust_eps0.0 | 20 | 0 | 2.466 | 31.54 | 0.199 | 0.14 |
| avec | scenario_03 | sgan_single_inf1.00 | 20 | 2 | 1.455 | 53.91 | 1.196 | 0.087 |
| avec | scenario_03 | sgan_single_inf1.10 | 20 | 0 | 1.881 | 49.1 | 0.962 | 0.093 |
| avec | scenario_03 | sgan_single_inf1.20 | 20 | 0 | 2.386 | 49.51 | 0.932 | 0.091 |
| avec | scenario_03 | sgan_single_inf1.35 | 20 | 0 | 2.944 | 37.69 | 0.556 | 0.135 |
| avec | scenario_03 | sgan_single_inf1.50 | 20 | 0 | 2.816 | 31.54 | 0.449 | 0.145 |
| calib | scenario_01 | sgan_robust_eps0.0 | 20 | 0 | 2.242 | 16.2 | 0.103 | 0.089 |
| calib | scenario_01 | sgan_single_inf1.00 | 20 | 0 | 1.85 | 15.28 | 0.124 | 0.146 |
| calib | scenario_01 | sgan_single_inf1.10 | 20 | 0 | 1.9 | 15.29 | 0.123 | 0.147 |
| calib | scenario_01 | sgan_single_inf1.20 | 20 | 0 | 1.944 | 15.33 | 0.121 | 0.144 |
| calib | scenario_01 | sgan_single_inf1.35 | 20 | 0 | 2.127 | 15.97 | 0.102 | 0.101 |
| calib | scenario_01 | sgan_single_inf1.50 | 20 | 0 | 2.187 | 15.87 | 0.11 | 0.114 |
| calib | scenario_02 | sgan_robust_eps0.0 | 20 | 0 | 1.534 | 17.87 | 0.086 | 0.073 |
| calib | scenario_02 | sgan_single_inf1.00 | 20 | 0 | 1.238 | 29.04 | 1.265 | 0.114 |
| calib | scenario_02 | sgan_single_inf1.10 | 20 | 0 | 1.356 | 26.94 | 1.986 | 0.196 |
| calib | scenario_02 | sgan_single_inf1.20 | 20 | 0 | 1.499 | 30.98 | 1.78 | 0.172 |
| calib | scenario_02 | sgan_single_inf1.35 | 20 | 0 | 1.678 | 28.74 | 1.235 | 0.167 |
| calib | scenario_02 | sgan_single_inf1.50 | 20 | 1 | 1.765 | 32.08 | 1.248 | 0.161 |
| calib | scenario_03 | sgan_robust_eps0.0 | 20 | 0 | 2.452 | 31.48 | 0.141 | 0.139 |
| calib | scenario_03 | sgan_single_inf1.00 | 20 | 10 | 1.331 | 59.9 | 1.579 | 0.13 |
| calib | scenario_03 | sgan_single_inf1.10 | 20 | 0 | 1.829 | 53.13 | 0.903 | 0.079 |
| calib | scenario_03 | sgan_single_inf1.20 | 20 | 0 | 2.402 | 49.47 | 0.933 | 0.091 |
| calib | scenario_03 | sgan_single_inf1.35 | 20 | 0 | 2.962 | 38.22 | 0.605 | 0.128 |
| calib | scenario_03 | sgan_single_inf1.50 | 20 | 0 | 2.818 | 31.87 | 0.412 | 0.142 |
| calib_hi | scenario_01 | sgan_robust_eps0.0 | 10 | 0 | 2.29 | 16.23 | 0.104 | 0.089 |
| calib_hi | scenario_01 | sgan_single_inf1.00 | 10 | 0 | 1.865 | 15.28 | 0.124 | 0.145 |
| calib_hi | scenario_01 | sgan_single_inf1.10 | 10 | 0 | 1.937 | 15.28 | 0.123 | 0.145 |
| calib_hi | scenario_01 | sgan_single_inf1.20 | 10 | 0 | 1.991 | 15.37 | 0.119 | 0.141 |
| calib_hi | scenario_01 | sgan_single_inf1.35 | 10 | 0 | 2.167 | 15.96 | 0.102 | 0.101 |
| calib_hi | scenario_01 | sgan_single_inf1.50 | 10 | 0 | 2.223 | 15.77 | 0.112 | 0.12 |
| calib_hi | scenario_02 | sgan_robust_eps0.0 | 10 | 0 | 1.562 | 16.99 | 0.076 | 0.056 |
| calib_hi | scenario_02 | sgan_single_inf1.00 | 10 | 0 | 1.265 | 24.34 | 0.667 | 0.084 |
| calib_hi | scenario_02 | sgan_single_inf1.10 | 10 | 0 | 1.379 | 21.03 | 1.04 | 0.122 |
| calib_hi | scenario_02 | sgan_single_inf1.20 | 10 | 0 | 1.53 | 31.0 | 1.25 | 0.146 |
| calib_hi | scenario_02 | sgan_single_inf1.35 | 10 | 0 | 1.719 | 25.31 | 1.055 | 0.164 |
| calib_hi | scenario_02 | sgan_single_inf1.50 | 10 | 1 | 1.805 | 27.14 | 1.117 | 0.176 |
| calib_hi | scenario_03 | sgan_robust_eps0.0 | 10 | 0 | 2.547 | 31.76 | 0.143 | 0.135 |
| calib_hi | scenario_03 | sgan_single_inf1.00 | 10 | 2 | 1.528 | 59.9 | 1.163 | 0.087 |
| calib_hi | scenario_03 | sgan_single_inf1.10 | 10 | 0 | 1.913 | 57.22 | 0.833 | 0.066 |
| calib_hi | scenario_03 | sgan_single_inf1.20 | 10 | 0 | 2.441 | 49.84 | 0.939 | 0.089 |
| calib_hi | scenario_03 | sgan_single_inf1.35 | 10 | 0 | 2.853 | 36.82 | 0.517 | 0.131 |
| calib_hi | scenario_03 | sgan_single_inf1.50 | 10 | 0 | 2.804 | 30.84 | 0.438 | 0.152 |
| calib_lo | scenario_01 | sgan_robust_eps0.0 | 10 | 0 | 2.224 | 16.21 | 0.103 | 0.09 |
| calib_lo | scenario_01 | sgan_single_inf1.00 | 10 | 0 | 1.796 | 15.33 | 0.126 | 0.147 |
| calib_lo | scenario_01 | sgan_single_inf1.10 | 10 | 0 | 1.868 | 15.29 | 0.124 | 0.147 |
| calib_lo | scenario_01 | sgan_single_inf1.20 | 10 | 0 | 1.923 | 15.4 | 0.12 | 0.142 |
| calib_lo | scenario_01 | sgan_single_inf1.35 | 10 | 0 | 2.104 | 16.06 | 0.099 | 0.096 |
| calib_lo | scenario_01 | sgan_single_inf1.50 | 10 | 0 | 2.164 | 15.87 | 0.111 | 0.114 |
| calib_lo | scenario_02 | sgan_robust_eps0.0 | 10 | 0 | 1.459 | 16.82 | 0.078 | 0.055 |
| calib_lo | scenario_02 | sgan_single_inf1.00 | 10 | 1 | 1.212 | 26.24 | 1.709 | 0.136 |
| calib_lo | scenario_02 | sgan_single_inf1.10 | 10 | 0 | 1.338 | 26.33 | 2.44 | 0.196 |
| calib_lo | scenario_02 | sgan_single_inf1.20 | 10 | 0 | 1.519 | 28.56 | 1.975 | 0.176 |
| calib_lo | scenario_02 | sgan_single_inf1.35 | 10 | 0 | 1.731 | 28.2 | 1.102 | 0.162 |
| calib_lo | scenario_02 | sgan_single_inf1.50 | 10 | 0 | 1.913 | 24.6 | 0.762 | 0.165 |
| calib_lo | scenario_03 | sgan_robust_eps0.0 | 10 | 0 | 2.553 | 31.88 | 0.259 | 0.137 |
| calib_lo | scenario_03 | sgan_single_inf1.00 | 10 | 3 | 1.427 | 59.9 | 1.25 | 0.101 |
| calib_lo | scenario_03 | sgan_single_inf1.10 | 10 | 0 | 1.841 | 54.51 | 0.853 | 0.075 |
| calib_lo | scenario_03 | sgan_single_inf1.20 | 10 | 0 | 2.349 | 49.93 | 0.935 | 0.09 |
| calib_lo | scenario_03 | sgan_single_inf1.35 | 10 | 0 | 2.776 | 37.49 | 0.614 | 0.128 |
| calib_lo | scenario_03 | sgan_single_inf1.50 | 10 | 0 | 2.953 | 32.04 | 0.575 | 0.145 |

### rand キャンペーン（衝突数）

| gt_label | scenario | condition | n | collisions | min_dist_mean |
|---|---|---|---|---|---|
| avec | scenario_01 | cv_single | 20 | 1 | 1.507 |
| avec | scenario_01 | lstm_robust_eps0.0 | 20 | 0 | 3.054 |
| avec | scenario_01 | lstm_single | 20 | 0 | 1.82 |
| avec | scenario_01 | sgan_robust_eps0.0 | 20 | 0 | 2.147 |
| avec | scenario_01 | sgan_single_inf1.00 | 20 | 0 | 1.796 |
| avec | scenario_02 | cv_single | 20 | 0 | 1.333 |
| avec | scenario_02 | lstm_robust_eps0.0 | 20 | 0 | 1.78 |
| avec | scenario_02 | lstm_single | 20 | 3 | 1.347 |
| avec | scenario_02 | sgan_robust_eps0.0 | 20 | 0 | 1.571 |
| avec | scenario_02 | sgan_single_inf1.00 | 20 | 6 | 1.309 |
| avec | scenario_03 | cv_single | 20 | 0 | 1.749 |
| avec | scenario_03 | lstm_robust_eps0.0 | 20 | 0 | 2.729 |
| avec | scenario_03 | lstm_single | 20 | 2 | 1.916 |
| avec | scenario_03 | sgan_robust_eps0.0 | 20 | 0 | 2.5 |
| avec | scenario_03 | sgan_single_inf1.00 | 20 | 1 | 2.007 |
| calib | scenario_01 | cv_single | 20 | 1 | 1.486 |
| calib | scenario_01 | lstm_robust_eps0.0 | 20 | 0 | 3.087 |
| calib | scenario_01 | lstm_single | 20 | 1 | 1.8 |
| calib | scenario_01 | sgan_robust_eps0.0 | 20 | 0 | 2.149 |
| calib | scenario_01 | sgan_single_inf1.00 | 20 | 0 | 1.743 |
| calib | scenario_02 | cv_single | 20 | 0 | 1.421 |
| calib | scenario_02 | lstm_robust_eps0.0 | 20 | 0 | 1.928 |
| calib | scenario_02 | lstm_single | 20 | 0 | 1.447 |
| calib | scenario_02 | sgan_robust_eps0.0 | 20 | 0 | 1.695 |
| calib | scenario_02 | sgan_single_inf1.00 | 20 | 0 | 1.4 |
| calib | scenario_03 | cv_single | 20 | 1 | 1.701 |
| calib | scenario_03 | lstm_robust_eps0.0 | 20 | 1 | 2.645 |
| calib | scenario_03 | lstm_single | 20 | 3 | 1.891 |
| calib | scenario_03 | sgan_robust_eps0.0 | 20 | 1 | 2.527 |
| calib | scenario_03 | sgan_single_inf1.00 | 20 | 3 | 1.932 |
| calib_hi | scenario_01 | cv_single | 10 | 0 | 1.552 |
| calib_hi | scenario_01 | lstm_robust_eps0.0 | 10 | 0 | 3.276 |
| calib_hi | scenario_01 | lstm_single | 10 | 1 | 1.783 |
| calib_hi | scenario_01 | sgan_robust_eps0.0 | 10 | 0 | 2.23 |
| calib_hi | scenario_01 | sgan_single_inf1.00 | 10 | 1 | 1.765 |
| calib_hi | scenario_02 | cv_single | 10 | 0 | 1.468 |
| calib_hi | scenario_02 | lstm_robust_eps0.0 | 10 | 0 | 1.966 |
| calib_hi | scenario_02 | lstm_single | 10 | 0 | 1.522 |
| calib_hi | scenario_02 | sgan_robust_eps0.0 | 10 | 0 | 1.758 |
| calib_hi | scenario_02 | sgan_single_inf1.00 | 10 | 0 | 1.454 |
| calib_hi | scenario_03 | cv_single | 10 | 0 | 1.608 |
| calib_hi | scenario_03 | lstm_robust_eps0.0 | 10 | 1 | 2.859 |
| calib_hi | scenario_03 | lstm_single | 10 | 2 | 1.727 |
| calib_hi | scenario_03 | sgan_robust_eps0.0 | 10 | 0 | 2.549 |
| calib_hi | scenario_03 | sgan_single_inf1.00 | 10 | 1 | 1.898 |
| calib_lo | scenario_01 | cv_single | 10 | 0 | 1.501 |
| calib_lo | scenario_01 | lstm_robust_eps0.0 | 10 | 0 | 3.16 |
| calib_lo | scenario_01 | lstm_single | 10 | 1 | 1.722 |
| calib_lo | scenario_01 | sgan_robust_eps0.0 | 10 | 0 | 2.126 |
| calib_lo | scenario_01 | sgan_single_inf1.00 | 10 | 0 | 1.772 |
| calib_lo | scenario_02 | cv_single | 10 | 0 | 1.425 |
| calib_lo | scenario_02 | lstm_robust_eps0.0 | 10 | 0 | 1.897 |
| calib_lo | scenario_02 | lstm_single | 10 | 0 | 1.468 |
| calib_lo | scenario_02 | sgan_robust_eps0.0 | 10 | 0 | 1.711 |
| calib_lo | scenario_02 | sgan_single_inf1.00 | 10 | 1 | 1.418 |
| calib_lo | scenario_03 | cv_single | 10 | 2 | 1.542 |
| calib_lo | scenario_03 | lstm_robust_eps0.0 | 10 | 0 | 2.936 |
| calib_lo | scenario_03 | lstm_single | 10 | 2 | 1.755 |
| calib_lo | scenario_03 | sgan_robust_eps0.0 | 10 | 0 | 2.425 |
| calib_lo | scenario_03 | sgan_single_inf1.00 | 10 | 1 | 1.848 |
