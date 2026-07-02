# RQ1b 感度分析レポート

較正済み反応モデル下での robust/inflation/single 計画の再ベンチ（感度分析、外的検証ではない）。
全ラン cruise=3.0 m/s（RQ2 較正有効域 ~[0.4, 4.0] m/s 内＝5-6 m/s 外挿を回避。RQ2 limitation #2）。AVEC の 6/5/5 m/s 結果とは直接非比較で、同一 ~3 m/s の `avec` アームが域内再ベースライン。

> M6 整合注記: GT の calib/±1SD は radius=0.35 較正値。DEFAULT_AGENT_RADIUS を 0.30 に整合した再較正は LOCO 平均 (1.168, 1.712) で、ここでスイープ済みの ±1SD box [1.040,1.272]×[1.542,1.820] 内に収まる（~1-2% シフト）。よって本キャンペーン（1980 ラン）は再実行せず、結論は補正後の点も感度範囲としてカバーする。

## GT 反応モデル設定（σ/v0）

| gt_label | sigma | v0 | meaning |
|---|---|---|---|
| avec | per-scenario | per-scenario | AVEC per-scenario default (re-baseline) |
| calib | 1.156 | 1.681 | LOCO mean (canonical calibration) |
| calib_lo | 1.04 | 1.542 | -1SD (weakest avoidance) |
| calib_hi | 1.272 | 1.82 | +1SD (strongest avoidance) |
| calib_loso_vmax | 1.085 | 2.617 | LOSO fold vci_back (real point; v0 above +1SD box) |
| calib_loso_smin | 0.743 | 1.849 | LOSO fold vci_lat_bi (real point; sigma below -1SD box) |

（avec の σ/v0 は各シナリオ YAML 値: S1/S3 σ0.7/v0 3.5, S2 σ0.3/v0 2.1。実効値は means.csv の min_dist 等とともに all_runs.csv の ego_repulsion_sigma/v0 列に記録。）

## 判定サマリ

- robust_gain_holds: 全シナリオ同時に robust を支配する inflation が無い（＝主張①保持）
- cv_danger_holds: CV single の collided-run > SGAN robust **かつ** run-level 片側 Fisher p<0.05（有意な主張②保持）。方向はあるが非有意なら `cv_danger_undetermined`＝判定保留（単桁差はノイズ grade）
- lstm_danger_holds: LSTM single vs LSTM robust に同じ有意性ゲート

> 注記（F4）: `*_single` 条件の「単一」は予測器の1描画ではなく medoid-of-20（20 サンプル生成→サンプル平均に最も近い1本＝分散抑制済み・モード探索的な代表値。CV は決定論のため縮退）。robust 利得は medoid 相手でも成立している＝主張①には保守的方向。手法間では CV（決定論）と SGAN/LSTM（medoid）で代表値の性質が非対称な点に注意。provenance は all_runs/master_runs の `single_mode` 列（旧キャッシュは空欄）。

| gt_label | robust_gain_holds | dominating_inflations | robust_collisions | cv_danger_holds | cv_danger_undetermined | cv_fisher_p | lstm_danger_holds | lstm_danger_undetermined | lstm_fisher_p |
|---|---|---|---|---|---|---|---|---|---|
| avec | True |  | 0 | False | True | 0.5 | True | False | 0.0287 |
| calib | True |  | 0 | False | True | 0.5 | False | True | 0.1822 |
| calib_lo | True |  | 0 | False | True | 0.2458 | False | True | 0.1186 |
| calib_hi | True |  | 0 | False | False | 1.0 | False | True | 0.306 |
| calib_loso_vmax | True |  | 0 | False | False | 1.0 | False | True | 0.5 |
| calib_loso_smin | True |  | 0 | False | True | 0.1186 | False | True | 0.2458 |

> 注意: 集計 `cv_danger_holds`/`lstm_danger_holds` は衝突をシナリオ横断で合算し、かつ corner GT は seed 予算が少ない（avec/calib=20・±1SD=10）ため、単桁カウントの flip は Monte-Carlo ノイズと区別できない。有意性ゲート（Fisher p<0.05）を課しても集計はなお GT-artifact シナリオに汚染されるため、**主張②は必ず下記の per-scenario 分類（有意セル `*`）で読むこと**。主張①（`robust_gain_holds`）はシナリオ横断の集計でも頑健。

> **未補正 p の降格（review 1.2-3）**: 上表の `cv_fisher_p`/`lstm_fisher_p` は**未補正の raw p**。この集計 Fisher 12 検定を1 family として BH-FDR(α=0.05) を掛けると生存 0 件（最小 raw p=0.0287）＝表の `*_danger_holds=True` 表示は補正後には有意でない。集計 danger は M8 の宣言どおり noise-grade であり、これらの p は multiplicity ledger にも auxiliary（canonical family 非算入）として全件開示される（headline_tests.json → make_multiplicity_ledger.py）。

## 感度（GT 間で判定が反転するか）

- **robust_gain_holds**: 全 GT で不変（頑健）
- **cv_danger_holds**: 全 GT で不変（頑健）
- **lstm_danger_holds**: 方向は全 GT で不変だが一部 GT で有意性が落ちる（少 seed corner の検出力差＝真の方向反転ではない）

> 注意: `cv_danger_holds`/`lstm_danger_holds` の反転は有意性ゲート後でも corner GT の少 seed 予算（10）に左右されやすい。集計 danger の反転は 感度の *示唆* に留め、確定的な per-scenario 信号は下表の有意セル（`*`）で読む。`robust_gain_holds`（主張①）の不変性が最も信頼できる結論。

## 主張② シナリオ別（per-scenario・汚染なし）

各 (GT, シナリオ) の rand 衝突数・run-level Fisher p と分類（single-danger=分布なし single が衝突しつつ robust 2種は無衝突＝真の主張②信号／mixed=single≫robust>0＝主張②方向は残るが robust も非ゼロ／GT-artifact=robust≧single＝弁別でなく GT 自体の衝突／no-conflict=無衝突。fisher_p=single 群 vs robust 群の run-level 片側 Fisher）:

> **fisher_p の読み方（2つの caveat）**: (1) `class` は衝突 *カウント* から、`fisher_p` は collided-run の有意性から独立に決まる。よって `single-danger`（robust=0）でも `fisher_p` が非有意（少 run の偶然パターン）なことがある＝class 名だけで主張②を確定せず必ず `fisher_p`/`*` を併読する。(2) single 群は3計画器（cv/lstm/sgan）×seed を1 run 単位でプールするが、同一 seed・同一シナリオの3計画器は初期条件と RNG を共有し独立でない（pseudo-replication）。よって run-level n は約3倍に水増しされ、Fisher p は反保守的（楽観的＝真の p の下界）。有意セルは『示唆』として読み、確定的結論にはしない。

| scenario | gt_label | cv_single | lstm_single | sgan_single_inf1.00 | lstm_robust_eps0.0 | sgan_robust_eps0.0 | fisher_p | class |
|---|---|---|---|---|---|---|---|---|
| scenario_01 | avec | 1 | 0 | 0 | 0 | 0 | 0.6 | single-danger |
| scenario_01 | calib | 1 | 1 | 0 | 0 | 0 | 0.3576 | single-danger |
| scenario_01 | calib_lo | 0 | 1 | 0 | 0 | 0 | 0.6 | single-danger |
| scenario_01 | calib_hi | 0 | 1 | 1 | 0 | 0 | 0.3551 | single-danger |
| scenario_01 | calib_loso_vmax | 0 | 0 | 0 | 0 | 0 | 1.0 | no-conflict |
| scenario_01 | calib_loso_smin | 1 | 0 | 0 | 0 | 0 | 0.6 | single-danger |
| scenario_02 | avec | 0 | 3 | 6 | 0 | 0 | 0.0078 | single-danger |
| scenario_02 | calib | 0 | 0 | 0 | 0 | 0 | 1.0 | no-conflict |
| scenario_02 | calib_lo | 0 | 0 | 1 | 0 | 0 | 0.6 | single-danger |
| scenario_02 | calib_hi | 0 | 0 | 0 | 0 | 0 | 1.0 | no-conflict |
| scenario_02 | calib_loso_vmax | 0 | 0 | 0 | 0 | 0 | 1.0 | no-conflict |
| scenario_02 | calib_loso_smin | 0 | 0 | 1 | 0 | 0 | 0.6 | single-danger |
| scenario_03 | avec | 0 | 2 | 1 | 0 | 0 | 0.2116 | single-danger |
| scenario_03 | calib | 1 | 3 | 3 | 1 | 1 | 0.2199 | mixed |
| scenario_03 | calib_lo | 2 | 2 | 1 | 0 | 0 | 0.0673 | single-danger |
| scenario_03 | calib_hi | 0 | 2 | 1 | 1 | 0 | 0.4716 | mixed |
| scenario_03 | calib_loso_vmax | 0 | 2 | 1 | 1 | 0 | 0.4716 | mixed |
| scenario_03 | calib_loso_smin | 2 | 2 | 1 | 0 | 0 | 0.0673 | single-danger |

**読み筋（per-scenario・全 GT をデータから自動生成）**:
- **scenario_01**: avec=single-danger / calib=single-danger / calib_lo=single-danger / calib_hi=single-danger / calib_loso_vmax=no-conflict / calib_loso_smin=single-danger
- **scenario_02**: avec=single-danger*(p=0.008) / calib=no-conflict / calib_lo=single-danger / calib_hi=no-conflict / calib_loso_vmax=no-conflict / calib_loso_smin=single-danger
- **scenario_03**: avec=single-danger / calib=mixed / calib_lo=single-danger / calib_hi=mixed / calib_loso_vmax=mixed / calib_loso_smin=single-danger

- **avec** で主張②（single-danger/mixed）が立つシナリオ: ['scenario_01', 'scenario_02', 'scenario_03']。
- **calib** で主張②（single-danger/mixed）が立つシナリオ: ['scenario_01', 'scenario_03']。
- **calib_lo** で主張②（single-danger/mixed）が立つシナリオ: ['scenario_01', 'scenario_02', 'scenario_03']。
- **calib_hi** で主張②（single-danger/mixed）が立つシナリオ: ['scenario_01', 'scenario_03']。
- **calib_loso_vmax** で主張②（single-danger/mixed）が立つシナリオ: ['scenario_03']。
- **calib_loso_smin** で主張②（single-danger/mixed）が立つシナリオ: ['scenario_01', 'scenario_02', 'scenario_03']。

**結論（データ駆動）**: 主張②（分布なし計画は危険）の成否は GT 反応モデルに依存する（上表が一次情報・`*` は per-scenario の single-vs-robust run-level Fisher が有意なセル）。集計 `cv_danger_holds` は単桁・不均等シードでノイズgrade なので、主張②の主証拠はこの per-scenario 有意セル。robust 利得（主張①）は別途 `robust_gain_holds` 参照（全 GT で頑健）。

> **循環性 caveat（M7）**: RQ1b は較正済み反応モデル下での *感度分析* であり外的検証ではない。衝突相手の『GT 歩行者』は実歩行者ではなく較正済み SFM（RQ2 で実 standoff を ~0.7m 過小再現）が生成する。よって主張②の所見は『SFM family 内のパラメータ感度』であって、実歩行者下での安全結論ではない。独立な実データ閉ループ検証は本研究の範囲外。

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
| calib_loso_smin | scenario_01 | sgan_robust_eps0.0 | 10 | 0 | 2.187 | 16.23 | 0.103 | 0.09 |
| calib_loso_smin | scenario_01 | sgan_single_inf1.00 | 10 | 0 | 1.765 | 15.34 | 0.127 | 0.148 |
| calib_loso_smin | scenario_01 | sgan_single_inf1.10 | 10 | 0 | 1.835 | 15.34 | 0.125 | 0.147 |
| calib_loso_smin | scenario_01 | sgan_single_inf1.20 | 10 | 0 | 1.89 | 15.44 | 0.121 | 0.142 |
| calib_loso_smin | scenario_01 | sgan_single_inf1.35 | 10 | 0 | 2.068 | 16.04 | 0.099 | 0.096 |
| calib_loso_smin | scenario_01 | sgan_single_inf1.50 | 10 | 0 | 2.131 | 15.95 | 0.108 | 0.108 |
| calib_loso_smin | scenario_02 | sgan_robust_eps0.0 | 10 | 0 | 1.49 | 16.84 | 0.078 | 0.055 |
| calib_loso_smin | scenario_02 | sgan_single_inf1.00 | 10 | 5 | 1.212 | 25.24 | 2.955 | 0.217 |
| calib_loso_smin | scenario_02 | sgan_single_inf1.10 | 10 | 0 | 1.343 | 25.45 | 2.107 | 0.196 |
| calib_loso_smin | scenario_02 | sgan_single_inf1.20 | 10 | 0 | 1.527 | 28.23 | 1.607 | 0.158 |
| calib_loso_smin | scenario_02 | sgan_single_inf1.35 | 10 | 0 | 1.705 | 29.55 | 1.25 | 0.163 |
| calib_loso_smin | scenario_02 | sgan_single_inf1.50 | 10 | 0 | 1.845 | 31.33 | 1.138 | 0.156 |
| calib_loso_smin | scenario_03 | sgan_robust_eps0.0 | 10 | 0 | 2.495 | 31.91 | 0.256 | 0.138 |
| calib_loso_smin | scenario_03 | sgan_single_inf1.00 | 10 | 3 | 1.41 | 59.9 | 1.243 | 0.102 |
| calib_loso_smin | scenario_03 | sgan_single_inf1.10 | 10 | 0 | 1.799 | 57.19 | 0.829 | 0.064 |
| calib_loso_smin | scenario_03 | sgan_single_inf1.20 | 10 | 0 | 2.291 | 49.99 | 0.931 | 0.089 |
| calib_loso_smin | scenario_03 | sgan_single_inf1.35 | 10 | 0 | 2.736 | 37.53 | 0.612 | 0.126 |
| calib_loso_smin | scenario_03 | sgan_single_inf1.50 | 10 | 0 | 2.937 | 31.85 | 0.571 | 0.143 |
| calib_loso_vmax | scenario_01 | sgan_robust_eps0.0 | 10 | 0 | 2.309 | 16.21 | 0.104 | 0.089 |
| calib_loso_vmax | scenario_01 | sgan_single_inf1.00 | 10 | 0 | 1.897 | 15.26 | 0.124 | 0.143 |
| calib_loso_vmax | scenario_01 | sgan_single_inf1.10 | 10 | 0 | 1.968 | 15.28 | 0.123 | 0.144 |
| calib_loso_vmax | scenario_01 | sgan_single_inf1.20 | 10 | 0 | 2.02 | 15.37 | 0.119 | 0.14 |
| calib_loso_vmax | scenario_01 | sgan_single_inf1.35 | 10 | 0 | 2.19 | 15.96 | 0.102 | 0.101 |
| calib_loso_vmax | scenario_01 | sgan_single_inf1.50 | 10 | 0 | 2.245 | 15.75 | 0.113 | 0.12 |
| calib_loso_vmax | scenario_02 | sgan_robust_eps0.0 | 10 | 0 | 1.61 | 16.99 | 0.076 | 0.056 |
| calib_loso_vmax | scenario_02 | sgan_single_inf1.00 | 10 | 0 | 1.302 | 21.84 | 0.475 | 0.072 |
| calib_loso_vmax | scenario_02 | sgan_single_inf1.10 | 10 | 0 | 1.413 | 19.62 | 0.781 | 0.113 |
| calib_loso_vmax | scenario_02 | sgan_single_inf1.20 | 10 | 0 | 1.586 | 24.64 | 0.911 | 0.138 |
| calib_loso_vmax | scenario_02 | sgan_single_inf1.35 | 10 | 0 | 1.772 | 30.71 | 0.818 | 0.138 |
| calib_loso_vmax | scenario_02 | sgan_single_inf1.50 | 10 | 1 | 1.861 | 29.71 | 0.957 | 0.161 |
| calib_loso_vmax | scenario_03 | sgan_robust_eps0.0 | 10 | 0 | 2.571 | 31.74 | 0.14 | 0.135 |
| calib_loso_vmax | scenario_03 | sgan_single_inf1.00 | 10 | 1 | 1.674 | 56.9 | 1.034 | 0.083 |
| calib_loso_vmax | scenario_03 | sgan_single_inf1.10 | 10 | 0 | 1.962 | 54.48 | 0.871 | 0.075 |
| calib_loso_vmax | scenario_03 | sgan_single_inf1.20 | 10 | 0 | 2.485 | 49.84 | 0.936 | 0.09 |
| calib_loso_vmax | scenario_03 | sgan_single_inf1.35 | 10 | 0 | 2.864 | 36.8 | 0.513 | 0.131 |
| calib_loso_vmax | scenario_03 | sgan_single_inf1.50 | 10 | 0 | 2.942 | 30.89 | 0.501 | 0.155 |

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
| calib_loso_smin | scenario_01 | cv_single | 10 | 1 | 1.494 |
| calib_loso_smin | scenario_01 | lstm_robust_eps0.0 | 10 | 0 | 3.121 |
| calib_loso_smin | scenario_01 | lstm_single | 10 | 0 | 1.716 |
| calib_loso_smin | scenario_01 | sgan_robust_eps0.0 | 10 | 0 | 2.064 |
| calib_loso_smin | scenario_01 | sgan_single_inf1.00 | 10 | 0 | 1.747 |
| calib_loso_smin | scenario_02 | cv_single | 10 | 0 | 1.409 |
| calib_loso_smin | scenario_02 | lstm_robust_eps0.0 | 10 | 0 | 1.863 |
| calib_loso_smin | scenario_02 | lstm_single | 10 | 0 | 1.454 |
| calib_loso_smin | scenario_02 | sgan_robust_eps0.0 | 10 | 0 | 1.683 |
| calib_loso_smin | scenario_02 | sgan_single_inf1.00 | 10 | 1 | 1.406 |
| calib_loso_smin | scenario_03 | cv_single | 10 | 2 | 1.523 |
| calib_loso_smin | scenario_03 | lstm_robust_eps0.0 | 10 | 0 | 2.921 |
| calib_loso_smin | scenario_03 | lstm_single | 10 | 2 | 1.742 |
| calib_loso_smin | scenario_03 | sgan_robust_eps0.0 | 10 | 0 | 2.4 |
| calib_loso_smin | scenario_03 | sgan_single_inf1.00 | 10 | 1 | 1.828 |
| calib_loso_vmax | scenario_01 | cv_single | 10 | 0 | 1.54 |
| calib_loso_vmax | scenario_01 | lstm_robust_eps0.0 | 10 | 0 | 3.262 |
| calib_loso_vmax | scenario_01 | lstm_single | 10 | 0 | 1.833 |
| calib_loso_vmax | scenario_01 | sgan_robust_eps0.0 | 10 | 0 | 2.26 |
| calib_loso_vmax | scenario_01 | sgan_single_inf1.00 | 10 | 0 | 1.809 |
| calib_loso_vmax | scenario_02 | cv_single | 10 | 0 | 1.512 |
| calib_loso_vmax | scenario_02 | lstm_robust_eps0.0 | 10 | 0 | 2.011 |
| calib_loso_vmax | scenario_02 | lstm_single | 10 | 0 | 1.552 |
| calib_loso_vmax | scenario_02 | sgan_robust_eps0.0 | 10 | 0 | 1.79 |
| calib_loso_vmax | scenario_02 | sgan_single_inf1.00 | 10 | 0 | 1.511 |
| calib_loso_vmax | scenario_03 | cv_single | 10 | 0 | 1.692 |
| calib_loso_vmax | scenario_03 | lstm_robust_eps0.0 | 10 | 1 | 2.872 |
| calib_loso_vmax | scenario_03 | lstm_single | 10 | 2 | 1.776 |
| calib_loso_vmax | scenario_03 | sgan_robust_eps0.0 | 10 | 0 | 2.565 |
| calib_loso_vmax | scenario_03 | sgan_single_inf1.00 | 10 | 1 | 1.937 |
