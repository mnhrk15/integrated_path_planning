# 最終ベンチマークレポート: 4世代比較（2026-06-11）

- 対象コミット: `82d46c0`（全レビュー修正適用済み、113テスト合格）
- 本レポートは第4世代 `*_final` キャンペーン（exp系 1,329ラン + comfort/rand 303ラン + proc_planning）の完走結果と、第1〜3世代との比較をまとめる。論文表の実体は **`output/*_final/` 側から生成すること**。

---

## 1. 世代系譜

| 世代 | outdir サフィックス | 含まれる修正 | 状態 |
|---|---|---|---|
| 第1世代 | （無印）`comfort_s*`, `benchmark_rand_s*`, `exp_margin_control` 等 | オリジナル（全バグ込み） | 比較用に残置 |
| 第2世代 | `*_obsfix` | + C-1（observer 経過時間多重加算の修正。観測間隔が 0.4s に正常化） | comfort/rand のみ |
| 第3世代 | `*_anchorfix` | + M-1/M-2（予測の時刻再アンカー: 最終観測サンプル→現在時刻） | comfort/rand + exp系 |
| **第4世代** | **`*_final`** | + C-3/M-9（緊急停止運動の統合・縮退候補の棄却）+ 本セッション分: M-7（横格子対称化）/ M-8（yaw 規約統一）/ M-10（復帰閾値 2.0/3.0）/ CAUTION 速度・加速度チェックの index 0 除外 / ホライゾン端点包含（実効 4.8s→5.0s）/ スプライン端切り詰めのロックステップ化 / エゴ曲率キャッシュ+失敗時保持 / C-2（モデル整合 fail-fast）/ M-12〜15（termination_reason・失敗ラン非ゼロ終了・per-method シード・リトライ込み t_plan）/ M-6（chance_epsilon 検証）/ M-5（派生YAML統一）/ M-4（default_config.yaml 廃止） | **本レポートの実体** |

**重要**: M-1/M-2（評価アンカー）と M-7/M-8/M-10（挙動）の変更により、**修正前後の ADE/FDE/挙動指標の絶対値は世代間で非比較**。世代間比較は定性的結論（ランキング、衝突の有無、トレードオフの存否）に限る。

完走実績（第4世代、全キャンペーン FAILED 0・クラッシュ 0）:

| キャンペーン | ラン数 | 備考 |
|---|---|---|
| `exp_margin_control_final` | 480（8条件×20シード×3シナリオ） | 引き継ぎ書の「540」は誤記 |
| `exp_margin_control_mc5_final` | 480 | 同上 |
| `exp_footprint_uncensored_final` | 369（3形状×41ラン×3シナリオ） | 引き継ぎ書の「249」は誤記 |
| `comfort_s{1,2,3}_final` | 123（41×3） | CV は n=1（決定論） |
| `benchmark_rand_s{1,2,3}_final` | 180（60×3） | v0 ランダム化 GT |
| `exp_proc_planning_final` | 4条件×~100ステップ | 単独実行で計測 |

---

## 2. comfort/rand: 手法ランキングと衝突

### 2.1 ランキング CV > LSTM > SGAN は維持（第2世代で確立、第4世代でも全シナリオ有意）

rand 系（n=20/手法、ADE、Welch 両側）:

| campaign | CV | LSTM | SGAN | CV vs LSTM | CV vs SGAN | LSTM vs SGAN |
|---|---|---|---|---|---|---|
| rand_s1 | 0.137 (n=8†) | 0.558 | 0.943 | p=3.0e-14 | p=1.6e-20 | p=5.5e-16 |
| rand_s2 | 0.105 | 0.235 | 0.465 | p=2.5e-25 | p=9.4e-16 | p=2.5e-12 |
| rand_s3 | 0.079 | 0.294 | 0.448 | p=1.1e-10 | p=2.6e-18 | p=1.8e-07 |

† CV の S1 は 14/20 ランが衝突で早期終了し、完全ホライゾン GT ペアが存在しないため標準 ADE が nan になる（§7 参照）。有効 n=8。

comfort 系（決定論 GT）でも LSTM < SGAN は全シナリオ有意（s1: p=9.9e-14、s2: p=1.6e-30、s3: p=1.4e-29）。

### 2.2 衝突（termination 列ベース、第4世代の新発見）

| campaign | CV | LSTM | SGAN |
|---|---|---|---|
| comfort_s1 | **1/1 衝突**（t=4.3s, min_dist 1.060） | **8/20 衝突** + 10 timeout | 0 衝突（ただし 19/20 timeout） |
| comfort_s2 | 0 | 0 | 4/20 衝突 |
| comfort_s3 | 0 | 0 | 0 |
| rand_s1 | **14/20 衝突** | 2/20 + 8 timeout | 4/20 + 8 timeout |
| rand_s2 | 0 | 0 | 1/20 |
| rand_s3 | 5/20 | 4/20 | 9/20 |

- **決定論 GT の「衝突ゼロ天井」は S1 で完全に崩壊**（検証ゲート §4-3 の seed0 単発確認がキャンペーン全体に一般化: CV 1/1、LSTM 8/20）。「正確だが分布を持たない予測は危険」の論点は CV だけでなく LSTM にも拡張された。
- S1 SGAN は衝突ゼロだが 19/20 が timeout（M-10 で CAUTION 滞在が伸び慎重化。安全⇔効率トレードオフの好例）。
- 旧 ★2 の「S2 ランダム化 GT のグレーズ」構図は消滅に近い（rand_s2 は SGAN 1 件のみ）。代わりに **S1/S3 が衝突弁別シナリオ**になった。

### 2.3 NLL（第4世代）

LSTM −0.10〜1.70 nats / SGAN 2.02〜9.20 nats。第2世代での結論（「全モデル重度過信」はバグの産物、SGAN のみ分布が広い/ずれる）を踏襲。SGAN S1 の 8.6〜9.2 は依然高い。

### 2.4 挙動の世代間変化（絶対値は非比較、傾向のみ）

- S3 の走行時間が第3世代 22.6〜28.0s → **第4世代 9.0s** に激変。第3世代で見られた S3 の長時間 CAUTION 走行は M-8 yaw 規約バグ等の産物だった可能性が高い。第4世代の S3（comfort）は**3手法が完全同一挙動**（time 9.0s、min_dist 1.7728、rms_jerk 2.435 がビット一致）— 予測の差が計画にバインドしないシナリオになった。
- 平均速度が全体に上昇（例: comfort_s2 CV 2.45→5.00 m/s）。M-7 対称格子・M-8 yaw 修正・端点包含で候補空間が正常化した効果。

---

## 3. ★1 主判定: 「inflation は robust を支配できない」— 維持

`exp_margin_control_final`（480ラン）の `make_margin_report` 判定:

> **平均ベース**: どの inflation も3シナリオ同時に「MinDist ≥ robust かつ Time ≤ robust」を達成できない → 分布情報の寄与を示唆
> **有意性ベース**: 全 inflation が少なくとも1シナリオで robust に有意に劣る（MinDist 低下 or Time 増加、p<0.05）→ 分布の形が情報を持つ証拠

代表値（Δ = inflation − robust）:

- S1: inf1.50 で MinDist +0.533 (p=5.2e-06) だが衝突1件発生、Time +2.19。inf1.00 は Time +5.98 (p=3.1e-06)。
- S2: inf1.35 までは MinDist 劣後、inf1.50 で +0.220 (p=6.7e-18) だが Time +7.07 (p=2.6e-04) — 支配不可の最明瞭例。
- S3: **inf1.00〜1.20・robust・lstm_single は全シードでビット一致挙動**（Δ=0、p=nan）。S3 の判定は inf1.35（Time +1.75, p=0.04）と inf1.50（MinDist −0.373, p=4.6e-11 かつ Time +8.64）の劣化のみが根拠。
- 実験B（LSTM 分布）: lstm_robust は3シナリオ全てで MinDist 有意改善（S1 +0.202 p=4.3e-06、S2 +0.515 p=9.6e-25、S3 +0.128 p=9.0e-05）かつ Time 増なし — robust 計画の利得は SGAN 固有でなく LSTM 分布でも成立。

注意: Time飽和（cap 打切り）が S1 で多い（inf1.00: 19/20）。打切りは Time ≤ robust 側に有利に働くため、実験Aの判定を弱めない（REPORT.md 付記）。

---

## 4. footprint: 矩形違反は解消、ただし完走性の旧結論は崩壊

`exp_footprint_uncensored_final`（369ラン、time cap 60s）:

- **Q1**: 論文構成（circle, r=1.0）の **50/123 ランが車両矩形（4.5×2.0m）に違反**（S1 は LSTM 19/20・SGAN 20/20 でほぼ全滅、worst clearance −0.200m）。論文の「衝突ゼロ」は楕円近似の楽観性の産物。
- **Q2**: multi_circle（3円 r=1.25）で矩形違反 **2/123**、multi_circle5 で 4/123 に激減。安全側の主張は強化。
- **ただし旧結論「multi_circle は完走性を損なわない」は崩壊**: goal_reached が S3 SGAN 20/20→**0/20**、S2 SGAN 16/20→1/20、S1 SGAN 20/20→9/20（multi_circle）。cap 60s 未達の stall が全体で156ラン。第3世代で観測された S3×mc の計画デッドロック自体は解消した（検証ゲート §4-2: seed0 は 11/74 失敗まで改善）が、**今度は footprint 基準の衝突が発現**する（seed0 は t=7.4s、min_dist 1.372 < 1.45。中心間 1.2m の旧基準なら非衝突）。
- 論文の書き方: 「multi_circle は安全性を矩形基準で保証するが、現行のコスト設計のままでは完走率を大きく犠牲にする」— マージン⇔完走性のトレードオフとして提示し、コスト再調整を今後の課題とする。
- 挙動保存チェック（旧 PoC・旧キャンペーンとの per-seed 照合）は全系統 FAIL — **挙動変更が意図的であるため想定どおり**（M-7/M-8/M-10 等）。

---

## 5. proc_planning: 償却係数の再計測（M-15 リトライ込み）

`exp_proc_planning_final`（S2, sgan, seed0, dt=100ms, 単独実行）:

| condition | plan mean [ms] | p50 | p95 | xbase |
|---|---|---|---|---|
| circle_single | 150.1 | 151.5 | 190.0 | 1.00x |
| mc5_single | 348.2 | 353.8 | 394.5 | 2.32x |
| circle_robust20 | 388.3 | 306.4 | 643.0 | **2.59x** |
| mc5_robust20 | 566.5 | 515.2 | 804.7 | 3.77x |

- robust(20サンプル) の償却係数は **2.59x**（旧 2.06x）。ただし **M-15 修正によりリトライ時間込みの計測**になったため旧値と直接比較不可（旧計測は初回 plan のみ）。論文には新値を採用し、計測定義を脚注で明記。
- 全条件で plan time が dt=100ms を超過（share 100%）— リアルタイム性は未達であり、これは従来から既知。

---

## 6. 論文修正が必要な箇所リスト

1. **旧★2「S3 SGAN ADE 優位の消失」は前提消滅**: SGAN 優位は C-1 バグの産物で最初から存在しなかった（第2世代で確定、第4世代も CV > LSTM > SGAN）。
2. **NLL 過信の記述を書き換え**: 「全モデル 16〜19 nats の重度過信」→「LSTM は較正域（−0.1〜1.7）、SGAN のみ高め（2.0〜9.2）」（`docs/OBSFIX_BENCHMARK_RECHECK.md` 留意点参照）。
3. **修正前後の ADE/FDE/挙動指標の絶対値は非比較**（M-1/M-2 再アンカー + M-7/M-8/M-10 挙動変更）。論文表は全て `*_final` から再生成。
4. **決定論 GT の「衝突ゼロ天井」は S1 で崩壊**（CV 1/1、LSTM 8/20 衝突）。「決定論 GT では衝突ゼロ」という記述は削除し、「分布を持たない予測の危険」の論拠として使う。
5. **フェイルセーフ記述の更新**: M-10 後は距離ゲートが実効（旧デフォルトは clearance 換算 −0.7/−0.2m で無効だった）。再計画は実質最大2回/ステップ（カウンタ上限3はバックストップ）。
6. **実効計画ホライゾンは 5.0s**（旧 4.8s、端点包含修正）。
7. **proc_planning は 2.59x**（リトライ込み定義、§5）。
8. **footprint の結論差し替え**: 「multi_circle は完走性を損なわない」→ マージン⇔完走性トレードオフ（§4）。
9. min_ttc の CSV 表現が旧世代（inf）と新世代（空欄/NaN）で異なる — 表生成時に注意。
10. `exp_sm_clearance` は再実行していない（footprint キャンペーンが実質カバー）。

---

## 7. 既知の限界（レポート/論文の limitation 節用）

1. **S3×multi_circle の衝突**: デッドロックは解消したが footprint 基準衝突が発現（seed0: t=7.4s、min_dist 1.372 < 1.45）。multi_circle の完走率低下（§4）と併せて、矩形整合 footprint 下での計画コスト設計は未解決。
2. **tv_values（目標速度格子）の上側端点非対称**: `arange` が上側端点を含まない既知の設計非対称（n_s_sample=1 で {6.94, 8.33}、9.72 は出ない）。意図的に温存（候補数+50%の計算量増を回避、「目標速度超過候補を出さない」とも解釈可能）。
3. **ego 斥力の等方性**: GT 側の歩行者は ego を等方斥力源として回避するため、GT が楽観的（実歩行者より協調的）。
4. **CV の短時間衝突ランで標準 ADE が nan**: 衝突で早期終了したランは完全ホライゾン GT ペアを持たず ADE 計算から脱落（rand_s1 CV は有効 n=8/20）。ADE 比較が「生存バイアス」を持つ点に注意。
5. **S3（通常 footprint）は予測がバインドしない**: 3手法・8条件中6条件がビット一致挙動。S3 単体では手法/計画方式の弁別力がない（衝突弁別は rand_s3 でのみ発生）。
6. Time飽和（cap 打切り）による右側打切り: S1 の time_s 平均は過小推定（★1 判定には保守側）。

---

## 付録: 生成物の所在

- ★1: `output/exp_margin_control_final/{REPORT.md, welch_tests.csv, welch_vs_baseline.csv, summary.csv, tradeoff_curve.png}`
- mc5: `output/exp_margin_control_mc5_final/{summary.csv, welch_vs_baseline.csv}`
- footprint: `output/exp_footprint_uncensored_final/{REPORT.md, all_runs.csv}`
- comfort/rand: `output/{comfort,benchmark_rand}_s{1,2,3}_final/{summary_stats.csv, all_runs.csv, latex_table.txt}`
- proc: `output/exp_proc_planning_final/` + `/tmp/final_proc_planning.log`
- 検証ゲート記録・修正全リスト: `docs/HANDOVER_20260611_S3.md` §3〜4
