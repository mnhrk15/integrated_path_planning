# 研究の課題点と解決策 — AVEC'26 論文の統合分析

作成日: 2026-06-10
対象: `~/Research/AVEC_FullPaper/main.tex`（AVEC'26 Full Paper）と本リポジトリの実装
根拠: 論文本文・`NUMERICAL_VERIFICATION.md`・本リポジトリ実装の突合、および文献調査（計47件、主要な主張はarXiv実在確認済み）

## 総括

本論文の内的妥当性は高い（123ラン全数値の実装との突合、Welch検定＋Holm–Bonferroni、挙動保存を確認した上での chance-constrained PoC）。課題は **(A) 評価指標・統計設計、(B) シミュレーション環境の妥当性とドメインギャップ、(C) プランナ・車両モデル、(D) 研究ポジショニング** の4群に整理できる。論文が future work とした項目のほぼすべてに確立された手法系譜が存在する。最優先は「保守性と分布情報の分離実験」（★1）と「Ground Truth の分布化」（★2）— いずれも現行コードベースへの小さな変更で実現でき、論文の中心的な未解決の問いに直接答える。

---

## A. 評価指標・統計設計の課題

### A-1. 決定論的 SFM が低分散予測器の ADE/FDE を不当に有利にする（論文認知済み・未対処）

Ground truth が決定論的な力学平衡で生成されるため、LSTM の S1/S2 での ADE 優位は「SFM への適合」を測っているに過ぎない。これは文献上既知の評価病理である:

- Schöller et al., "What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction," IEEE RA-L 2020 (arXiv:1903.07933) — 低分散モデルが best-of-N 指標で深層モデルに勝つ構造を解明。
- Uhlemann et al., "Evaluating Pedestrian Trajectory Prediction Methods With Respect to Autonomous Driving," IEEE T-ITS 2024 (arXiv:2308.05194) — best-of-20 評価と AV 実用の乖離を指摘、single-trajectory 評価を提唱。

**解決策（実装コスト最小の順）:**
1. **single-sample（most-likely）ADE と NLL を併記** — シミュレータを変えずに病理を中和。診断用 `planning_ade/fde` は既にエクスポート済みで、表掲載の判断のみ。
2. **SFM パラメータの個体間ランダム化**で ground truth を分布化 — 希望速度 N(1.34, 0.26) m/s（Helbing & Molnár 1995 / Weidmann 系の標準値）等の分布からのサンプリング。エージェント挙動のランダム化で ground truth を分布化する先例として Anderson et al., IROS 2019 (arXiv:1903.01860) があるが、同論文の手法は SFM パラメータ操作ではなく「実軌道の摂動＋速度の経験分布サンプリング」である点に注意。なお Sensors 2024, 24(15):5011 は集団に対する単一パラメータセットの較正（v0=1.37, τ=0.53 等）であり個体間異質性の実証ではなく、異方性 λ=0.11±0.07 という値も同論文には存在しない — パラメータ分布の幅を文献値で正当化するには別途出典調査が必要。実装は `PedestrianSimulator._apply_social_force_params` への per-agent サンプリング追加。

**実施結果（2026-06-10、★2 前半 = 解決策1の single-sample ADE 併記）:**

- 実装: `run_statistical_benchmark.py` の LaTeX 表生成に ADE（best-of-N）と P-ADE（プランナが消費する単一予測軌道のローリング誤差 = 既存 `planning_ade`）の2列を追加。`--table-only` でキャッシュ済み all_runs.csv から再シミュレーションなしに表を再生成可能。
- **結果: best-of-20 ADE の深層手法優位は single-sample では消失し、全3シナリオで CV が最小**（P-ADE: S1 CV 1.90 vs LSTM 2.09 / SGAN 2.15、S2 CV 1.98 vs 2.01 / 2.07、S3 CV 2.10 vs 2.13 / 2.12）。「プランナが実際に使う1本の軌道では等速モデルが最も正確」であり、Schöller et al. (RA-L 2020) の指摘がこの環境でも成立。A-1 の病理診断をデータで確認するとともに、論文の中心主張（open-loop 精度 ≠ closed-loop 価値）を直接強化する。
- LSTM–SGAN の相対順序は P-ADE でも保存（S1/S2 で LSTM、S3 で SGAN が低誤差、いずれも Welch p<2e-4）— 病理は「深層 vs CV」の比較を歪めるが、深層同士の比較は頑健。
- 注意: P-ADE は標準 ADE と評価ホライゾン・解像度が異なる（ローリング評価、プランナ dt、5 s までの等速外挿込み）ため、**絶対値の列間比較は不可**。手法間の行比較のみに使う。CV の P-ADE はフォールバックと同一モデルである点も解釈時に留意。
- 残り（★2 後半）: NLL 併記、および SFM パラメータの per-agent ランダム化（分布幅の出典調査が前提）は未実施。

**実施結果（2026-06-10、★2 後半の前提 = 分布幅の出典調査。完了 — 実装・NLL は未着手）:**

- **λ=0.11±0.07 の真の出典を特定**: Johansson, Helbing & Shukla, "Specification of a Microscopic Pedestrian Model by Evolutionary Adjustment to Video Tracking Data," Advances in Complex Systems 11(2), 2008 (arXiv:0810.4587)。本文に "the optimal value of the anisotropy parameter was λ = 0.11 ± 0.07"、Table 1 に circular 仕様 A=0.42±0.26, B=1.65±1.01, λ=0.12±0.07（仕様ごとの値も記載）。Sensors 2024 への帰属が誤りだっただけで値自体は実在。ただし (i) この λ は Helbing 系の後方異方性（0≤λ≤1）であり、pysocialforce の `lambda_importance`（相対速度の重み、Moussaïd 仕様で 2.0±0.2）とは**別パラメータ**、(ii) ± はビデオ3本にわたる良好フィット集合の幅（フィット変動）で厳密な個体間分布ではない。同論文は「フィットネスの 0 からの乖離は主に歩行者挙動の異質性による」と明言しており、per-agent ランダム化の動機付けには使える。
- **per-agent ランダム化の本命出典は Moussaïd et al., Proc. R. Soc. B 276:2755 (2009, arXiv:0908.3131)**。pysocialforce が実装するまさにその相互作用仕様（Eq. 6; γ=0.35, n=2, n′=3, λ=2.0 が `.venv` の default.toml と一致）の較正元で、**希望速度の個体間分布 v0 ~ N(1.29, 0.19) m/s（"normally distributed", mean±sd、被験者40名）**、τ=0.54±0.05 s、A=4.5±0.3、γ=0.35±0.01、n=2.0±0.1、n′=3.0±0.7、λ=2.0±0.2 を報告。v0 の ±0.19 のみが個体間 SD で、τ 以下の ± はフィット不確かさ（個体間分布の根拠にならない）。Weidmann の N(1.34, 0.26)（文献総覧の古典値）と整合的で、両者を併引用すれば分布幅 0.19〜0.26 m/s を正当化できる。
- 実装フック（コード確認済み）: pysocialforce の `DesiredForce` は `peds.max_speeds`（= `max_speed_multiplier` × 初期速度ノルム）を希望速度として駆動するため、**`sim.peds.max_speeds` への per-agent サンプル代入だけで v0 ランダム化が完結**する。τ は state 配列に per-agent 列が存在するが `DesiredForce` はスカラー config（`relaxation_time`）を読むためフォース側のパッチが必要 → v0 のみのランダム化を推奨（出典の強さとも整合）。
- 補助文献: Seer et al., Transp. Res. Procedia 2:724 (2014) は235歩行者の per-trajectory 非線形回帰でパラメータ分布を推定（個体別較正の方法論先例。本文 PDF は有料壁で具体値未取得）。個体異質性が集団流特性を変える先例として Campanella, Hoogendoorn & Daamen, TRR 2124 (2009)。

### A-2. 衝突ゼロ＝天井効果でシナリオの識別力が不足

全123ランで衝突ゼロ、S3 の MinTTC は全手法 0.85 s で幾何学的に飽和（LSTM–SGAN 差 0.002 s は p=4.7e-3 で統計的には有意だが実用上無意味）。「安全性の差を検出する実験」としては感度不足。interPlan（Hallgarten et al., IROS 2024）は、集計スコアの良さが out-of-distribution シナリオでの脆弱性を隠すことを実証している。

**解決策:** 飛び出し（遮蔽からの ambush）・急な方向転換・信号無視など、予測が本質的に困難なシナリオを YAML で追加し、衝突・急制動が実際に起きるレジームで手法を分離する。安全側の天井を破らない限り「SGAN の相互作用考慮が効く状況」は原理的に観測できない可能性が高い。

### A-3. 比較設計の非対称性（軽微）

CV は n=1 で検定外、CV の ADE は single-shot で best-of-20 と非可換（論文内で明示済み）。CV に観測ノイズを与えて20ラン化すれば表が対称になる。ジャーナル拡張ではシード数 50〜100 への増強が望ましい。

### A-4. 衝突判定の幾何が想定車両形状と不整合（論文未認知・本分析の新規指摘）

実装上、自車の衝突判定・MinDist・TTC はすべて**車両中心の単一円**（`ego_radius`=1.0 m）に基づく:

- 判定: `frenet_planner.py:820`（`inflated_radius = robot_radius + obstacle_radius`）
- 指標: `data_structures.py:320-326` — MinDist は**中心間距離**、衝突は中心間距離 < 1.2 m

一方、車両は 4.5×2.0 m の長方形として描画されている（`animator.py:349-350`。論文本文には車両寸法の記載がなく、footprint の定義自体が論文上明示されていない — このこと自体も本指摘を補強する）。半長 2.25 m に対し判定円 1.0 m では前後端がカバーされず、S2 の MinDist 1.69〜1.85 m（中心間）は縦方向なら矩形 footprint 内に入り得る距離である。「衝突ゼロ」の頑健性が footprint の取り方に依存している。

**解決策:** 車両を 2〜3 個の円でカバーする multi-circle footprint（Frenet プランナの定石）へ置換。ベクトル化済みの衝突判定に円中心オフセットを追加するだけで実装可能。**ジャーナル化前に必ず再検証すべき**。

**検証結果（2026-06-10 実施、★3 Step 1–2。詳細: `output/footprint_recheck/REPORT.md`）:**

- 実装: `ego_footprint: circle | multi_circle` を config に追加（`src/core/footprint.py`、デフォルト circle で既存挙動保存、全58テストパス）。multi_circle は 4.5×2.0 m 矩形を3円（オフセット 0, ±1.5 m、半径 1.25 m）で正確被覆。評価メトリクス側（`compute_safety_metrics_static`）に配線済み、プランナ側判定は未変更（Step 3）。trajectory.npz に ego_yaw を保存するよう拡張。
- 後付け検証: 保存済み代表ラン19件の trajectory.npz に対し、(i) multi-circle 判定、(ii) 厳密な矩形–歩行者円距離、を再計算（yaw は有限差分復元、保存 min_distances との突合 max|Δ|=0）。
- **結論: 懸念は実証された。S2 の SGAN ランで矩形 footprint への実侵入を検出**（2/19 ラン: `scenario_02` と `timing_s02_r3`、最深 −0.13 m、t≈9.3–10.1 s の9ステップ連続）。接触は減速しながらのすれ違い中の**左前角**（歩行者は車両座標 (+2.3, +1.07) ≒ 角 (2.25, 1.0) 近傍、中心間距離 2.0–2.6 m で単一円判定 1.2 m の射程外）。走行中（v=1.3–3.0 m/s）で yaw は滑らかなため復元アーチファクトではない。S2 の CV/LSTM ランと S1/S3 全ランはクリア（最小クリアランス: S2 CV +0.57 m、S2 LSTM +0.66 m）— SGAN の「タイトなすれ違い」（論文の S2 MinDist 最小 1.69 m）が footprint 違反に直結している。
- 限界: 123ラン/480ランの統計キャンペーンはスカラーのみ保存のため後付け検証不可。**「衝突ゼロ」の主張は multi-circle でのベンチマーク再実行（Step 3）まで保留とすべき**。なお multi-circle 判定は矩形に対し保守的（横方向スラック 0.25 m）で、19件中3件をフラグするが厳密矩形では2件（`timing_s02_r1` は保守性による偽陽性）。

**再検証ベンチマーク結果（2026-06-10 実施、★3 Step 3。詳細: `output/exp_footprint/REPORT.md`）:**

- 実装: multi-circle footprint をプランナの全衝突判定（static・単一サンプル動的・chance-constrained）に拡張（`_path_collision_geometry` で候補経路点を円ごとに heading 方向オフセット展開）。論文の123ランキャンペーンを circle（論文構成アンカー）と multi_circle の2条件で再実行（計246ラン、失敗0）。両条件で条件非依存の観測幾何（中心距離・3円クリアランス・厳密矩形クリアランス）を記録。挙動保存: circle 条件は margin-control キャンペーンのキャッシュと全6セル（3シナリオ×SGAN/LSTM×20シード）で完全一致（max|Δ|=0）。
- **Q1（論文構成は矩形を侵すか）: 現行コードでは 1/123 ラン**（S2 SGAN、最悪クリアランス −0.004 m の限界的接触）。保守的な multi-circle 閾値では S2 で 14/123 ランがフラグ（CV 1、LSTM 10、SGAN 3、最悪 +0.059 m）— S2 のすれ違いは footprint 余裕がほぼゼロのレジームで運用されている。Step 2 で検出した −0.13 m の侵入は旧コード（715d7b3 以前）生成の保存軌道上のもので、現行コードでは侵入は限界的。**従来の 1.2 m 中心距離指標では全246ランで衝突ゼロのまま** — 指標が footprint 違反を構造的に検出できないことの直接証拠。
- **Q2（multi-circle 計画で侵入は消えるか）: 0/123 ラン、全シナリオで矩形クリアランスが有意に増加**（ΔS1 +0.70〜+1.25 m、S2 +0.27〜+0.36 m、S3 +0.45〜+1.62 m、確率手法は全セル p<0.05）。
- **Time コスト（右側打切り解消後の確定値。total_time=60 s で全246ラン再実行、`output/exp_footprint_uncensored/REPORT.md`）**: S1 +2.1〜+2.7 s、S2 +1.5〜+1.6 s、S3 +4.0〜+5.7 s（CV は S3 で +17.1 s、ただし n=1）。**全246ランがゴール到達（恒久スタックなし）** — multi-circle 計画は完走性を損なわない。当初キャンペーン（シナリオ既定の total_time）では S1 multi_circle の 30/41 ランが上限 19.9 s で打切られ、S1 の Time コストを +1.0 s と半分以下に過小評価していた。クリアランス利得は打切りの影響を受けず不変（接近イベントは打切り前に発生するため）。挙動保存チェックは打切り対応版（censored アンカーシードを除外）で全セル PASS（max|Δ|=0）。
- **副産物（再現性の確認、2026-06-10 決着）**: 当初 `output/statistical_benchmark{,_s2,_s3}` の CSV が現行コードを再現せず（ADE max|Δ|≈0.6〜0.8、S2 で time max|Δ|=10.7 s）「論文表の再生成が必要」と疑われたが、123ランを現行コードで再生成（`output/statistical_benchmark_v2_s{1,2,3}`）した結果、**論文表の実体である `output/comfort_s{1,2,3}` とビット単位で一致**することを確認。すなわち論文の公表値は現行コードで正確に再現され、再現性に問題はない。不一致だった `statistical_benchmark*` は 715d7b3（ego-歩行者斥力修正 + ADE/FDE 再設計）以前の古いローカル遺物（git 管理外）であり、アンカーとして使用してはならない（誤用防止に削除推奨）。論文の主要主張8項目（衝突ゼロ、S1/S2 の LSTM ADE 優位、S2 MinDist 序列、S3 MinTTC 飽和、S3 jerk 増、S3 LSTM–SGAN 有意差）はすべて再生成データで成立（`output/benchmark_regen_comparison/REPORT.md`）。footprint キャンペーンの circle 条件も comfort_s* と全123ランで丸め精度内一致（max|Δ|≤5e-5）。
- 結論: A-4 の修正方針は有効（multi-circle 計画は矩形違反を排除し、妥当な走行時間コストで安全余裕を増やす）。ただし S2 が「footprint 余裕ほぼゼロ」で運用されている事実は、単一円幾何の下での「衝突ゼロ」が安全性の証拠として弱いことを定量的に裏付けており、ジャーナル版では multi-circle を既定の評価幾何とすべき。
- **衝突の最終整理（評価幾何別、全246ラン）**: 従来指標（中心間 1.2 m）= 両条件とも 0。真の矩形 = circle 条件 1/123（S2 SGAN、−0.004 m）、multi_circle 条件 0/123。保守的 multi-circle 閾値（1.45 m）= circle 条件 14/123（全て S2、最悪 +0.059 m）、multi_circle 条件 0/123。**multi-circle 計画下ではどの定義でも接触ゼロ**。

**multi-circle 化で確認された課題と対応案（2026-06-10、`output/exp_footprint_uncensored/all_runs.csv` の分析に基づく）:**

1. **被覆の保守性によるコスト水増し（横スラック 0.25 m）**: 3円被覆の半径 1.25 m は車両半幅 1.0 m に対し 0.25 m 余分。S2 で「multi-circle 閾値違反 14 件 vs 矩形違反 1 件」の乖離として顕在化しており、Time コストの一部は被覆アーチファクト。→ **対応（優先・小コスト）**: `ego_footprint_n_circles: 5` でスラック 0.10 m に削減（config のみ、判定点 3→5 倍）。根本対応はカプセル（点–線分距離）判定でスラックゼロ化（中コスト、ベクトル化済み判定の距離計算置換）。
   - **実施結果（2026-06-10、`multi_circle5` 条件 123 ラン追加、計369ラン）**: 矩形違反 0/123・全ランゴール到達を維持したまま、circle 比の Time コストが S2 で +1.5〜+1.6 → **+0.9 s（約4割減）**、S3 LSTM +5.7→+4.6 s、S3 CV +17.1→**+5.5 s**（クリアランスも +0.28→+2.08 m に改善 = 過剰回避ホモトピーからの脱出）。3円 vs 5円の直接 Welch では時間短縮は 6 セル中 5 セルで符号一致するがセル単位では非有意（p=0.086〜0.98、n=20）— 効果は実在するが分散に対して中程度。クリアランスの劣化はなし（全セル p>0.29、ただし S3 SGAN の対 circle 利得は +0.45→+0.29 m とやや弱まり p=0.085 の境界に）。**ジャーナル版の既定は 5 円を推奨**（判定点増のリアルタイム影響は課題 5(ii) で未実測）。
2. **S3（右折）の極端な Time コスト（CV +17.1 s、平均速度 3.07→1.74 m/s）**: 前方円（オフセット +1.5 m）が旋回時に交差点内側を掃くため回避が過剰になる。→ **対応**: C-2（参照経路の曲率平滑化）と統合して経路側を改善するのが本筋。短期的には C-4 のコンテキスト依存 speed cap で「広く避ける」を「減速して通す」に置換。
3. **S1 で jerk 増（LSTM 6.59→7.10、SGAN 6.93→7.46）**: multi-circle では in-loop の min_distance が前後円から測られ小さくなり、単一円用にチューニングされた状態マシン閾値（`safe_distance_caution/emergency`）が早期・頻繁に発火する兆候（S2/S3 では減速が効き jerk はむしろ減少: S3 10.3→9.1）。→ **対応（小コスト）**: 状態マシン判定を絶対距離から clearance ベース（metrics に追加済みの `clearance` キー = 距離 − combined radius）に置換し、footprint モード非依存の意味を持たせる。
   - **実施結果（2026-06-10、clearance 化を実装し 123ラン×3条件で再検証。`output/exp_sm_clearance/`）**: 回復閾値を `safe_distance − (ego_radius + ped_radius)` で clearance 空間へ等価変換（circle 挙動保存を設計で保証、`state_machine.py`、全63テストパス）。検証キャンペーンは circle 条件が margin-control アンカーと per-seed 完全一致（全セル PASS、max|Δ|=0）、かつ **multi_circle / multi_circle5 を含む全369ランが旧状態マシンのキャンペーン（`exp_footprint_uncensored`）とビット単位で一致** — つまり clearance 化は全 footprint モードで挙動を一切変えない。
   - **機序の解明（仮説は棄却）**: 旧閾値（caution 0.5 m / emergency 1.0 m の中心距離比較）は無衝突なら min_distance ≥ combined radius（1.2/1.45/1.30 m）が常に成り立つため**現行レジームでは一度もバインドしない**（全369ランの実測最小 min_dist は 1.51 m）。状態遷移は実質 plan_found のみで駆動されており、「単一円用閾値の早期発火」という上記仮説は誤り。S1 の jerk 増は状態マシン経由ではなくプランナ側（候補棄却の増加に伴う回避機動）に由来する。実際、状態占有（新規記録の `caution_steps`/`emergency_steps`）は multi-circle で増える（S1 SGAN の EMERGENCY 平均 15.3→19.6 ステップ、S3 CV 21→167）が、S1 の per-seed では Δjerk と ΔEMERGENCY は負相関（r=−0.55）— 緊急停止はむしろ低速化で jerk を抑える側。jerk 緩和には C-2（参照経路の曲率平滑化）/ C-4（speed cap）側の対処が必要。
   - **副次的発見**: 状態マシンの安全距離閾値が衝突レジーム以外で vacuous であることが判明（閾値がバインドするのは footprint 重複中のみ）。clearance 化により閾値は footprint モード非依存の物理的意味を獲得したので、高ストレスシナリオ（A-2）導入時や閾値の再較正（正の clearance への引き上げ = 距離感応的な回復遅延の導入）の土台になる。ただし引き上げは挙動を変えるため別実験として扱うこと。
4. **条件間で ADE が変動（S1 LSTM 2.28→2.11 等）**: ego 減速 → 歩行者反応変化 → 予測誤差変化という endogeneity で、予測指標が計画条件に依存する。→ **対応**: 条件間の ADE 比較はしない方針を明記し、endogenous distribution shift（arXiv:2511.11567、C-1 で引用済み）の文脈で Limitations に記載。
5. **検証の負債**: (i) ★1（chance-constrained の分布価値）の結論は単一円幾何で導出されており multi-circle 下での頑健性が未確認 → `run_da_poc.py` に `ego_footprint` オーバーライド追加で再検証可能（小コスト）。**実施済み（2026-06-10）**: 5円・60 s cap で480ラン再実行。主判定（inflation は robust を支配できない）は維持されたが、S1 では robust の便益が footprint 拡大に吸収されて消失 — 詳細は C-1 の「multi-circle 再検証」ブロック参照。(ii) 判定点3倍 × 分布判定20サンプルのリアルタイム性（C-3 と同根）が未実測 → proc_planning の既存記録機構で1ラン実測すれば足りる。**実施済み（2026-06-10）**: 償却は実証（5円×20サンプルで 2.06×）、ただし分布判定は絶対値で 100 ms 予算を超過 — 詳細は C-3 の実測結果ブロック参照。

付記: CV S1 のみ multi-circle で高速化かつクリアランス微減（19.2→18.7 s、rect +2.19→+2.05 m）という符号反転が観測された。footprint 拡大で候補経路の選好が別ホモトピーへ飛んだもので、コスト地形の敏感性を示す例（n=1、参考情報）。

---

## B. シミュレーション環境・ドメインギャップの課題

### B-1. ETH/UCY→SFM の 3〜4 倍の誤差ギャップ（論文認知・対処は future work のまま）

このギャップ自体は文献的に正常: cross-domain 劣化は T-GNN（Xu et al., CVPR 2022, arXiv:2203.05046）以降「trajectory domain shift」として定式化されており、引用により所見をノーマライズできる。

**解決策（軽い順）:**
1. **プロンプトチューニング型適応**: Latent Corridors（Thakkar et al., ECCV 2024, arXiv:2312.06653）— 予測器を凍結したまま <0.1% のパラメータでシーン適応。適用実績は synthetic→real 方向（MOTSynth→MOT/Wildtrack）であり、本研究で必要な real→synthetic はその逆適用になる。
2. **SFM 生成軌道でのファインチューニング**。直接の先例は未発見。Li et al., IROS 2024 (arXiv:2309.10121) は車両ドメインの合成軌道による事前学習であり手法は異なるが、「合成データで予測器を対象分布へ寄せる」方向性の傍証にはなる。
3. **予測分布のキャリブレーション評価**: SGAN の20サンプルのカバレッジを測定（Cheraghi Pouria et al., arXiv:2603.10407 — 較正の不備が計画品質を損なうことを実証。「ADE と独立に」という限定は abstract では未確認、引用時は本文要確認）。C-1 の「分布情報 vs 保守性」問題と直結。

### B-2. 背景エージェントモデル（SFM）自体が結論を左右する

Hagedorn et al.（arXiv:2510.14677）は、nuPlan の背景エージェントを IDM から学習モデル（SMART）に替えるだけでプランナのランキングが変動することを実証。本研究の SFM 歩行者（ego 斥力 σ/v0 は手動設定）にも同じ構造的リスクがある。

**解決策:**
- **自車斥力パラメータの実データ較正** — SRFM（Agrawal et al., arXiv:2409.14844）が JRDB でロボット斥力項を fit した手続きを流用可能。
- 較正先として **DUT/CITR データセット**（Yang et al., IEEE IV 2019, arXiv:1902.00487）が最適: 車両影響下の歩行者軌道の実録で、引用済み Yang et al. T-ITS 2023 と同一系譜。

### B-3. 実データ検証の欠如（外的妥当性の最大の穴）

**解決策:** Yang et al. T-ITS 2023 の「シミュレーションで pre-crash 網羅 → 実データで実用性検証」の2段構造を踏襲し、**inD**（Bock et al., IEEE IV 2020, 交差点の歩行者・車両 11,500 軌道、誤差 <10 cm）または DUT での**リプレイ評価**（記録歩行者に対して再計画）を追加。nuPlan の log-replay vs reactive の分類学を使い、現行 SFM 評価（reactive）と実データリプレイ（log-replay）が相補的であると位置付けられる。

---

## C. プランナ・車両モデルの課題

### C-1. 単一サンプル interface と chance-constrained PoC の未解決点（最重要）

PoC は ε=0 で全シナリオ有意な余裕増（+0.16〜+0.50 m）を示したが、論文自身が認める通り「**分布情報の価値**と**単なる保守化**を分離できていない」。分離実験の標準形は文献上確立されている:

1. **膨張マージン付き単一サンプル対照**: 単一サンプル判定の安全半径をスイープし、「同じ走行時間コストで同じ余裕」を達成できる設定の有無を調べる（Zhou et al., IEEE T-IV 2023, arXiv:2212.11819 の tunable-occupancy が同型）。存在しなければ分布が情報を持つ証拠。**実装は config に半径インフレ係数を1つ追加し、`run_da_poc.py` の CONDITIONS に行を足すだけ**。
2. **LSTM の20サンプルで同じ robust 計画を実行**: 「相互作用考慮分布」固有の価値の直接検証（論文が open question と明記）。コード変更ほぼゼロ。
3. **conformal prediction による較正済みマージン**: Lindemann et al., IEEE RA-L 2023 (arXiv:2210.10254) — 上記1の膨張量を理論保証付きで決定。adaptive conformal（Dixit et al., L4DC 2023）は論文の「adaptive ε」の文献上の答え。

**理論面の注意:**
- scenario optimization 理論（de Groot et al., IJRR 2025, arXiv:2307.01070）によれば、20サンプルで ε=0 を課しても保証できるリスク水準は弱く、per-step ε はホライゾン全体のリスクを過小評価する。
- CVaR 拡張は20サンプルでは尾部推定の分散が大きい（MMD-OPT, arXiv:2412.09121）。サンプル増か、GMM 圧縮（Ren et al., IEEE L-CSS 2023）または RKHS-MMD 代理（MMD-OPT）の併用が安全。
- プランナが慎重になると歩行者の反応分布も変わり較正が崩れる「endogenous distribution shift」（arXiv:2511.11567）は Limitations で言及する価値がある。

**実験結果（2026-06-10 実施、★1 = 上記 1+2 を実装・実行済み。詳細: `output/exp_margin_control/REPORT.md`）:**

- 実装: `collision_margin_inflation`（単一サンプル動的衝突判定の combined radius 乗率、static・分布判定・評価メトリクスには不適用）を config/planner に追加。inflation ∈ {1.0, 1.1, 1.2, 1.35, 1.5} × 3シナリオ × seeds 0..19、計480ラン。挙動保存チェックは旧 PoC 出力と per-seed 完全一致（max|Δ|=0、全3シナリオ）。
- **実験1（膨張マージン対照）の結論: 利得は保守性だけでは説明できない。** どの inflation も3シナリオ同時に「MinDist ≥ robust(ε=0) かつ Time ≤ robust」を達成できず（平均ベース）、全 inflation が少なくとも1シナリオで robust に有意に劣る（有意性ベース、p<0.05）。S2/S3 では robust がトレードオフ曲線を支配: robust と同等の MinDist に到達するには inflation 1.35〜1.5 が必要だが、その走行時間コストは +1.77〜+2.84 s（robust は +0.05〜+1.0 s）。S1 のみ inflation で MinDist は robust を超えるが、有意な Time 増（+0.4〜+0.7 s、かつ inflation 1.1〜1.35 で 16〜19/20 ラン、1.5 でも 13/20 ランがゴール未到達で右側打切り）を伴うトレードオフであり支配ではない。
- **実験2（LSTM 分布での robust 計画）の結論: robust 化はどの分布でも効くが、SGAN（pooling）分布は利得の「効率」で優位。** ΔMinDist(robust−single) は LSTM の方がむしろ大きい（S1 +0.383 vs +0.170、S3 +0.996 vs +0.502、利得差 p=7.2e-4）一方、LSTM robust の Time コストは大きい（S3 +3.24 s vs SGAN +1.00 s、コスト差 p=3.6e-5）。すなわち「相互作用考慮サンプリングだけが robust 利得を生む」は不支持だが、pooling 分布はより小さい走行時間コストで安全余裕に変換される（分布がタイトで較正が良いことと整合）。
- 全480ランで衝突ゼロ。ADE は robust 化で統計的に実質不変（SGAN は全シナリオ p>0.15、ただし S3 は p=0.159 の境界値。LSTM は S1/S3 でむしろ僅かに改善 — ego 挙動変化→歩行者反応の波及による副次効果と解釈できるが、これは仮説であり生データから直接立証されたものではない）。

**multi-circle 再検証（2026-06-10 実施、A-4 課題5(i)。5円 footprint・total_time=60 s で全8条件×3シナリオ×20シードを再実行、計480ラン・失敗0・衝突0。詳細: `output/exp_margin_control_mc5/REPORT.md`）:**

- **両判定は multi-circle 幾何でも成立**: どの inflation も3シナリオ同時に「MinDist ≥ robust かつ Time ≤ robust」を達成できず（平均ベース）、全 inflation が少なくとも1シナリオで robust に有意に劣る（有意性ベース）。S3 は単一円と同型で robust がトレードオフを支配（同等 MinDist に inflation 1.35 が必要だが Time +13.6 s vs robust +3.3 s；inf1.5 は 16/20 ランが 60 s cap 未到達で真のコストはさらに大）。
- **ただし S1 では robust 化の便益が消失・逆転**: plain single-sample が robust より MinDist +0.30 m（p=0.054）かつ Time −1.2 s（p=8e-5）、inflation 1.10/1.20 は MinDist でも有意に上回る（p=0.029/0.015）。実験2側でも S1 の robust 利得は SGAN −0.30 m（負！）/ LSTM +0.03 m（ns）。すなわち **footprint 拡大が既に幾何マージンを供給している S1 では、分布 robust 化は Time コストだけを残して価値を失う**。結論の条件付けが必要: 「分布情報の価値は幾何マージンが薄いタイトなレジーム（S2/S3）で発現し、footprint 保守性と部分的に代替関係にある」。
- SGAN 分布の効率優位は維持（S3: LSTM robust の利得 +2.43 m は大きいが Time +21.1 s かつ 6/20 打切り、SGAN は +1.14 m を +3.3 s で達成。利得差 p=0.015、コスト差 p=1.8e-6）。S2 では SGAN 分布のみ有意な余裕増（+0.11 m, p=0.02 vs LSTM +0.02, ns）。
- 副次: 単一円では「ADE 実質不変」だったが、multi-circle では robust 化で ADE が全セル有意に改善方向（−0.05〜−0.54 m、全 p<2e-3）— 減速・回避の増加が歩行者反応を予測しやすい方向へ変える endogeneity の増幅で、課題4（条件間 ADE 比較不可）の追加証拠。
- ハーネス注記: レポートの旧 PoC アンカー照合は幾何が異なるため「DIFFERS（想定どおり）」表示（挙動保存チェックとしては無効、`make_margin_report.py` を campaign 設定検知に対応させた）。Time飽和列は goal_reached ベースに修正済み。

論文の診断（スプライン曲率変化率ピーク＋無フィルタ状態代入＋状態マシン介入の結合）は説得的。解決は2層:

- **(a) 参照経路側**（数日規模）: 曲率連続なクロソイド/弧長パラメータ化への置換、またはウェイポイント再配置で dκ/ds ピークを除去。`NUMERICAL_VERIFICATION.md` の |d|max=2.166 m を回帰指標に使える。
- **(b) 追従側**（数週間規模）: Kong et al.（IV 2015）の kinematic bicycle + 操舵レート制約付き MPC 追従層。乗り心地指標（RMSJerk が S3 で全手法約2倍）が整備済みのため、検証パイプラインは揃っている。

### C-3. 分布対応計画のリアルタイム性

論文は「ベクトル化で20倍負荷を償却可能」と見込みのみ。**解決策:** 実装して実測する（静的障害物で実績のあるバッチ判定の拡張）。サンプル→ガウス近似で判定を解析的制約に置換する経路（arXiv:2502.10585 は deep ensemble 出力を単一ガウスに集約して線形制約化。多峰性を保つ GMM 版は Ren et al., IEEE L-CSS 2023）はサンプル数への線形依存自体を消せる。

**実測結果（2026-06-10、A-4 課題5(ii) と統合実施。`examples/measure_proc_planning.py`、結果: `output/exp_proc_planning/`）:**

- S2/SGAN/seed 0、アイドルマシンで4条件を逐次実行（footprint circle/5円 × 判定 single/分布20サンプル）、プランナ実時間予算 dt=100 ms。per-step proc_planning（mean / p95 / 超過率）: circle+single **53.7 ms** / 88.5 / 0% → 5円+single **59.1 ms**（1.10×）/ 92.2 / 0% → circle+robust20 **98.0 ms**（1.82×）/ 147.9 / 48% → 5円+robust20 **110.7 ms**（2.06×）/ 168.4 / 60%（max 199 ms）。予測（SGAN 推論）は別途約 20 ms/ステップ。
- **スケーリングの償却は実証**: 名目判定点 100×（5円×20サンプル）が wall-clock 2.06× に収まり、サンプル当たり限界コストは約4% — 「ベクトル化で20倍負荷を償却可能」の見込みは正しい。5円化単体の限界コストは +10%（リアルタイム上の障害にならない）。
- **ただし絶対値では分布判定が予算を超過**: robust20 系は半数前後のステップで 100 ms を超える（このハードウェア・Python/NumPy 実装で）。ボトルネックは判定点数ではなくベース計画自体（53.7 ms）。改善の優先順位は (i) 候補生成・スプライン評価の最適化、(ii) ガウス/GMM 圧縮による解析的制約化（上記）、(iii) 再計画周期の緩和（現状毎ステップ）。
- 限界: 1シナリオ・1シードの指標値（macOS wall-clock）。論文には「相対コスト（償却）は実証、絶対リアルタイム性は実装最適化の余地」と書くのが正確。

### C-4. セマンティックコンテキスト（横断歩道の譲り義務等）

論文の提案 (iii) のまま future work で妥当。横断歩道領域は既に YAML と可視化に存在するため、コスト項追加のインフラはある。

---

## D. 研究ポジショニングの課題

「open-loop 精度 ≠ closed-loop 性能」という中心的主張は、車両領域では確立されつつある:

- What Truly Matters in Trajectory Prediction for Autonomous Driving? (NeurIPS 2023, arXiv:2306.15136) — dynamics gap の指摘、最も近い先行
- nuPlan 2023 チャレンジ / Dauner et al. (CoRL 2023) — open/closed スコアがほぼ無相関
- Bouzidi et al., "Closing the Loop" (arXiv:2505.05638) — **最近傍研究**。予測器×プランナの閉ループ系統評価（車両領域、双方向歩行者モデルなし）

novelty 維持には「**歩行者予測器を、SFM による双方向の歩行者⇆車両反応の下で、制御された閉ループで比較した初の系統的研究**」という差別化軸の明示的強化が必要。Ivanovic & Pavone の planning-aware 評価系譜（arXiv:2107.10297, 2110.03270）と、Joint Metrics Matter（Weng et al., ICCV 2023 — SGAN 型 best-of-K 評価そのものへの批判）の引用追加が位置付けを固める。

---

## 優先順位

| 優先 | 施策 | 答える課題 | 実装コスト |
|---|---|---|---|
| ★1 | 膨張マージン単一サンプル対照 + LSTM分布でのrobust計画 | C-1（保守性vs分布情報の分離 = 論文最大の未解決問） | 小（config+PoCハーネス拡張） |
| ★2 | SFMパラメータ個体ランダム化 + single-sample ADE併記 | A-1（ADE病理） | 小 |
| ★3 | 車両footprintのmulti-circle化と衝突再判定 | A-4（衝突ゼロの頑健性、未認知） | 小 |
| ★4 | 飛び出し等の高ストレスシナリオ追加 | A-2（天井効果） | 中 |
| ★5 | conformal較正マージン + CVaR/GMM拡張 | C-1理論強化 | 中 |
| ★6 | 参照経路の曲率平滑化 → bicycle MPC追従層 | C-2（横方向アーチファクト） | 中〜大 |
| ★7 | DUT/inDリプレイ評価 + ego斥力の実データ較正 | B-2/B-3（外的妥当性） | 大 |

★1〜3 は現行コードベースで完結し、合わせてジャーナル拡張の核になり得る。特に★1は、現論文の結論（「相互作用考慮サンプリング単体では余裕を広げない／全サンプル消費で回復する」）を「**分布の形そのものが効くのか、単に保守的になっただけか**」まで一段深く決着させる。

**実施状況（2026-06-10 時点）:**

- ★1 **完了**（C-1 の実験結果ブロック参照）。ただし multi-circle footprint 下での頑健性再確認が残る（A-4 課題5(i)）。
- ★2 **前半完了**（single-sample P-ADE の表併記、A-1 の実施結果ブロック参照）。後半（SFM パラメータランダム化・NLL）は分布幅の出典調査が前提のため未着手。
- ★3 **完了**（A-4 の検証結果・再検証ベンチマーク結果ブロック参照、右側打切り解消済み）。フォローアップ5件のうち実験系4件は**すべて実施済み**: 5円化（課題1）、状態マシンの clearance 化（課題3、挙動不変を369ランで実証・jerk 仮説は棄却）、★1 の multi-circle 再検証（課題5(i)、判定維持・S1 で便益吸収）、リアルタイム実測（課題5(ii)、償却実証・絶対値は超過）。残る課題2（S3 の Time コスト）は C-2/C-4、課題4（ADE の endogeneity）は Limitations 記載事項。
- 副次成果: 論文表の現行コードでの完全再現性を確認（A-4 副産物）。論文キャンペーンの実体は `output/comfort_s{1,2,3}`。

## 検証メモ

- コードに関する主張（A-4 の衝突幾何、MinDist の中心間距離定義、★1/★2 の実装箇所と工数見積もり）は 2026-06-10 に実装と突合済み。
- 知識カットオフ以降の文献のうち本文の論拠となる arXiv:2510.14677、arXiv:2603.10407 はarXiv上で実在・内容一致を確認済み。その他の文献も調査時にarXiv/出版社ページで確認済み。
- 2026-06-10 に本ドキュメント全体の再検証を実施（コード主張・実験数値・論文本文・全引用文献の4系統突合）。引用 arXiv ID は全件実在（arXiv:2511.11567 含む）。同時に以下を修正済み: (1) A-4 の出典記述 — 論文 Fig. 3 キャプションに車両寸法の記載はなく実装側のみ、(2) A-1 — Sensors 2024 の λ=0.11±0.07 は同論文に存在せず（個体間異質性の実証でもない）、Anderson et al. の手法記述と N(1.34, 0.26) の出典を訂正（追記: 同値の真の出典は Johansson et al. 2008 と後日特定、A-1 の★2後半実施結果ブロック参照）、(3) B-1 — Latent Corridors の適応方向（synthetic→real）と Li et al. の内容（車両ドメイン合成データ事前学習）を訂正、(4) C-1/C-3 — Ren et al. に RKHS は無い、arXiv:2502.10585 は GMM ではなく単一ガウス近似、(5) 実験数値の丸め（S2 Time コスト下限 +1.77 s、S1 inflation 1.5 のゴール未到達 13/20、SGAN S3 ADE p=0.159 の境界性）。なお実験2の「LSTM の20サンプル」は正しい（`run_da_poc.py` はベースシナリオ YAML の num_samples=20 を使用し、モデルパスのみ切替）。
