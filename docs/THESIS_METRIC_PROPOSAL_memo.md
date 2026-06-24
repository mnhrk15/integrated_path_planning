# 修論方針メモ：「統合的評価指標の作成」提案の検討材料

**目的**: 教授提案「効率性・安全性・その他を統合した新しい AV 評価指標の作成」を、文献調査に基づいて批判的に検討し、教授との方針擦り合わせの材料とする。
**作成日**: 2026-06-24 ／ **基づく調査**: deep-research（28 一次ソース・132主張抽出・22主張を3票敵対的検証で確定）

---

## 1. 一行結論

> **「効率＋安全＋その他を単一スコアに統合する」だけでは新規性は成立しない**（既に標準化済み・むしろ学界は単一スコアから後退中）。
> ただし **①予測不確実性 ②実データ較正された相互作用モデル ③歩行者特化** の交差点には空白があり、**本研究はこの交差点に既に立っている**。新規性はそこに限定して設計すれば成立し得る。

---

## 2. 既存の「統合指標」は飽和している（＝そのままでは新規性なし）

| 指標 / ベンチマーク | 統合の中身 | 出典 |
|---|---|---|
| **nuPlan Closed-Loop Score (CLS)** ＝デファクト標準 | 乗算ペナルティ（衝突なし・走行可能域・進行方向・進捗）の積 × 重み付き平均（**TTC:5／進捗:5／速度遵守:4／乗り心地:2**）。全項[0,1]正規化 | Karnchanachari et al., ICRA 2024 (arXiv:2403.04133); Dauner et al., CoRL 2023 (arXiv:2306.07962) |
| **CARLA Leaderboard Driving Score** | Route Completion × 違反ペナルティ積（歩行者衝突0.50 等） | leaderboard.carla.org (v2.0) |
| **Bench2Drive** | DS＋成功率＋5スキル分解 | Jia et al., NeurIPS 2024 (arXiv:2406.03877) |
| **Waymo Open Sim Agents (WOSAC)** | kinematic（jerk相当）／interactive（衝突・最近接・TTC）／map-based を**実データ分布一致度**で単一realismに統合 | Montali et al., NeurIPS 2023 (arXiv:2305.12032) |

- **乗り心地（jerk）の定量化も確立済み**：nuPlan comfort 指標は縦jerk閾値 ±4.13 m/s³ 等の pass/fail（Dauner et al. 2023 supp.）。当研究の rms_jerk / mean_accel は新規性なし。
- **当研究の FrenetPlanner コスト関数（横/縦jerk・横偏差・速度偏差・時間の重み付き和）は、それ自体が小さな「統合スコア」**。「重み付き和を作る」は実質すでに実装済み。

## 3. 「単一スコアへの統合」は今や批判の対象（＝時流に逆行）

- **高分散批判**：CARLA 系の乗算ペナルティは「90%達成で赤信号0/1/2回→DS=90/63/44.1, std=18.9、手法比較が不可靠」（Bench2Drive, arXiv:2406.03877）。CARLA は実際 v2.1 で積→和に改訂。
- **集計が長所短所を隠す**：単一スコアのルート平均は個別能力を不可視化（同上）。
- **著者自身が限界を自己開示**：at-fault衝突ルールの欠陥、進捗 vs 法令遵守のトレードオフで「最適スコア達成は事実上不可能」（Dauner et al. 2023 supp. §5.3）。
- **業界は単一スコアから後退**：nuPlan-R (arXiv:2511.10403, 2025/11) は gaming 対策に Success Rate と **All-Core Pass Rate**（安全・乗り心地・効率の全てが0.5超）を追加し3指標体系へ拡張。
- **重み・正規化の恣意性**は一般論として既知（OECD, Composite Indicators Handbook, 2008）。

→ 「もう一つ重み付き和を作る」では審査で「Goodhart問題は？重みの根拠は？CLSと何が違う？」で潰れる。

## 4. 歩行者特化の指標は「安全だけ」で止まっている（＝ここに空白）

- VRU特化指標は2024年に複数あるが**すべて安全リスクのみの単一サロゲート**：
  - **P-PET**（予測到達時刻で衝突前にリスク評価, arXiv:2404.15635）
  - **Risk Factor**（sigmoidで(0,1)化, arXiv:2404.14935, IEEE VNC 2024）
  - **Pedestrian Danger Index**（通過距離・車速・減速度を統合, Traffic Injury Prevention 2024）
- 最も提案に近い **arXiv:2503.01852 (2025)**「Socially-Compliant AD」は Safety / Social Compliance / Efficiency / Comfort の**4軸を並記**。ただし歩行者特化の**単一統合指標**として確立してはいない。

→ **「歩行者相互作用 × 多目的統合」は確認範囲では空いている**（※ absence of evidence。関連研究章で arXiv:2503.01852 等との差別化明示が必要）。

## 5. 本研究の差別化ポイント（＝既に持っている武器）

| 文献上のギャップ | 当研究の既存資産（実装済み） |
|---|---|
| (a) 予測不確実性を**陽に**織り込んだ指標 | `distribution_aware_planning` / `chance_epsilon`（chance-constrained）/ KDEベース **NLL** |
| (b) **実データ較正**された相互作用モデルに基づく指標 | RQ2: DUT/CITR から SFM 自車斥力を較正（σ≈1.16, v0≈1.68）＋ closest-approach / avoidance-onset の **KS忠実度評価**（`calibration_harness`） |
| (c) 歩行者特化 | 研究全体が歩行者-AV相互作用 |

→ **(a)×(b)×(c) の交差点に既に立っている。** 教授提案の「統合指標」をこの交差点に位置づけると初めて差別化される。

## 6. 新規性が成立し得る3つの切り口

- **案A：予測不確実性を織り込んだ安全余裕指標**（uncertainty-aware surrogate）
  TTC/PET は点予測ベース。SGAN分布から chance-constrained な「分布的安全余裕」を定義し効率・乗り心地と統合。NLL・chance_epsilon が直接効く。
- **案B：実データ較正ベースの「社会的非適合コスト」**
  RQ2 の実測分布（closest-approach / avoidance-onset）を使い「AVが自然な人間-歩行者相互作用からどれだけ逸脱したか」を距離化して social 軸に。**較正baselineに対する** social compliance 定量化が arXiv:2503.01852 等の heuristic と差別化。DUT/CITR 較正資産が武器。
- **案C：指標の妥当性検証を貢献に**
  新指標＋「既存の単一統合スコアが planner を識別できない場面で提案指標が識別できる」ことを S1/S2/S3 × cv/lstm/sgan で実証。RQ1b の「S2 が clean discriminator・集計は汚染」知見が直結。

> 推奨：**案A or 案B を主軸、案C を検証フレームとして併用**（既存コード再利用率が最大・「重み付き和」批判を構造的に回避）。

## 7. 教授と擦り合わせたい論点

1. **位置づけ**：軸A（実データ較正の実証研究）を**土台に再利用**するか／別軸として切り離すか。本提案は性質が「測定論・ベンチマーク」寄りに変わるピボット。
2. **スコープ**：単一スコアへの「統合」を主張するのか、nuPlan-R 的に**多軸＋バランス指標**に留めるのか（後者の方が時流に整合）。
3. **検証の置き方**：新指標の「妥当性（既存指標が測れないものを測る）」をどう実証するか（案C）。

## 8. 残リスク・要追加調査

- 「絶対に存在しない」とは言えない（網羅証明ではない）。**未確認**：Waymo以外の sim-agents、PDM詳細、distributional safety の具体論文、social compliance を統合した既存例 → 関連研究章で個別に潰す追加サーベイ要。
- **要精読**：arXiv:2503.01852（4軸 socially-compliant）、arXiv:2403.02297（Shao et al. 2024, uncertainty-aware prediction survey）。
- **注意**：「RSS の歩行者向け較正は未実施」という主張は**反証された**（既存研究の可能性）。案Bでは SFM較正 vs RSS較正の差別化を明示。

---

### 参考文献（主要・検証済み）
- Karnchanachari et al. "Towards Learning-based Planning: The nuPlan Benchmark." ICRA 2024. arXiv:2403.04133
- Dauner et al. "Parting with Misconceptions about Learning-based Vehicle Motion Planning (PDM)." CoRL 2023. arXiv:2306.07962
- Jia et al. "Bench2Drive." NeurIPS 2024 (Datasets & Benchmarks). arXiv:2406.03877
- Montali et al. "The Waymo Open Sim Agents Challenge." NeurIPS 2023. arXiv:2305.12032
- "nuPlan-R." 2025. arXiv:2511.10403
- Lin et al. "Predicted Post-Encroachment Time (P-PET)." 2024. arXiv:2404.15635
- Xhoxhi & Wolff. "Risk Factor for VRU-CAV." IEEE VNC 2024. arXiv:2404.14935
- Dąbkowski et al. "Pedestrian Danger Index." Traffic Injury Prevention 2024. DOI:10.1080/15389588.2024.2435048
- Reimann et al. "RSS as safety metric for formalisation." 2024. arXiv:2403.18764
- "Socially-Compliant AD in Mixed Urban Traffic (Safety/Social/Efficiency/Comfort)." 2025. arXiv:2503.01852
- Shao et al. "Uncertainty-Aware Prediction and Application in Planning: Survey." 2024. arXiv:2403.02297
- OECD. "Handbook on Constructing Composite Indicators." 2008.
