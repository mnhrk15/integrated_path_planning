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
2. **SFM パラメータの個体間ランダム化**で ground truth を分布化 — 希望速度 N(1.34, 0.26) m/s 等の較正済み分布からのサンプリング。先例: Anderson et al., IROS 2019 (arXiv:1903.01860)。実歩行者の SFM パラメータ異質性の実証（Sensors 2024, 24(15):5011, 例: 異方性 λ=0.11±0.07）により分布を文献値で正当化できる。実装は `PedestrianSimulator._apply_social_force_params` への per-agent サンプリング追加。

### A-2. 衝突ゼロ＝天井効果でシナリオの識別力が不足

全123ランで衝突ゼロ、S3 の MinTTC は全手法 0.85 s で幾何学的に飽和（差 0.002 s）。「安全性の差を検出する実験」としては感度不足。interPlan（Hallgarten et al., IROS 2024）は、集計スコアの良さが out-of-distribution シナリオでの脆弱性を隠すことを実証している。

**解決策:** 飛び出し（遮蔽からの ambush）・急な方向転換・信号無視など、予測が本質的に困難なシナリオを YAML で追加し、衝突・急制動が実際に起きるレジームで手法を分離する。安全側の天井を破らない限り「SGAN の相互作用考慮が効く状況」は原理的に観測できない可能性が高い。

### A-3. 比較設計の非対称性（軽微）

CV は n=1 で検定外、CV の ADE は single-shot で best-of-20 と非可換（論文内で明示済み）。CV に観測ノイズを与えて20ラン化すれば表が対称になる。ジャーナル拡張ではシード数 50〜100 への増強が望ましい。

### A-4. 衝突判定の幾何が想定車両形状と不整合（論文未認知・本分析の新規指摘）

実装上、自車の衝突判定・MinDist・TTC はすべて**車両中心の単一円**（`ego_radius`=1.0 m）に基づく:

- 判定: `frenet_planner.py:820`（`inflated_radius = robot_radius + obstacle_radius`）
- 指標: `data_structures.py:320-326` — MinDist は**中心間距離**、衝突は中心間距離 < 1.2 m

一方、車両は 4.5×2.0 m の長方形として描画・想定されている（`animator.py:349`、論文 Fig. 3 キャプション）。半長 2.25 m に対し判定円 1.0 m では前後端がカバーされず、S2 の MinDist 1.69〜1.85 m（中心間）は縦方向なら矩形 footprint 内に入り得る距離である。「衝突ゼロ」の頑健性が footprint の取り方に依存している。

**解決策:** 車両を 2〜3 個の円でカバーする multi-circle footprint（Frenet プランナの定石）へ置換。ベクトル化済みの衝突判定に円中心オフセットを追加するだけで実装可能。**ジャーナル化前に必ず再検証すべき**。

---

## B. シミュレーション環境・ドメインギャップの課題

### B-1. ETH/UCY→SFM の 3〜4 倍の誤差ギャップ（論文認知・対処は future work のまま）

このギャップ自体は文献的に正常: cross-domain 劣化は T-GNN（Xu et al., CVPR 2022, arXiv:2203.05046）以降「trajectory domain shift」として定式化されており、引用により所見をノーマライズできる。

**解決策（軽い順）:**
1. **プロンプトチューニング型適応**: Latent Corridors（Thakkar et al., ECCV 2024, arXiv:2312.06653）— SGAN を凍結したまま <0.1% のパラメータでシーン適応。real→synthetic 方向の適用実績あり。
2. **SFM 生成軌道での few-shot ファインチューニング**（Li et al., IROS 2024, arXiv:2309.10121 の逆方向適用）。
3. **予測分布のキャリブレーション評価**: SGAN の20サンプルのカバレッジを測定（Cheraghi Pouria et al., arXiv:2603.10407 — 較正の不備が ADE と独立に計画品質を損なうことを実証）。C-1 の「分布情報 vs 保守性」問題と直結。

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
- CVaR 拡張は20サンプルでは尾部推定の分散が大きい（MMD-OPT, arXiv:2412.09121）。サンプル増か GMM/RKHS 圧縮（Ren et al., IEEE L-CSS 2023）の併用が安全。
- プランナが慎重になると歩行者の反応分布も変わり較正が崩れる「endogenous distribution shift」（arXiv:2511.11567）は Limitations で言及する価値がある。

### C-2. 運動学的直接代入モデルと S3 横方向アーチファクト

論文の診断（スプライン曲率変化率ピーク＋無フィルタ状態代入＋状態マシン介入の結合）は説得的。解決は2層:

- **(a) 参照経路側**（数日規模）: 曲率連続なクロソイド/弧長パラメータ化への置換、またはウェイポイント再配置で dκ/ds ピークを除去。`NUMERICAL_VERIFICATION.md` の |d|max=2.166 m を回帰指標に使える。
- **(b) 追従側**（数週間規模）: Kong et al.（IV 2015）の kinematic bicycle + 操舵レート制約付き MPC 追従層。乗り心地指標（RMSJerk が S3 で全手法約2倍）が整備済みのため、検証パイプラインは揃っている。

### C-3. 分布対応計画のリアルタイム性

論文は「ベクトル化で20倍負荷を償却可能」と見込みのみ。**解決策:** 実装して実測する（静的障害物で実績のあるバッチ判定の拡張）。サンプル→GMM 圧縮で判定を解析的制約に置換する経路（arXiv:2502.10585）はサンプル数への線形依存自体を消せる。

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

## 検証メモ

- コードに関する主張（A-4 の衝突幾何、MinDist の中心間距離定義、★1/★2 の実装箇所と工数見積もり）は 2026-06-10 に実装と突合済み。
- 知識カットオフ以降の文献のうち本文の論拠となる arXiv:2510.14677、arXiv:2603.10407 はarXiv上で実在・内容一致を確認済み。その他の文献も調査時にarXiv/出版社ページで確認済み。
