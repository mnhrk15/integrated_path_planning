# 修論 新規性方針：確定提案

**目的**: 「現在の研究は新規性が薄い」という教授指摘に対し、自動運転研究全体を包括リサーチした上で、敵対的反証を生き残る確実な新規性方針を1つ確定する。
**作成日**: 2026-06-24（2026-06-25 ファクトチェック反映）
**根拠**: deep-research 2本（計28+28一次ソース）＋6候補方向の文献ダイブ＆敵対的反証ワークフロー（landscape 5角度・5方向が完走、各方向を「既にやられている」証拠探索で全力反証）。
**ファクトチェック**: 全 load-bearing 引用18件を canonical URL から実取得検証（捏造ゼロ／15件 VERIFIED・Farid 引用の取り違えは下記で修正・著者名2件訂正）。内部数値（σ=1.156/v0=1.681・S2 fisher_p=0.0078・standoff sim1.82/real2.50・356 test green）はリポジトリ実出力と一致を確認済。

---

## 0. 結論（確定提案）

> **主方向＝「歩行者-AV 閉ループ評価の妥当性(validity)研究」。**
> 問い: *AV のベンチマーク結論（予測手法の序列・「CV は危険」・robust 計画の安全利得）は、どこまで歩行者反応モデルの人工物か。実データ較正はそれをどこまで是正するか。*
>
> **貢献の種類＝新しい指標でも新しい計画器でもなく、「測定・ベンチマークの妥当性」貢献。** 既存の軸A（RQ1a/RQ1b/RQ2）を1つの首尾一貫した論文に束ね直す。

**なぜこれが確実か**: 5方向すべての敵対的反証が、この一点に残存新規性として収束した（下表）。単独技術は飽和だが、**「実データ較正された反応的歩行者モデルを制御器具として使い、閉ループ評価結論が較正依存の人工物であることを示す＋較正自体の限界を正直に定量化する」**という交差点を本質的に占有する先行研究は、実在検索で発見できなかった。**ただしこの「人工物」結果は SFM family 内の感度**である（閉ループの衝突相手歩行者も較正済み SFM＝実歩行者ではない）。外的妥当性は開ループ ETH/UCY（P1）に置き、「AVEC の斥力が過大」は ADE では弁別不能なので standoff KS に依拠する（§8 参照）。

---

## 1. 5方向の敵対的評価サマリ

| 方向 | 新規性 | 実現性 | 反証 | 最大の脅威（必ず差別化） |
|---|---|---|---|---|
| ②シミュレーション忠実度が序列を変える（方法論的警告） | moderate | high | **生存** (conf: 中) | When Planners Meet Reality (2510.14677), nuPlan-R, Closing the Loop (2505.05638) ＝**車両のみ** |
| ③計画タスク認識型の予測評価 | moderate | high | **生存** (conf: 高) | What Truly Matters (NeurIPS'23, 2306.15136), Tartu (2410.16864)＝歩行者だが**反応群衆/較正なし** |
| ⑤過保護・効率安全トレードオフ | moderate | high | **生存** (conf: 高) | Markkula 2601.02082＝人間らしい歩行者→過保護を実証だが**認知モデル・1対1・予測序列なし** |
| ①較正反応的歩行者を評価基盤に | moderate | high | **生存** (conf: 高) | Transportmetrica A 2026＝**同一データ(CITR/DUT)で SFM 較正**だが planner なし |
| ④較正不確実性→リスク認識計画 | **weak** | high | **反証された** | Rethinking Gaussian (2603.10407, 2026)・CCTR (AAAI'24)・DtACI (2508.05634)＝**ほぼ同一構成** |

**読み方**:
- **④は捨てる（主貢献にしない）**。「較正不確実性を計画に織り込む / conformal を直す」は Driggs-Campbell 研ら大規模研究室が既に占有（あなたが破棄した conformal の cross-scenario coverage 崩壊すら DtACI が解決済み）。→ *これは前回の metric memo で「案A＝最強」とした見立てを、より深い反証で覆した重要な訂正。*
- **②③⑤は同じ核に収束する**。①も「ベンチマーク/妥当性論文として書け」という結論で同じ核に合流。

### 全方向に共通して「反証しきれなかった」残存新規性（＝あなた固有の堀）
1. **実データ較正済みの反応的歩行者 SFM（DUT/CITR で σ,v0 系同定 ＋ closest-approach/avoidance-onset の KS 忠実度 ＋ LOCO/LOSO 安定性）を、閉ループ都市 AV 計画器評価の制御器具として用いる初の歩行者特化研究。** 較正系(Transportmetrica)は planner なし／planner 系(nuPlan-R)は歩行者 SFM 較正なし／social-nav 系は車両相互作用較正なし＝**垂直スタックの組合せが固有資産**。
2. **鋭い負の結果**: 手調整した反応パラメータ（AVEC 既定斥力＝**シナリオ別**：S1/S3 σ0.7/v0=3.5・S2 σ0.3/v0=2.1）が「CV は危険」結論を製造し、実データ較正(σ1.156/v0=1.681)で S2 の単一衝突が消えて結論が溶ける＝較正ノブを**実データアンカー＋null/手調整対照**付きでランキング/安全結論の反転に帰属させた、歩行者では文献に無い具体的・反証可能な知見。*注: 見出しの S2 では dose-response が 2.1→1.68 と小さく効果は控えめ＝§4 の bounded-claim に従う。*
3. **正直な部分的較正**: 較正後も standoff/closest-approach が系統的に過小再現される＋最適 v0 が ADE をほぼ動かさない(identifiability 欠如)＝既存 SFM 較正論文が見逃す失敗モードを、**ベンチマーク自体の誤差棒**として定量化。
4. **速度域外挿の明示**: 低速(1-3 m/s)較正 vs 運用(5-6 m/s)を限界として証明書に組込む方法論的開示。

---

## 2. 確定提案：貢献の定式化と3本柱

**論文タイプ**: benchmark / measurement-validity（NeurIPS D&B・IV・ITSC・T-ITS 系の評価妥当性論文の体裁）。

3本柱（**すべて既存データで実行済み**＝新規大規模実験ほぼ不要）:

- **P1（開ループ脚＝非循環の独立証拠）**: ETH/UCY で予測手法(cv/lstm/sgan)を ADE/FDE/NLL 評価 ← RQ1a/H1。閉ループの循環性を断ち切る独立軸。*注: CV が競合的・序列がシーン依存という所見自体は Schöller RA-L 2020 (1903.07933) と Tartu 2410.16864 で既出。ここは「土台の再確認」であって新規性の本体ではないと明記する。*
- **P2（閉ループ人工物＝見出し）**: 反応モデルを「実データ較正(σ1.156/v0≈1.68) ↔ 手調整(AVEC＝シナリオ別 v0=3.5[S1/S3]・2.1[S2]) ↔ 斥力なし(null)」の**較正忠実度連続軸**で振り、閉ループ結論（序列・「CV は危険」・robust 利得）がどれだけ較正に依存するかを dose-response として定量 ← RQ1b。*モデルクラス差し替え(IDM vs SMART)ではなく単一パラメトリックモデルの較正度を独立変数にする点が、車両論文が構造的に作れない固有の計器。ただし衝突相手歩行者も SFM＝結論は「SFM family 内の感度」（外的妥当性は P1 に置く）。*
- **P3（忠実度→妥当性の誤差予算＝信頼性の堀）**: 較正忠実度を KS(closest-approach/avoidance-onset)＋LOCO/LOSO で監査し、部分的較正(standoff 過小再現・v0 非可識別)が結論に与える不確かさを誤差棒として付与 ← RQ2。HABIT/social-nav ベンチマークが欠く「忠実度→妥当性の誤差予算」を持つ。

**3つのリサーチクエスチョン**:
- **RQ-A（中核）**: 都市交差点の閉ループ AV シミュレータで、予測手法の安全効率序列および「CV は危険」結論は、自車斥力反応モデルの較正に依存する人工物か？ 較正(σ≈1.16/v0≈1.68) vs 手調整(AVEC＝S1/S3 v0=3.5・S2 v0=2.1) vs null を比較し、KS 忠実度と衝突/効率指標の反転を**多重比較補正付き**で定量。
- **RQ-B（較正の限界）**: 実データ較正 SFM 自車斥力は、歩行者の社会的距離(standoff)と回避開始(avoidance-onset)をどこまで忠実に再現するか？ KS と v0 の可識別性(ADE 平坦谷)で、SFM 反応モデルの構造的限界(部分的較正)を特徴づける。
- **RQ-C（妥当性の境界）**: 低速(DUT/CITR ~1-3 m/s)較正の反応的歩行者は、どの車速・幾何の範囲で運用速度(5-6 m/s)の planner 評価基盤として妥当か。独立な開ループ ETH/UCY 証拠と結びつけ、「序列は反応性の人工物(閉ループ)」と「序列は予測品質の事実(開ループ)」を分離する。

---

## 3. 死活的に重要な差別化（関連研究章で必ず潰す）

acceptance を左右する load-bearing な引用。各々に「彼らは X、我々は Y」を1文で書くこと。

| 先行研究 | 彼らがやったこと | 我々の差別化 |
|---|---|---|
| **Transportmetrica A 2026** (DOI:10.1080/23249935.2026.2627403) | **同一データ CITR/DUT で SFM を ADE＋相互作用現実性の二目的較正**（あなたの RQ2 をほぼ先取り、しかもより強い較正） | 彼らは SFM を**行動モデルとして較正するだけ**で planner に入れない。我々は較正済みモデルを**評価基盤**として使い、planner 評価の較正依存性を示す |
| When Planners Meet Reality (2510.14677, 2025) | IDM→学習反応エージェント(SMART)で nuPlan の planner 序列が反転 | **車両背景エージェントのみ（VRU 反応は対象外）**。我々は歩行者・SFM・較正連続軸 |
| nuPlan-R (2511.10403, 2025) | 反応的基盤＝ベンチマーク貢献型を確立、SR/All-Core PR 追加 | 車両・歩行者 SFM 較正なし |
| Closing the Loop (2505.05638, 2025) | 開ループ序列が閉ループに転移しない | **非反応 log-replay**＝反応モデル人工物を問えない（我々が問える交絡） |
| What Truly Matters (2306.15136, NeurIPS'23) | ADE は運転性能と無相関、Dynamic ADE 提案 | 車両・SUMMIT・**実データ較正反応群衆ではない** |
| Tartu 2410.16864 (2024) | 歩行者予測器を AD 評価、dynamics gap・DynADE を**適用**（起源は Wu et al. 2023） | **開ループ log-replay・反応群衆なし・較正なし** |
| Markkula 2601.02082 (2026) | 人間らしい歩行者→AV 過保護を実証 | 認知モデル(SFM でない)・横断1対1・**予測手法序列の比較なし・KS は closest のみ** |
| Schöller RA-L 2020 (1903.07933) | CV が ETH/UCY で SOTA に競合 | あなたの開ループ H1 の**土台**＝ここは新規性ではないと正直に書く |

---

## 4. やってはいけない過大主張（bounded claims）

信頼性が貢献の本体なので、ここを外すと崩れる。

- ❌「開ループ≠閉ループを初めて示した」「CV が競合的だと示した」→ Schöller 2020・Tartu・Closing the Loop で既出。**新規性は較正を器具にした人工物の帰属にある**。
- ❌「劇的な序列反転」→ **あなた自身のデータの信号は控えめで family 定義に敏感**（集計 cv-danger 判定はノイズ、per-scenario S2 の fisher_p=0.0078 のみが清潔な discriminator）。最強の正直版は「**1つの清潔な弁別シナリオ(S2)＋多重比較台帳を伴う、境界づけられた慎重な実証**」。
- ❌「新しい統合評価指標を作った」→ 単一スコア統合は飽和・批判対象（前回 metric memo 参照）。指標は作っても**主貢献にしない**。
- ❌「閉ループの planner 序列反転が独立の真実」→ 閉ループは較正と機構を共有＝循環性。**独立証拠は開ループ(P1)に置く**こと。

---

## 5. 妥当性検証の方法論（借用して厳密化）

新指標/新主張の妥当性は、確立した方法論をそのまま借りられる（deep-research で確認）:
- **Tarko 2018**（良いサロゲートの三条件・aetiological consistency）
- **Johnsson et al. 2021** (AAP 161:106350)：**実事故記録なしの「相対的」検証**＝文献由来の ground-truth 安全ランキングを baseline に、サロゲートの弁別力を検証（min-TTC は再現するが PET は再現しない、を実証）。
→ あなたの「closest-approach KS・collision rate が反応モデル較正で序列を再現/弁別できるか」を、この相対検証フレームで論じれば、提案の妥当性が借り物の確立手法で支えられる。

---

## 6. 教授提案「統合評価指標」の位置づけ

捨てずに**従属的な構成要素**として活かす:
- 主貢献は上記の妥当性研究。統合指標は「閉ループ結論を要約する従属指標」として登場させ、**「単一スコアに統合すると人工物性が隠れる(集計は S3 汚染でノイズ)→ per-scenario 分解が必要」**という、まさに nuPlan-R/Bench2Drive の批判と同じ教訓を**あなたのデータで実証**する材料にできる。
- これにより教授の関心(統合指標)に応えつつ、新規性は妥当性側に置ける。

---

## 7. 実行計画（追加実験を最小化）

実現性 high の根拠＝load-bearing な結果は既に committed（356 test green・結果保全済み）:
- 残作業の大半は**執筆・framing・関連研究差別化**であって新規大規模実験ではない。
- 追加で価値が高い小実験: P2 の**較正忠実度を連続軸にする中間点**（AVEC手調整 と CITR較正 の間に部分較正点を1-2個＝feature-frozen 較正など）を `run_rq1b_sensitivity.py` の override で1-2スイープ追加。
- 公開データ(ETH/UCY/DUT/CITR)・既存コードのみ。大規模実車/収集は不要＝個人研究規模に完全適合。

---

## 8. リスクと限界（正直な開示＝信頼性の堀）

1. **信号が控えめ**: 過大主張不可。境界づけた慎重な実証に徹する（§4）。
2. **新規性は moderate**（strong ではない）: 修士論文としては十分だが、**関連研究差別化(§3)が acceptance の生命線**。各ピラー単独は飽和なので、必ず「交差点＋負の結果」として書く。
3. **速度域外挿**: 低速較正→運用速度評価は真の妥当性脅威。層別・運用速度制限・limitation 明記で対処。
4. **循環性＋ within-SFM-family 限定（最重要）**: 閉ループの「人工物」結果は、衝突相手の歩行者も**較正済み SFM（実歩行者ではない）**＝「SFM family 内の感度」であり外的検証ではない。加えて「AVEC の v0 が過大／較正モデルの方が real」という前提は**ADE 上は弱 identifiable（v0=1.68 と 3.5 が ADE で区別不能）**で、standoff KS（sim1.82 vs real2.50・~0.68m gap・忠実度 family で BH q=0.007）に依拠すべき。独立証拠は開ループ(P1)に置く。※根拠は `docs/CODE_REVIEW_axisA_20260619.md` の M7/C2/M4。
5. **残る研究判断**: multivehicle KS の統計独立性(single-vehicle 主＋caveat)・`_meanstd` ddof は実質クローズ済み。

---

### 主要参考文献
- Hagedorn et al. "When Planners Meet Reality: How Learned, Reactive Traffic Agents Shift nuPlan Benchmarks." arXiv:2510.14677, 2025
- Peng et al. "nuPlan-R: Closed-Loop Planning Benchmark via Reactive Multi-Agent Simulation." arXiv:2511.10403, 2025
- "Closing the Loop: Motion Prediction Models beyond Open-Loop Benchmarks." arXiv:2505.05638, 2025
- Tran et al. "What Truly Matters in Trajectory Prediction for Autonomous Driving?" NeurIPS 2023. arXiv:2306.15136
- Zabolotnii et al. "Pedestrian motion prediction evaluation for urban autonomous driving." arXiv:2410.16864, 2024（dynamics gap/DynADE は Wu et al. 2023 を適用）
- Ye et al. "Risk-aware pedestrian-vehicle interaction: a multi-objective Bayesian-optimised social force model." Transportmetrica A: Transport Science, 2026. DOI:10.1080/23249935.2026.2627403
- Yang, Johora, Redmill, Özgüner & Müller. "Sub-Goal Social Force Model (SG-SFM)." arXiv:2101.03554, 2021（CITR/DUT データセット）
- Wang, Dogar, Darling & Markkula. "Realistic adversarial scenario generation via human-like pedestrian model for AV control parameter optimisation." arXiv:2601.02082, 2026
- Schöller et al. "What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction." IEEE RA-L 2020. arXiv:1903.07933
- Ivanovic & Pavone. "Injecting Planning-Awareness into Prediction and Detection Evaluation." IEEE IV 2022. arXiv:2110.03270
- Nakamura, Tian & Bajcsy. "Not All Errors Are Made Equal: A Regret Metric for Detecting System-level Trajectory Prediction Failures." CoRL 2024. arXiv:2403.04745
- Farid, Veer, Ivanovic, Leung & Pavone. "Task-Relevant Failure Detection for Trajectory Predictors in Autonomous Vehicles." CoRL 2022. arXiv:2207.12380
- Tarko. "Surrogate Measures of Safety." 2018. doi:10.1108/S2044-994120180000011019
- Johnsson, Laureshyn & D'Agostino. "...relative validation of surrogate measures." AAP 161:106350, 2021
- （反証された方向の参照: Rethinking Gaussian arXiv:2603.10407 2026・CCTR AAAI'24・DtACI arXiv:2508.05634）
