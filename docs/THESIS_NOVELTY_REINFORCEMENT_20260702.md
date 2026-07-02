# 修論 新規性強化レビュー：新鮮コードレビュー＋文献検証＋強化提案

**目的**: 「研究の新規性が少し弱い」という課題に対し、(1) 研究コードを新鮮な目で入念にレビューし、(2) 既存研究を敵対的に再調査（既存メモ・方針文書に依拠せず独立実施）した上で、確実に新規性を主張できる進め方を確定する。
**作成日**: 2026-07-02
**方法**: 独立エージェント7本 — コードレビュー2本（コア sim/planning・評価/統計パイプライン）＋文献調査3本（閉ループ評価妥当性 約35本／SFM 較正・歩行者-車両相互作用 約40本／予測×計画統合 約25本）＋新出 load-bearing 文献9件の一次ソースファクトチェック1本＋コード上の重要指摘3件の直接スポット検証。
**先行文書との関係**: `docs/THESIS_NOVELTY_DIRECTION.md`（2026-06-24 確定方針）の主方向を**追認**しつつ、未収載の競合・手元の弱点・強化策を追加する。`docs/CODE_REVIEW_axisA_20260619.md` / `docs/CODE_REVIEW_full_20260623.md` の既指摘は重複報告しない（本書は「その後に残っていたもの」のみ）。

---

## 0. 結論

> **確定方針「歩行者-AV 閉ループ評価の妥当性研究」の新規性は生存。ただしそのままでは弱い。**
> 理由①: 敵対的探索でも完全一致（kill）は不在だが、包囲網は急速に狭まっている（車両版の同型空白は 2025-10〜2026-06 の8ヶ月で埋まった。歩行者版の窓は推定 12–18 ヶ月）。
> 理由②: コードレビューで確定した現在の最大の弱点は、**較正がどの held-out 指標でも手調整（AVEC）値に勝てていない**こと（§1 F1）。「器具」の説得力自体が不足している。
>
> **提案＝方針転換ではなく「器具の修理と実データ接地」による強化**:
> (A) 較正ハーネスの構造的欠陥（速度キャップ）を修正した上で**分布マッチング較正＋識別性分析**を第2の貢献に（文献上の実ギャップ・どちらに転んでも出版価値）。
> (B) VCI 実遭遇ジオメトリ上で **replay 実歩行者 vs 較正 SFM の閉ループ比較**（「Closing the Loop の歩行者版」＝完全な空白）を旗艦実験に。
> (C) 審査防御の小修正群（LOSO コーナー arm・対応検定・H1 枠組み・medoid 文言 等）。

---

## 1. コードレビュー所見（研究妥当性の観点）

過去レビューの通り **Critical 級の実バグはゼロ**。以下は設計レベルで研究主張に直結する新規発見。F1–F5 の裏付け箇所は本レビューでファイルを直接開いて確認済み。

### 1.1 主張の成立を左右する発見（F1–F5）

**F1.「較正の成功」を支持する held-out 証拠が存在しない（最重要）。**
- LOCO held-out ADE: 較正 0.640±0.205 vs AVEC 手調整 **0.639**±0.207（LOSO: 0.645/0.643）。
- pooled closest-approach KS: 較正/AVEC/斥力なしの3アームで**数値が飽和し同一**（0.462）。
- DUT 汎化: single (n=9) ADE 0.384 vs AVEC **0.381**。multivehicle (n=58) KS は **AVEC の方が良い**（0.276 vs 0.293）。
- 唯一一貫するのは「斥力なしが最悪」（CITR ~5% / DUT ~4.4% の ADE 差、KS 最悪）のみ。
- **帰結**: 「実データ較正が手調整の人工物を*是正する*」は現データでは主張不能。対称な読み方 —「結論は反応パラメータに感度があり、かつ**どのパラメータ集合も実データで妥当性検証されていない**」— しか成立しない（既存 M7 caveat の徹底）。これが新規性を弱くしている根本。

**F2. 較正ハーネスの速度キャップが識別性欠如の機構的原因である可能性。**
- `_set_cruise_speed`（`src/simulation/calibration_harness.py:200-212`）が pysocialforce の `max_speeds` を各歩行者の**記録中央値速度に固定**。巡航速度で歩いている歩行者は回避のために**加速することが構造的に不可能**で、斥力応答がキャップに飽和する。
- これは (a) v0 の ADE 平坦谷（C2 識別不能）と (b) standoff 過小再現（+0.68 m）の**両方を説明しうる仮説**。
- さらに閉ループ側（AVEC/RQ1b）は `max_speeds = 1.3×初期速度` という**別レジーム**（`integrated_simulator.py:93-104`）で動く＝較正値の移植先が較正時と異なる応答上限を持つ。ハーネスの docstring はこの点を開示していない。
- **帰結**: 「standoff 過小再現＝SFM の構造的限界」という現在の解釈はハーネス人工物かもしれない。安価に検証可能（§3-A ①）。

**F3. LOSO 較正不確実性が RQ1b の掃引域の外。**
- `outputs/rq2_evaluation/folds_loso.csv` の v0 = 2.617 / 1.947 / 1.849 / 1.762 に対し、RQ1b の ±1SD 箱は v0∈[1.542, 1.820]（`examples/run_rq1b_sensitivity.py:67-76`）＝ **4 fold 中 3 が箱の外**。σ も最小 0.743 < 下限 1.040。
- リポジトリ自身が「正直な安定性の見出しは LOSO」と宣言（`summary_loso.txt`）している以上、「robust 利得は較正不確実性に頑健」の掃引域には穴がある。M6 注記は radius-0.30 再較正点（箱の内側）しか確認していない。
- **対処**: LOSO コーナー（σ≈0.74, v0≈2.6）の GT アームを1本追加（§3-C 1）。

**F4.「単一サンプル」アームは実は medoid-of-20。**
- `predict_single_best`（`src/prediction/trajectory_predictor.py:340-353`）は 20 サンプル生成→**平均に最も近い1本**を返す（シーンレベル medoid）。シナリオ YAML の `num_samples: 20` が非 DA ベンチマークにもそのまま効く。
- 分散抑制済み・モード探索的な代表値であり、「SGAN の1描画で計画すると危険」という文言は測定内容と一致しない。robust 利得の主張には保守的な方向（medoid 相手でも利得が出ている）だが、CV は決定論なので手法間で非対称。provenance 列にも未記録。
- **対処**: 文言修正＋可能なら「真の1サンプル」アームを追加（§3-C 4）。

**F5. pooled KS の統計仕様ミス（修正すると主張が強まる）。**
- RQ2 見出しの p=0.007 は、**同一26遭遇**由来の sim/real 対標本に独立2標本 KS（`_pooled_ks_stat`、`run_rq2_evaluation.py:281-296`）を適用したもの。sim_i と real_i は遭遇ジオメトリを共有し強く従属。
- 正しい単位は per-encounter の対応差：26 fold 中 24 で real>sim（符号検定なら p≈2e-6 相当）＝**修正後の方が忠実度ギャップの主張は強くなる**。
- **対処**: 対応検定（符号 / Wilcoxon）を committed 成果物に追加（§3-C 2）。

### 1.2 その他の設計レベル所見（要対処・重要度順）

1. **フェイルセーフ調整軸が未掃引**: 状態機械の定数群は S1–S3 に対して経験調整されたことがコード内で明言（`integrated_simulator.py:536-549`、`scenario_01.yaml` の "verified" 注記、`config/__init__.py:322-325`）。RQ1b は σ/v0/cruise しか振っておらず、「結論は反応パラメータに頑健」は「フェイルセーフ調整に頑健」を含まない。妥当性ストーリー最大の未検証次元。
2. **S2 弁別セルは実質 σ コントラスト**: AVEC S2 は σ=0.3/v0=2.1、較正は σ≈1.16/v0≈1.68 ＝ v0 はほぼ同じで反転は斥力**範囲**（σ 0.3→1.16）が駆動。一方、識別性の議論（C2）は v0 中心で、σ 側の識別性分析は未実施。最重要セルが「検証の薄い方のパラメータ軸」に載っている。
3. **台帳を迂回する未補正 p 値**: (a) `verdicts.csv` の集計 Fisher 8 検定 — 特に avec の `lstm_danger_holds=True`（p=0.0287、未補正のまま REPORT の verdict 表に表示。8 検定 BH では非生存）。(b) DUT multivehicle KS（p=0.013/0.024/0.0019）。(c) 各アームの `welch_vs_baseline.csv` 電池。修論に載せるなら ledger への sidecar 化 or 有意性表現の降格が必要。
4. **H1 の主枠が非正準指標に依存**: 「シーン間で序列が反転（cv が hotel/univ 最良・eth/zara1 最悪）」は scene-level joint best-of-N（M1 が非正準と認定した指標）でのみ成立。正準 per-agent minADE では反転は hotel の 0.0057 m 差のみに縮む。**より強く指標頑健な言明**が同じデータから可能: 「sim は CV を ADE 最良と順位づけたが、実データ（正準指標）では学習系が全集計で明確に優位（cv 0.534 / lstm 0.333 / sgan 0.363）」。こちらを主枠に。
5. **較正点が3種併存**: (1.2005, 1.6219)（cruise CSV・DUT CLI 既定）／(1.156, 1.681)（RQ1b GT・DUT CSV）／(1.168, 1.712)（現行 LOCO 正準）。RQ1b のみ M6 注記あり。執筆前に正準点と注記の統一が必要。
6. **claim-① の判定は無検定の点平均優越**: `margin_verdict` は per-cell 平均比較でノイズゲートなし（数 cm 差のセルあり）。結論は S2 の大差で安定と思われるが、「検定不要の決定的判定」という ledger の表現はブートストラップ未実施の点で過大。
7. **予測失敗フォールバックが特権的 CV**: 失敗時は GT の SFM 速度で外挿（`integrated_simulator.py:477-496`）＝ CV *手法*（0.4s 間隔の有限差分）より正確な情報を使い、かつ当該ステップは ADE/NLL から silently 除外（≤4 連続失敗は無記録）。
8. その他（開示済み or 二次的）: 同時刻衝突判定の1サブステップ位相ずれ（≤0.15 m・手法対称）／点サンプリング衝突判定のトンネリング余地（上限速度近傍のみ）／歩行者半径3種（agent 0.30 / obstacle 0.2 / ped 0.2）の非連動／ウォームアップ中の停車 ego への事前回避／timeout 打ち切りの censoring／SGAN 末尾外挿クランプの手法非対称。

### 1.3 主張の棚卸し（現 outputs が支持する範囲）

**強く支持（審査で守れる3点）**:
1. **robust 利得の反応パラメータ頑健性**（RQ1b claim-①）: 全4 GT・1980 ラン・margin キャンペーン robust 衝突ゼロ。機構は大マージン（S2: robust 17.7 s / min_dist 1.86 vs inf1.00 25.1 s / 10/20 衝突）。※LOSO 域（F3）の但し書き or 追加 arm が条件。
2. **sim→real での予測序列の非転移**（RQ1a/H1）: sim は CV を ADE 最良としたが、ETH/UCY の正準 per-agent minADE では学習系が全集計変種・4/5 シーンで明確に優位。SGAN(Pooling) は LSTM に全シーン・ADE/NLL とも劣る。
3. **計装された負の結果**（RQ2）: ADE 較正は v0 に弱識別（0.640 vs 0.639 の平坦谷・cruise 交絡の定量否定つき）、standoff +0.68 m 過小再現（24/26 fold）、onset KS=1.0 完全分離。

**弱い/不支持**:
- claim-②「分布なし計画は危険」: 唯一の清潔セル S2/avec p=0.0078 は (a) pseudo-replication で反保守的、(b) family 定義に敏感（m=3/6 生存・m=12/13 非生存）、(c) 事後選択された family。しかも 3 m/s では **CV 自体はほぼ衝突しない**（信号は lstm/sgan single が担う）＝「AVEC の CV 危険結論が人工物」という文は二重に過大（CV でない・AVEC 動作点 6/5/5 m/s は calib 下で未再走）。
- 「較正が手調整に勝る」: 不支持（F1）。

### 1.4 審査攻撃点（想定問答の核）

1. 「較正はどの意味で成功したのか。較正 GT 下の結論を手調整 GT 下より信頼すべき理由は？」→ 現データに肯定的回答なし。対称な読み（感度＋全パラメータ未検証）に徹するか、§3-A で器具を修理する。
2. 「claim-② は境界セル1個」→ ledger の通り「示唆・境界・反応モデル依存」以上を主張しない。
3. 「実歩行者の挙動と閉ループ安全結論を結ぶ実験が1つもない」（証拠連鎖の断絶: ETH/UCY は車両なし・較正は 26 遭遇 2.4 m/s・閉ループは合成幾何 3 m/s）→ §3-B の replay 対照つき実データ接地閉ループが部分修復。

---

## 2. 文献調査の結論（一次ソース検証済み）

### 2.1 判定

**完全一致（kill）は不在**。4要素の交差点 —「(i) 実車両-群衆データ較正の (ii) 反応的（閉ループ結合）歩行者モデルを (iii) 統制器具として (iv) ベンチマーク結論（予測手法序列・危険性判定）の頑健性を検定」— を同時に満たす先行は約60本の精査で見つからなかった。
**ただし時間的圧力が最大リスク**: 車両版の同型空白は When Planners Meet Reality (2025-10) → nuPlan-R (2025-11) → ReactSim-Bench (2026-06) の8ヶ月で埋まった。WOSAC 陣営も VRU 反応シミュレーションを未解決と公言しており、歩行者版の窓は推定 **12–18 ヶ月**。

### 2.2 THESIS_NOVELTY_DIRECTION.md に未収載の競合（本レビューで追加・全て一次ソース確認済み）

| 先行研究 | 内容（検証済み事実） | 残る差別化 |
|---|---|---|
| **HABIT** (arXiv:2511.19109, WACV 2026) | 30k 汎用モーションプールから厳選した **4,730** 歩行者モーションを CARLA に配置。Leaderboard でほぼ無衝突の e2e エージェントが **InterFuser 5.24 衝突/km**（最大 TransFuser 7.43・BEVDriver 7.19） | 歩行者は**トリガー式再生**（発火は ego 接近依存だが動作はクリップ再生）＝ ego へ連続反応しない。実データ**較正**なし。e2e 絶対性能の暴露であり**手法間相対判定の頑健性分析ではない** |
| **Prédhumeau et al., JAIR vol.73 (2022)** | SFM＋予期的意思決定。CITR 定量較正・**DUT は定性評価**・**汎化検証は Nantes**。AV テスト利用は将来課題として言及 | **AV は録画軌道（ground truth）で駆動**＝プランナなし。プランナ比較・妥当性検証なし |
| **Agrawal, Dengler, Bennewitz** (arXiv:2409.14844, 2024) | **JRDB 学習**の Social Robot Force Model を器具に、ナビアルゴリズム別の歩行者軌道偏差をベンチマーク。「客観計測が可能という **proof of concept**」と自称 | 屋内ロボット・ベンチマーク結論の頑健性は不問。**機構レベル最近接**（較正済み反応 SFM の器具化）として引用必須 |
| **arXiv:2602.16035** (2026) | nuPlan 閉ループ（**反応エージェント**）で CV / Trajectron++ / HAICU を UA-SMPC に接続し比較。「**CV baseline は competitive**」（nuPlan シナリオの near-constant-velocity 性が説明。長ホライゾン CLS は HAICU 最良の留保付き） | 車両ドメイン。「CV 競合的」の**閉ループ先行**として引用必須 |
| **ReactSim-Bench** (arXiv:2606.14058, 2026-06) | world model シミュレータ自体の反応性を 2,636 シナリオでベンチマーク | 車両のみ。「評価器具の品質測定」というメタ問題意識の傍証 |
| **ED-Eva / Measuring What Matters** (arXiv:2512.12211, Honda RI, 2025-12) | シナリオ重要度適応の統合予測スコアを 9,059 シナリオで計画性能相関により検証 | 車両中心。**K1（統合指標の新提案）飽和の追加証拠** |
| Rashid et al., SIMPAT 2024 ／ AAP 2025 仮想安全試験 ／ ESV 2023 AEB 認知歩行者 | 較正 SFM＋単一プランナ統合デモ／歩行者能動回避の有無で衝突 27.77% 差／歩行者モデル選択が AEB 評価に影響 | いずれも単一システム・プランナ間比較なし・較正器具化なし。「歩行者反応性→評価結果」の傍証群 |

### 2.3 既収載競合の追認（一次ソースで再確認）

- **When Planners Meet Reality** (arXiv:2510.14677): SMART で IDM を置換し **14 プランナ**再評価、「IDM は計画性能を過大評価」。**全文に pedestrian/VRU/bicycle の言及ゼロを確認**（「歩行者は log-replay のまま」と断定引用せず「車両間のみで VRU 言及なし」と書くこと）。
- **Closing the Loop** (arXiv:2505.05638): 「*In this work, we use non-reactive agents to ensure better comparability and isolate the effects of different motion predictors*」の原文を取得＝**反応性交絡は明示的に回避されている**。
- nuPlan-R (2511.10403)・Ye et al. Transportmetrica A 2026・Wang & Markkula (2601.02082)・What Truly Matters (2306.15136)・Tartu (2410.16864)・Schöller RA-L 2020: 従来の位置づけ通り（`THESIS_NOVELTY_DIRECTION.md` §3 の差別化表は有効）。

### 2.4 文献上の実ギャップ（新手法貢献の余地・§1 の弱点と噛み合う）

1. **歩行者-車両 SFM 較正の識別性分析が不在**: 車両追従では Punzo, Ciuffo, Montanino (TRR 2315, 2012, DOI:10.3141/2315-02) が較正の局所解・設定依存性を体系実証した定番参照（S2 被引用 132）だが、歩行者-車両斥力の対応物は未確認。最近傍は Kretz et al. (TRR 2018, arXiv:1801.00276) の一般論と Gödel et al. (Safety Science 2022) の群衆（車両なし）ベイズ事後分布。手元には既に「v0 の ADE 平坦谷」の実証がある。
2. **closest-approach / onset の分布形一致（KS/EMD）を fitting 目的にした車両-歩行者較正が未確認**: 最接近の Ye et al. 2026 も**指標の平均誤差**（RPETE 等）の多目的であり分布形マッチングではない・識別性分析なし・プランナなし。群衆動力学の ABC/SBI 系譜（arXiv:2001.10330、Gödel 2022、arXiv:2602.05246）は車両-歩行者斥力に未適用。
3. **反応的歩行者下での予測器閉ループ順位付けが空白**: Closing the Loop は非反応を明示選択、2602.16035 は車両。歩行者では誰もやっていない。

### 2.5 キルリスト更新（回避すべき飽和方向）

`THESIS_NOVELTY_DIRECTION.md` §1 の判定は**全て追認**（④較正不確実性→計画は反証済みを再確認）。追加の飽和証拠: K1（統合指標）に ED-Eva (2512.12211)、K2（conformal→計画）に Interaction-aware CP (2502.06221)・AdaptNC (2602.01629)・SPARC (2410.15660)、K3（chance-constrained/contingency 手法貢献）にレビュー論文 (2601.14880) が既出＝成熟の証拠。K5（予測器の閉ループ**訓練**）は計算資源勝負（2603.24155 等）で回避。

---

## 3. 強化提案

方針の背骨（測定妥当性研究・P1–P3）は維持し、以下の3本で強化する。優先順。

### (A) 器具の修理 →「識別性を監査した分布マッチング較正」を第2の貢献に【2–3週間規模】

手順:
1. **速度キャップの切り分け（F2）**: `max_speeds` を解放 or 較正対象化して再較正し、standoff 過小再現が**ハーネス人工物か SFM 構造限界かを判定**する。閉ループ側とのレジーム整合（1.3× vs 中央値固定）も統一。
2. **分布マッチング較正**: rollout-ADE に closest-approach／avoidance-onset 分布の KS（or EMD／エネルギー距離）を加えた多目的較正を実装。「分布目的を足すと v0 の谷が立つか（識別性の回復）、standoff 再現が改善するか」を測る。`--interaction-distance`（現状 None＝希釈 fit）の非希釈設定も併用。
3. **識別性分析の正面論文化**: profile likelihood／損失面幾何（`plot_rq2_loss_surface.py` 資産）で (σ, v0) の縮退を明示し、「どの相互作用統計を目的に足せば縮退が解けるか」を系統比較。**σ 側**（S2 弁別セルの実駆動軸）を必ず含める。

**これが最善手である理由 — どちらに転んでも勝ち**: 改善すれば「較正が手調整に勝つ」証拠が初めて生まれ（F1 解消）、器具の説得力が立ち、方法論貢献（ギャップ 2.4-①②の同時充足）になる。改善しなければ「ADE でも分布目的でも SFM は実回避を再現できない」構造的限界の実証となり、妥当性研究の主張（ベンチマークの誤差棒）がむしろ強化される。

### (B) 実データ接地閉ループ ＝「Closing the Loop の歩行者版」を旗艦実験に【3–4週間規模】

VCI（CITR/DUT）の**実遭遇ジオメトリ上で ego をプランナで駆動**し、歩行者を**反応性軸**で入れ替える:

> replay（記録された実歩行者・非反応） ↔ 較正 SFM（反応） ↔ 手調整 SFM ↔ 斥力なし
> × 予測手法 {cv, lstm, sgan} × 計画方式 {真の単一サンプル, robust}

効果:
1. Closing the Loop が「比較可能性のため」に明示回避した**反応性交絡そのもの**を、歩行者で初めて正面測定する完全な空白。
2. replay アームは「実際の歩行者がした行動」という**非 SFM 参照点**＝最大の審査攻撃点「実歩行者と閉ループ安全結論を結ぶ実験がない」（within-SFM-family 循環性）を部分修復。
3. 技術的には `ReplayPedestrianSource` と較正ハーネス（録画 ego×SFM 歩行者の**逆構成**）が既にあり、新規実装は「プランナ ego×replay/SFM 歩行者」の結線が主。
4. 合成 S1–S3 の RQ1b は「感度分析」として従属化し、実データ接地版を見出しに昇格。

### (C) 審査防御の小修正群【数日規模・執筆前に必須】

1. LOSO コーナー GT アーム（σ≈0.74, v0≈2.6）を RQ1b に追加（F3）。
2. pooled KS を per-encounter 対応検定（符号/Wilcoxon）に置換 — 主張はむしろ強まる（F5）。
3. H1 の主枠を「sim は CV 最良 ↔ real は学習系優位（正準 per-agent minADE）」へ組み替え（§1.2-4）。
4. medoid 問題の文言修正＋「真の1サンプル」アームを (B) に含める（F4）。
5. フェイルセーフ定数の感度をサブセットで1回掃引（§1.2-1）。
6. 較正点3種の正準化注記・台帳外 p 値の sidecar 化 or 降格（§1.2-3,5）。
7. **現代的予測器を1本追加**（例: Trajectron++）— 2602.16035 が閉ループで CV/Trajectron++ を既比較しており、「SGAN 旧式」批判の封じ込めは必須に近い。

### 3.4 主張文（この形なら守れる）

> 「実車両-群衆データで較正し**識別性と分布忠実度を監査した**反応的歩行者モデルを統制器具として、歩行者-AV 閉ループ評価のベンチマーク判定（予測手法序列・単一サンプル計画の危険性・robust 計画の利得）が反応モデル仮定にどこまで依存するかを、**実遭遇ジオメトリ上の replay 対照付き**で系統定量した初の研究」

関連研究章で必ず潰す5線:
1. **vs HABIT**: 再生 vs 反応（閉ループ結合）・e2e 絶対性能 vs 手法間相対判定の頑健性・非較正 vs 実データ較正＋識別性監査。
2. **vs When Planners Meet Reality / nuPlan-R**: 車両のみ（VRU 言及ゼロ）・ブラックボックス学習エージェント置換 vs **解釈可能パラメータの較正度を独立変数化**・プランナ序列 vs 予測手法序列。
3. **vs Closing the Loop**: 意図的非反応 log-replay vs 反応性の正面測定。
4. **vs Ye et al. 2026 / Prédhumeau 2022**: 較正・行動再現止まり（AV は録画駆動）vs 器具化＋プランナ評価＋識別性。
5. **vs Agrawal 2024**: 屋内ロボット・影響測定 PoC vs 車両・ベンチマーク結論の頑健性検定。

### 3.5 投稿戦略

- **IV/ITSC 2027（締切 2026 年秋〜2027 年冬）に会議版を先行投稿**（時間窓 12–18 ヶ月への対処）。
- ジャーナル版（IEEE T-ITS / TR Part C）で (A)+(B)+既存 P1–P3 を統合。
- 教授提案の「統合指標」は従属要素として維持（単一スコア化が人工物性を隠す実証材料。`THESIS_NOVELTY_DIRECTION.md` §6 の位置づけを踏襲）。

---

## 4. 引用時の注意（ファクトチェックで判明した罠）

| 文献 | 罠 | 正しい書き方 |
|---|---|---|
| HABIT | 「30k の歩行者モーション」は誤り | 「~30k 汎用モーションプールから 4,730 の交通適合歩行者モーションを厳選」。5.24=InterFuser、7.43=TransFuser（最大） |
| Prédhumeau JAIR 2022 | 「DUT で汎化検証」「SPACiSS 提供」は不正確 | 汎化検証は **Nantes**（DUT は定性評価）。**SPACiSS の名は JAIR 論文に存在しない**（Pedsim ros 実装）— SPACiSS を引くなら AAMAS 2021 デモ等を典拠に |
| Punzo TRR 2012 | abstract に「non-identifiability」の語はない | 「局所解の存在・較正設定依存性・結果の信頼性を実証」の文言で引用 |
| When Planners Meet Reality | 歩行者の扱いは本文で明示されていない | 「歩行者は log-replay のまま」と断定せず「車両間相互作用のみを扱い VRU への言及がない」と書く |
| arXiv:2602.16035 | 「CV が閉ループ最良」ではない | 「competitive（jerk 最小タイ・長ホライゾン進捗最良）、ただし長ホライゾン CLS は HAICU 最良。nuPlan シナリオの等速性が説明」 |

**UNVERIFIED（要一次確認のまま）**: DynADE/DynFDE の原典 Wu et al.／WOSAC 参加系の VRU log-replay 慣行（二次情報のみ）／HABIT・nuPlan-R・When Planners Meet Reality の最終査読 venue（HABIT は著者申告 WACV 2026）／「AEB への歩行者モデル影響」Academia.edu 掲載論文の掲載誌。

---

## 5. 次アクション（推奨順）

1. **(C) 小修正群から着手**（最低リスク・執筆前必須）: LOSO arm → 対応検定 → H1 枠組み → 文言・注記整理。
2. **(A)-① 速度キャップ切り分け実験**（最高情報価値）: 結果次第で (A)-②③ の設計が決まる。
3. **(B) 実データ接地閉ループ**の結線実装 → キャンペーン実行。
4. 会議版（IV/ITSC 2027）ドラフト起こし — 関連研究章は §3.4 の5線＋ §2.2 の表をそのまま骨格に。

---

## 付録: 本レビューの主要参照（§2 で新規に追加したもの）

- Ramesh, Azer, Flohr. "HABIT: Human Action Benchmark for Interactive Traffic in CARLA." WACV 2026 (著者申告). arXiv:2511.19109
- Prédhumeau, Mancheva, Dugdale, Spalanzani. "Agent-Based Modeling for Predicting Pedestrian Trajectories Around an Autonomous Vehicle." JAIR 73, 2022
- Agrawal, Dengler, Bennewitz. "Evaluating Robot Influence on Pedestrian Behavior Models for Crowd Simulation and Benchmarking." arXiv:2409.14844, 2024
- "The Impact of Class Uncertainty Propagation in Perception-Based Motion Planning." arXiv:2602.16035, 2026
- Zhang et al. "ReactSim-Bench: Benchmarking Reactive Behavior World Model Simulation in Autonomous Driving." arXiv:2606.14058, 2026
- Da et al. (Honda RI). "Measuring What Matters: Scenario-Driven Evaluation for Trajectory Predictors in Autonomous Driving." arXiv:2512.12211, 2025
- Punzo, Ciuffo, Montanino. "Can Results of car-following Model Calibration Based on Trajectory Data be Trusted?" TRR 2315, 2012. DOI:10.3141/2315-02
- Kretz, Lohmiller, Sukennik. "Some Indications on How to Calibrate the Social Force Model of Pedestrian Dynamics." TRR 2018. arXiv:1801.00276
- Gödel et al. "Bayesian inference methods to calibrate crowd dynamics models for safety applications." Safety Science, 2022
- Rashid, Seyedi, Jung. "Simulation of pedestrian interaction with autonomous vehicles via social force model." SIMPAT 132, 2024
- "Pedestrian modeling with realistic dynamic behaviors and its application in virtual safety testing for autonomous vehicles." Accident Analysis & Prevention, 2025
- （既存方針文書の参照は `docs/THESIS_NOVELTY_DIRECTION.md` 主要参考文献を参照）
