<!-- 自動生成: ultracode マルチエージェント・ワークフロー review (143 agents, 検証済み) 2026-06-19 -->
<!-- 軸A(実データ較正)重点・研究ロジック妥当性主眼。修正は未適用(レポートのみ)。 -->
<!-- findings: total 47 / confirmed 45 / dismissed 2。Critical/Major/一部Minor は .venv 再実行で裏取り済み。 -->

# 研究コード徹底レビュー（軸A重点・研究ロジック妥当性）

## エグゼクティブサマリ

実装そのものの土台（AVEC コア sim/planner/state-machine、Replay 開ループ経路、座標/単位変換）は健全で、致命的なバグやリークは見つからなかった。`ReplayPedestrianSource` が ego を厳密に無視すること（回帰テスト pin 済）、固定母集団窓がフレーム穴を実質排除すること、fps=29.97 が記録 m/s 速度と整合することは、いずれも肯定的に確認できた。一方で、**研究主張の妥当性に効く欠陥は較正の identifiability と忠実度メトリクスの統計的解釈に集中している**。最重大は (1) LOCO 保留 closest-approach KS が n=1 縮退で `1.000±0.000` と表示され、calibrated/AVEC/no-repulsion で同一＝忠実度を一切弁別しない点、(2) RQ1a の ADE が scene-level best-of-N + CV 非対称により stochastic 手法だけを膨らませ序列結論を歪める点、(3) RQ2 較正値 v0≈1.68 が ADE 上 AVEC 3.5 と区別不能（弱 identifiability）＋goal 汚染/ped-ped 交絡/cruise バイアスが揃って v0 を下方に押す点。RQ1b の robust 利得（主張①）は全 GT で頑健に成立するが、CV 危険性（主張②）は単一桁の衝突カウント差・不均等シード・per-scenario narrative の自己矛盾を抱える。総じて「実装は概ね正しいが、報告される統計量の解釈と identifiability の主張が過大」というのが本レビューの核心所見である。

---

## Critical findings

### C1. LOCO 保留 closest-approach KS が n=1 縮退で `1.000±0.000`＝較正と「何もしない」を区別できない

- **file:line**: `examples/run_rq2_evaluation.py:197-213,253-255` / `src/simulation/calibration_harness.py:458-459,476`
- **影響RQ・主張**: RQ2 — 較正パラメータが実回避を再現するという held-out 忠実度の中核証拠
- **説明**: `fidelity_report` は各 encounter につき closest-approach を**1スカラ**しか作らない（`calibration_harness.py:458-459`）。LOCO は各 fold で clip を1本だけ保留し、CITR は 1 clip = 1 encounter なので、保留分布は常に長さ1。`compare_distributions_ks` の `ks_2samp` は size-1 同士で値が異なれば数学的に必ず 1.0 を返す。結果 `summary_loco.txt` は calibrated/AVEC-default/no-repulsion で `KS_closest : 1.000 +/- 0.000` を**完全に同一**に印字する（実機確認済）。LOSO でも 3 群で `0.725±0.277` がバイト一致。
- **なぜ研究妥当性に効くか**: 論文は closest/onset KS を「較正が物理的回避を捉えた」held-out 証拠として提示する。n=1 で 1.0 に縮退し、かつ no-repulsion ヌルと同値の統計量は何も検証しない。KS=1.0 は最大「不一致」であり、審査官には「強い一致」と誤読され得る方向の誤りでもある。
- **根拠（再実行・裏取り済）**: `folds_loco.csv` で全26 fold `n_test_encs==1`、`test_ks_closest`/`base_default`/`base_norepulsion` すべて unique==[1.0]。`ks_2samp([1.0],[2.0]).statistic==1.0`。`summary_loco.txt` の本レビュー再取得でも上記印字を確認。closest sim 1.820 vs real 2.502（sim<real が24/26 fold）。
- **推奨対応**: LOCO の per-fold KS は廃止。26 fold の保留 closest スカラを**全プールして1本の sim-vs-real KS**（n≥26、autocorrelation-clean）にするか、`mean_closest_sim vs real` の生差（~0.68m 過小）を正直な standoff 指標として報告。`1.000±0.000` 行は n=1 artifact である旨を明記。
- **確信度**: high（3検証者中、severity を Major に下げる意見が1件あったが「忠実度証拠として読ませる」フレーミングを保持する限り Critical 相当）

### C2. 較正 v0≈1.68 が held-out ADE 上 AVEC v0=3.5 と区別不能（0.640 vs 0.640）＝(σ,v0) は弱 identifiable

- **file:line**: `examples/run_rq2_evaluation.py:195-213,243-251` / `src/calibration/optimize.py:1-9`
- **影響RQ・主張**: RQ2（較正が意味ある distinct な (σ,v0) を同定）／RQ1b 循環性の前提
- **説明**: held-out ADE は calibrated=0.640、AVEC-default(0.7,3.5)=0.640、no-repulsion=0.672（`summary_loco.txt` 実機確認）。較正最適点と AVEC 手調整値は ADE 上**統計的に区別不能**（paired t p=0.94、Cohen's dz=0.014）で、斥力の有無のみが ~5% 動かす。`objective_rollout_ade` は全 frame/全 ped をプールし、損失の大半は goal-following が支配＝斥力信号が希薄な浅い谷。optimizer docstring 自身が v0 と 1/σ の confound ridge を認める。
- **なぜ研究妥当性に効くか**: 「実データは AVEC の v0=3.5 より弱い斥力を示す（＝論文は過調整）」という RQ2 ヘッドラインを支える metric が、1.68 と 3.5 を区別できない。RQ1b の「CV 危険性は AVEC artifact」論はこの「較正モデルは AVEC より real」という前提に依存するため、弱 identifiability は下流主張も揺らす。
- **根拠（裏取り済）**: `folds_loco.csv` 平均で calibrated 0.6403 / AVEC 0.6399 / no-rep 0.6719。pooled grid 最小 0.6568 at (1.156,1.68)、v0=3.5 列ベスト 0.6604（Δ0.0036）。standoff は sim 1.82 vs real 2.50 で両者とも過小再現。
- **推奨対応**: profile-likelihood/ridge 沿いの ADE 図で v0=1.68 vs 3.5 の差を fold-level 不確かさ付きで示し、ADE では弁別不能と明言。「AVEC v0 が高すぎ」主張は ADE でなく有効な pooled standoff KS に依拠させるか、「斥力は必要だが強さは軌道 ADE で弱 identifiable」へ緩和。
- **確信度**: high（検証2件 keep、1件 Minor 格下げ＝主張は崩れないとの判断だが、ヘッドライン解釈に効くため Critical 維持が妥当）

### C3. v0≈1.68 を下方に押す3つの系統バイアスが揃う（cruise/goal/ped-ped）— ただし損失上は低 v0 が最良

- **file:line**: `src/simulation/calibration_harness.py:75-146`（cruise）, `149-179`（_far_goals）, `307-351`（ped-ped 込み rollout）
- **影響RQ・主張**: RQ2 — 較正 v0 が「実回避強度」であるという力の帰属/identifiability
- **説明**: 3 機構が同方向（v0↓）に働く。(a) cruise 推定が遭遇全窓の中央値速度を取るため回避減速が desired speed に焼き込まれ斥力を弱く見せる。(b) `_far_goals` が録画（回避済）net 変位を goal heading にするため永続レーンシフトの回避が driving force に漏れ、斥力なしでも deflection を肩代わり（実データで goal heading が free-walk 基準から median ~11°、free-walk goal 再 fit で v0 1.62→~2.5 に上昇）。(c) rollout が ped-ped 社会力を含むため多人数 encounter（CITR は全26が N=8）で隣接反発が回避を吸収（ped-ped ablation で v0 1.62→1.90、+17%）。
- **なぜ研究妥当性に効くか**: 「実データはより弱い斥力で説明できる」という結論の頑健性に直接効く。3 confound が揃って低 v0 を作っているなら、低 v0 は「実歩行者が弱く避ける」ではなく推定器の系統的過小同定の可能性を排除できない。
- **根拠（裏取り済・重要な反証あり）**: cruise sweep（`cruise_sensitivity.csv` 実機確認）で de-bias 推定は v0 を 3.1–5.3 に押すが、**`loss_ade` は単調悪化**（baseline 0.657 が最良→upper_q85 0.899 が最悪、Pearson(v0,loss)=0.83）。すなわち「de-bias すると AVEC 3.5 に戻り過調整主張が反転」という強い主張はデータ自身の損失で反証され、cruise finding は Minor へ格下げ。goal/ped-ped は方向性が確認され Major 維持。
- **推奨対応**: cruise・goal・ped-ped の3感度を identifiability エンベロープとして併記し、低 v0 は3 confound と交絡する旨を明記。**ただし「de-bias で v0 が上がる＝そちらが正しい」とは結論できない（held-out ADE はむしろ低 v0 を好む）**点も正直に書く。single-ped/ped-ped ablation 較正を補助結果として提示。
- **確信度**: high（goal/ped-ped）／cruise の「過調整反転」部分は refuted 寄り

---

## Major findings

### M1. RQ1a ADE/FDE が scene-level (joint) best-of-N＝stochastic 手法だけを膨らませ序列を歪める

- **file:line**: `src/core/metrics.py:78-85`
- **影響RQ・主張**: RQ1a（cv<lstm<sgan 序列が sim 人工物か＝H1 の実データ再測）
- **説明**: `ade_samples = np.mean(displacement, axis=(1,2))`（ped 軸も平均）→`np.min` で**シーン全体に1サンプルを共同選択**。SGAN 標準 minADE は ped ごとに best-of-N を取る。joint min は ped ごとの sample 多様性を使えず、N>1 窓で系統的に大きく出る。CV は N=1 で best-of-N が恒等＝**膨張は lstm/sgan のみ**にかかる。
- **なぜ研究妥当性に効くか**: H1 は「序列がシーン依存で入替る＝人工物」を実データで判定する。非正準・手法非対称な metric は stochastic 手法の優位を機械的に縮め/反転させ、混雑窓ほど bias が大きく scene-dependent でもある。
- **根拠（裏取り済）**: 実 ETH/UCY で univ（80窓）scene-level cv 0.588/lstm 0.614/sgan 0.803 → per-agent で lstm<sgan<cv に**反転**。CV 膨張は全シーン 0%、lstm/sgan は ped/窓数に単調比例（eth +5.7%、univ +84–89%）。`tests/test_metrics.py:50-72` が scene-level を意図的に pin。
- **推奨対応**: per-agent best-of-N（`displacement.mean(axis=2).min(axis=0)`）へ変更するか、scene-level を論文で明示的に正当化し cv/lstm/sgan 序列が選択不変であることを再実行で確認。published SGAN minADE と scene-level 数値を比較しない。
- **確信度**: high（C1 と並ぶ RQ1a の最重大、検証3件すべて keep）

### M2. best-of-N の非対称性: SGAN/LSTM は best-of-20 オラクル min、CV は単一決定論＝ADE 比較がリンゴ対オレンジ

- **file:line**: `examples/run_openloop_prediction.py:97,150-151,171` / `src/core/metrics.py:82`
- **影響RQ・主張**: RQ1a（手法序列の比較公平性）
- **説明**: M1 と同根だが軸が異なる。SGAN/LSTM は full [20,P,T,2] を min、CV は `dist=None`→1サンプルで min 恒等。`--num-samples` 既定20。ADE は「SGAN/LSTM の20本中ベスト」対「CV の1本」を並べる＝SGAN 公式 minADE_20 プロトコルそのものだが、CV を同じ土俵に乗せられない。
- **なぜ研究妥当性に効くか**: 「CV が思ったほど悪くない/序列入替り」を主張する際、best-of-20 が確率的手法を系統的に有利化する一方 CV には無い。N を揃えないと予測器性能でなくサンプリング数差を測る。
- **根拠（裏取り済）**: zara1 で CV(N=1)=0.436 / SGAN N=1=0.621→N=20=0.366 / LSTM N=1=0.679→N=20=0.362。N=1 公平比較では CV が両学習器に勝つ。best-of-20 は学習器 ADE を ~2–3x 削減。ただし**同一プロトコルが AVEC sim baseline にも適用**されるため H1（序列入替り）自体は崩れない。
- **推奨対応**: minADE_20 vs CV-single の非対称を論文に明記し、可能なら SGAN/LSTM の mean-sample（N=1 等価）ADE も併記して「best-of-N 利得を除いても序列が保たれるか」を示す。
- **確信度**: high（検証で keep 2/downgrade 1。H1 を崩さないため Minor 寄りの見解もあり）

### M3. LOCO `±std` を安定性/SE として提示するが26 fold は訓練データを ~96% 共有＝過小精度

- **file:line**: `examples/run_rq2_evaluation.py:217-223,244-256`
- **影響RQ・主張**: RQ2（σ/v0 の実データ較正安定性）
- **説明**: 各 LOCO fold は26中25 clip で再 fit するため fold 間推定は near-replicate（92–96% 重複）で強く正相関。`±0.116/±0.139` は「clip 1本除去への鈍感さ」を測り、較正の sampling 不確かさではない。docstring は LOCO を「MAIN evidence … parameter-stability claim」とするが、系同定では LOSO（geometry holdout）が正しい安定性指標。per-scenario の honest spread は v0=1.21/2.02/8.13/2.05（6.7x）。
- **なぜ研究妥当性に効くか**: 「データが安定な (σ,v0) を pin する」が RQ2 中核。実際は geometry を変えると v0 が ~1.2→~8 に振れ、小 LOCO std は leave-one-of-many artifact。
- **根拠（裏取り済）**: jackknife SE = √(n-1)×raw spread = 5.0x（σ 0.116→0.58、v0 0.139→0.70）。LOSO std は LOCO の 2.4x。`lat_bi` v0=8.13 は grid 上端で損失がほぼ平坦＝essentially unidentified。
- **推奨対応**: 安定性は LOSO で headline、per-scenario v0 spread（1.2–8.1）を前面に。LOCO は sampling-jitter チェックとし、spread を「descriptive fold-to-fold range（SE ではない）」と明記。
- **確信度**: high（検証で keep 多数。1件 Minor 格下げだが mechanism は全員肯定）

### M4. Standoff が calibrated 点で系統的に過小（sim ~1.82m vs real ~2.50m、~30%）

- **file:line**: `examples/run_rq2_evaluation.py:258-259` / `src/simulation/calibration_harness.py:456-459`
- **影響RQ・主張**: RQ2（較正が実回避距離を再現）
- **説明**: 較正点で closest-approach は sim 1.820 vs real 2.502（LOCO）、sim 1.470 vs real 2.274（LOSO）。sim ped が ~0.7–0.8m 近づきすぎ＝~30% 過小再現。ADE が goal-following 支配で standoff にほぼ盲目なため、position-ADE 最適点は忠実な standoff 分布を与えない。
- **なぜ研究妥当性に効くか**: RQ2 の目的は sim 回避を real に合わせること。30% standoff 不足は「捉えた回避」主張を直接弱め、ADE 最小化が弱 v0 を選んだことを補強する。
- **根拠（裏取り済）**: paired 解析で gap mean 0.681、24/26 fold で real>sim、paired t p=2e-06、Cohen's dz=1.20。LOSO replicate（4/4, dz=1.95）。
- **推奨対応**: standoff gap を主忠実度成果として報告（KS でなく）。standoff/onset 分布に直接 fit するか ADE と joint で fit。ADE-only fitter の standoff 過小再現を key limitation として明記。
- **確信度**: high（検証 keep 2。1件は claim_affected の前提（RQ2 は standoff 再現を主張しない）を理由に Minor 格下げ＝RQ2 のスコープ次第）

### M5. avoidance-onset の加速度が2分岐で微分回数が異なる＝docstring 不変条件が false、回帰テストが偽の安心

- **file:line**: `src/core/metrics.py:370-384`
- **影響RQ・主張**: RQ2（avoidance-onset KS）
- **説明**: `ped_vel=None` 時 vel=grad(pos)→acc=grad(vel)＝位置の2階微分。`ped_vel` 供給時 acc=grad(ped_vel)＝供給系列の1階微分。docstring(L378-384)は両分岐が「SAME rule・identical step count」で「一致せねば KS が biased」と明記するが、供給速度が位置の有限差分と厳密一致しない限り**一致しない**（独立記録 vx で onset 3.5 vs 2.9 を再現）。回帰テストは `np.gradient(ped)` を velocity に与えるため分岐が構造的に一致し、不変条件を**検証していない**。
- **なぜ研究妥当性に効くか**: 現 RQ2 は両側 `ped_vel=None` なので live KS は偶然 consistent。だが docstring が保証する不変条件が偽＝将来 real に記録 vel、sim に None を渡すと onset KS が1階微分分 silent に biased。
- **根拠（裏取り済）**: 独立 vel で onset 分岐が不一致を再現。現 call site は両側 None（`calibration_harness.py:466-467`）で live は無害。
- **推奨対応**: 常に位置から有限差分するか、両分岐に同一 grad-count を適用。テストは有限差分でない velocity を与え documented behavior を assert。docstring を実契約に修正。
- **確信度**: high（live は無害なので3検証者とも Minor 寄りに格下げ。docstring の偽保証＋テストの空虚さは実装欠陥として Major〜Minor 境界）

### M6. 較正 clearance 基準（agent_radius）が RQ1b/AVEC 本番と不一致（0.35 vs 0.30）

- **file:line**: `calibration_harness.py:62`（DEFAULT_AGENT_RADIUS=0.35）/ `integrated_simulator.py:170-174` / `scenarios/*.yaml:agent_radius:0.3`
- **影響RQ・主張**: RQ2→RQ1b 接地（較正反応モデルの供給）
- **説明**: 斥力 clearance = distance-(ego_radius+agent_radius)。較正 harness は 0.35 固定で (σ,v0) を同定するが、AVEC/RQ1b シナリオは 0.3。override 注入は σ/v0 のみ差し込み 0.3 を温存＝較正時 clearance0=1.35m に対し適用時 1.30m で 0.05m 内側シフト＝同一真距離でより強い斥力で評価。
- **なぜ研究妥当性に効くか**: 「RQ2 で較正した反応モデルをそのまま RQ1b に接地」の厳密性に効く。σ（場の広がり）は特定 clearance 原点に対し最適化されており 0.05m ずれると exp 引数原点が物理的に変わる。
- **根拠（裏取り済）**: 適用/較正の力比 = exp(0.05/σ)＝calibrated で +4.4%、AVEC 既定で +7.4%。σ 不確かさ ±0.116 と同オーダー。
- **推奨対応**: DEFAULT_AGENT_RADIUS を 0.3 に揃えるか RQ1b/AVEC を 0.35 に統一。一致をテストで pin。limitation でなく実装で揃えるべき。
- **確信度**: high（mechanism 確定。効果 ~4% で検証は Minor 格下げ。σ 不確かさ同オーダーゆえ Major〜Minor 境界）

### M7. RQ1b 検証が GT 反応モデル自身の SFM を ground-truth とする＝閉ループ自己整合性であり外的妥当性でない

- **file:line**: `examples/run_rq1b_sensitivity.py:243-263` / `examples/run_da_poc.py:94-156`
- **影響RQ・主張**: RQ1b（robust 利得・CV 危険性）の外的妥当性/循環性
- **説明**: 各 arm で ego planner は SGAN/LSTM/CV で予測し、衝突相手の「ground-truth」歩行者は GT (σ,v0) を入れた同じ `PedestrianSimulator` が生成する。予測誤差も衝突結果も SFM rollout に対し測られる＝「CV 危険性は較正で生き残るか」は「同じ SFM family を較正点に再パラメータ化したら生き残るか」を問う内部感度。歩行者は real ではなく SFM（較正済）で、RQ2 自身がその SFM は standoff を ~0.7m 過小再現すると示す。
- **なぜ研究妥当性に効くか**: RQ1b の説得力は「AVEC 閉ループ危険性は一部 artifact」。だが SFM パラメータ A を SFM パラメータ B に置換しても real 歩行者の挙動は確立できず、計画結論の SFM パラメータ感度しか示せない。
- **根拠（裏取り済）**: override→simulator 経路を code 確認。REPORT は preamble(L3)に「感度分析、外的検証ではない」と1回記すが結論文(L62)に SFM-vs-real caveat なし。
- **推奨対応**: 「感度であり外的検証でない」caveat を全 CV 危険性結論文に隣接させ、GT 歩行者が SFM（real でなく）かつ較正 SFM も standoff 過小再現＝artifact 主張は「SFM family 内」に限定と明記。
- **確信度**: high（mechanism 確定。検証で keep 1/Minor 格下げ 2＝既に preamble に caveat があり honestly labeled とも読める）

### M8. RQ1b CV 危険性 verdict が不均等シード（20 vs 10）の衝突カウント差を統計検定なしで比較＝flip が seed ノイズの可能性

- **file:line**: `examples/run_rq1b_sensitivity.py:154-171,237-240,345-354`
- **影響RQ・主張**: RQ1b（CV 危険性は反応モデル依存・sensitivity verdict）
- **説明**: avec/calib は20 seed、±1SD corner は10 seed。`rand_verdict` は生衝突 COUNT を比較し `_sensitivity_status` が boolean 差で「flip」を宣言。`calib_hi` で cv_danger_holds=False、他は True（verdicts.csv 実機確認）だが基底カウントは極小（cv_single 0–2、sgan_robust 0–1）。「1衝突 vs 0」over 10 seed（10% vs 0%）は Monte-Carlo ノイズと区別不能で、corner の小 N が機械的にノイジー＝flip と seed budget が交絡。
- **なぜ研究妥当性に効くか**: RQ1b ヘッドラインは「CV 危険性は反応モデル依存（AVEC artifact）」で、証拠が verdict flip。flip が seed ノイズなら結論が過大。
- **根拠（裏取り済）**: master_runs.csv で avec/calib=20 seed、corner=10 seed 確認。Fisher one-sided で各 cv_danger arm p≥0.246（多くは 0.50）＝集計 cv_danger と calib_hi flip は noise-grade。**対照的に per-scenario S2 信号**（avec S2 single 9/60 vs robust 0/40、Fisher p=0.0078）は有意。
- **推奨対応**: seed budget を均等化（または衝突 RATE を Fisher/binomial CI で比較）し、各 verdict を有意性/効果量で gate。1-vs-0 flip は undetermined 扱い。集計でなく per-scenario S2 を discriminator として前面に。
- **確信度**: high（検証3件すべて keep。集計 verdict はノイズだが per-scenario S2 は有効＝結論の出し方の問題）

### M9. RQ1b per-scenario REPORT narrative が avec/calib のみ手書きで、同梱の calib_lo/calib_hi 行と矛盾

- **file:line**: `examples/run_rq1b_sensitivity.py:384-405`
- **影響RQ・主張**: RQ1b（per-scenario CV 危険性の読み筋）
- **説明**: narrative は「両 GT で」「較正 GT では」と**2 GT 物語**で書かれるが標準 run は**4 GT**を出す（実機 scenario_rand.csv 確認）。具体的矛盾: (a)「S1 主に cv」は calib_hi/calib_lo で cv_single=0（lstm/sgan が衝突）＝false。(b)「S2 較正 GT→no-conflict, 主張②消失」は calib_lo S2 が single-danger（sgan=1）を無視。(c)「S3 較正 GT→mixed」は calib_lo S3 が single-danger（robust_total=0）を無視。guard(L385)は scenario 存在のみ check し GT 集合の一致を見ない。
- **なぜ研究妥当性に効くか**: REPORT は審査官が読む artifact。自データ表と矛盾する hardcode narrative は信頼を損ね、誤った per-scenario 読みを修論へ伝播させ得る。
- **根拠（裏取り済）**: 本レビューで `scenario_rand.csv`（4 GT×3 scenario=12行）と REPORT.md narrative（「両 GT」「主に cv」）の矛盾を直接確認。
- **推奨対応**: per-scenario 読みを `rand_scenario_rows` から全 GT 分プログラム生成するか、手書き prose を厳密に2-GT core set に gate。「主に cv」等の GT 依存記述を除去。
- **確信度**: high（検証3件すべて keep、artifact で直接確認）

### M10. RQ1a 開ループ ADE/NLL が stride=1 重複窓で集計され有意性検定なし＝序列の不確かさが未定量

- **file:line**: `examples/run_openloop_prediction.py:118-143`
- **影響RQ・主張**: RQ1a（cv/lstm/sgan 序列の妥当性）
- **説明**: `extract_fixed_windows` 既定 stride=1 で隣接窓は seq_len-1=19/20 フレーム重複。同一 ped 軌跡が最大 ~20 窓に再出現し count 重みで pooled。CI/bootstrap/検定なしで点推定のみ。
- **なぜ研究妥当性に効くか**: H1 は数 cm の ADE 差に依存。95% 内容共有窓で有効標本は名目軌跡数を大きく下回り、数 cm 差を clustered/blocked 検定なしで「序列」と主張できない。
- **根拠（裏取り済）**: zara1 stride=1 で 705窓/2356 origin、stride=12（独立）で 58窓/198 origin＝~11.9x 膨張。grep で開ループ経路に検定皆無を確認。**点推定のみなら即時の誤りではない**ため検証で Minor 格下げ意見あり。
- **推奨対応**: CI/有意差を報告するなら stride≥pred_len で非重複窓か ped/窓クラスタ block bootstrap。点推定のみなら「重複窓のため標準誤差は報告しない」と明記し序列反転を noise 内と主張しない。
- **確信度**: high（mechanism 確定。報告に検定を付さない限り Major〜Minor 境界）

---

## Minor findings

### m1. ddof 不整合: RQ2 安定性 std は ddof=0、他スクリプトは ddof=1（LOSO で ~13–15% 過小、RQ1b corner が継承）
`run_rq2_evaluation.py:217-223`。`vals.std()`（ddof=0）が唯一の ddof=0 site で、他5箇所は ddof=1。LOCO で ~2%（無視可）だが LOSO（n=4）で v0 std を 13–15% 過小、corner（calib_lo/hi=mean±0.116/±0.139）は ddof=0 spread を継承。**M3 が示す通り LOCO fold は非独立でいずれの ddof も clean な frequentist 解釈を持たない**ため、ddof より「descriptive spread と明記」が本質。確信度 high（数値裏取り済、影響軽微）。

### m2. KDE-NLL の per-point log-likelihood floor (-20) が out-of-support GT で飽和＝censored 統計
`metrics.py:19,150`。floor が eth/univ で最大 ~11–15% の点に binding。lstm-vs-sgan NLL 序列は floor 値 -15/-20/-25 で**安定**（lstm が同 3/5 シーンで勝利）と裏取り済＝序列は飽和 artifact でない。num_samples=20 確認済（N=2 worst case は実 run で発生せず）。floor-hit 率の開示推奨。確信度 high（mechanism 実在、序列影響なし）。

### m3. avoidance-onset KS / closest-approach KS の標本独立性違反（同一 ego/同一 ped の多重計上）
`calibration_harness.py:466-467,474-477`（onset: 1 encounter 内複数 ped が同一 ego 軌道に対し測定）/ `vci_encounter.py:290-354`（DUT multivehicle: 1 ped を K 車両で再計上）。pooled KS p 値が anti-conservative。CITR で onset 実膨張は ~1.8x（16 onset/9 encounter）と小。closest は 1-encounter-1-scalar で clean。single-vehicle を headline、multivehicle KS は descriptive 扱いを推奨。確信度 medium–high（mechanism 確定、影響限定）。

### m4. DUT multivehicle CLI 既定 (σ=1.20,v0=1.62) が canonical LOCO 平均 (1.156,1.681) と不一致
`run_rq2_dut_validation.py:65-67`。既定で走らせると thesis 較正点と異なる点を validate。**ただし同梱 CSV は canonical 1.156/1.681 で生成済**（実機確認、worst case は実現せず）。`--citr-ref-ade` 未指定で degradation 文脈が出ない。既定を LOCO 平均にするか必須化を推奨。確信度 high（影響軽微）。

### m5. CITR fps の justification anchor が weak（forced 1.3 m/s ヒューリスティック）だが最終 fps=29.97 は data-validated
`examples/inspect_vci_data.py:315-348`。inspector の walking self-consistency は median を 1.3 に強制するが、記録 vx_est/vy_est（fps-independent）median は ~1.26 で pos-diff@29.97（~1.27）と一致＝**fps=29.97 自体は正しい**。RQ2 は inspector を使わず `--fps 29.97` を hardcode。justification の anchor のみ弱い。確信度 high（最終 fps は妥当、検証で NotAnIssue 寄り格下げ）。

### m6. RQ2 ego 速度域（CITR ~2.4 m/s median）が AVEC ego（5–6 m/s）と乖離＝位置のみ斥力の速度外挿
`vci_loader.py:212-222`。位置のみの斥力は速度項を持たず ~2x 外挿。**ただし RQ2 limitation #2 として既に文書化**（summary に明記、本レビューで確認）。finding の p95=2.64 主張は誤りで実測 p95=3.99＝RQ1b cruise=3.0 は in-domain。既文書化につき検証で Minor 格下げ。確信度 high。

### m7. 中心間距離 vs 表面 clearance の convention 混在
`vci_encounter.py:241-245`（min_sep 中心間, 閾値8m）vs `integrated_simulator.py:170-174`（SFM clearance は半径1.35m 減算）。KS は両側中心間で unbiased。standoff を物理 gap として報告する箇所があれば 1.35m 過小。convention 明記推奨。確信度 medium（mechanism 確定、結論非影響）。

### m8. DUT velocity 列が px/m 比に対し未検証＝将来の単位ミスを catch しない構造的リスク
`vci_loader.py:151-172` / `vci_encounter.py:106-120`。記録 vx/vy を m/s と無検証で信頼。現状 DUT 1.27 vs pos-diff 1.28（一致）で無害だが consistency guard なし。±20% 逸脱を flag する per-clip guard 追加を推奨。確信度 medium（latent risk）。

### m9. 線形リサンプル＋`_far_goals` の goal 汚染（C3-b の data-loading 視点）
`vci_loader.py:137-146` / `calibration_harness.py:149-179`。0.4s grid 補間が sub-0.4s 曲率を平滑化し、goal が回避済み path から導出。C3 と同根の Major 機構の data-loading 側面。確信度 medium。

### m10. robust 利得 verdict の Time が collision-free run のみ平均＝右側打切り
`run_rq1b_sensitivity.py:131-140` / `run_da_poc.py:264-272`。衝突 run 早期終了の除外は不可避だが非対称。**実 run では robust 衝突0かつ MinDist gate が全 inflation で先に fail**＝Time/censoring は決定的でない（censored/uncensored で verdict 不変を裏取り）。verdict に per-cell 衝突カウント併記＋inflation 衝突≤robust の前提明記を推奨。確信度 medium（現状無害だが guard なし）。

### m11. NLL は CV で未定義（<2 サンプルで skip）＝NLL の「3手法序列」は実質 LSTM vs SGAN
`metrics.py:121,135-138`。CV は num_samples=1 で NLL=NaN。NLL 序列は2手法比較として提示し CV を含めない旨を明記。回帰テスト pin 済。確信度 high（既に正しく scoped、影響軽微）。

### m12. state machine envelope/stop が1step stale クリアランスを使う tuned 実装に依存
`integrated_simulator.py:539-567` / `state_machine.py:98-114,252-266`。zero-lag だと S1 SGAN timeout / S3 cv リミットサイクルが起き lag は tuned behaviour。反応モデル変更で binding タイミングがずれ衝突有無に寄与し得る。**S2 は envelope_decel=0.0 で無効**（実機 load_config 確認）＝S2 discriminator は lag 非依存。limitation 明記を推奨。確信度 low。

### m13. `_hits_dynamic` の時刻 index clip がホライゾン超部で歩行者凍結（条件付き）
`frenet_planner.py:1226-1233`。path が予測より長いと末尾 index に張り付き。**実 config（dt=0.1, max_t=5.0）では t=0 prepend により CV フォールバック n_time=51 で clip は no-op**＝凍結0点を裏取り。max_t/dt が非整数の場合のみ1点凍結。CV フォールバック長 == max_t/dt+1 をテストで pin 推奨。確信度 low（検証で1件 refuted/NotAnIssue）。

---

## RQ別 妥当性総括

### RQ1a（予測序列が sim 人工物か＝H1）
**開ループ経路の独立性は実装上厳密に支えられている**（ReplayPedestrianSource が ego 無視・回帰 pin、固定母集団窓がフレーム穴排除、観測は GT 純ダウンサンプル）。**しかし序列を測る metric 自体が手法非対称で、これが RQ1a の最大の弱点**。(1) scene-level joint best-of-N が stochastic 手法だけを膨らませ（M1）、univ で per-agent 化すると序列が反転、(2) best-of-20 vs CV-single の非対称（M2）、(3) NLL は CV 未定義で2手法比較（m11）、(4) stride=1 重複窓で有効標本膨張・検定なし（M10）。**満たすべき条件**: per-agent minADE への変更 or scene-level の明示的正当化＋序列の選択不変性の再実行確認。N を全手法で揃えた感度表。序列差に CI を付すなら clustered bootstrap。**残る限界**: 同一 metric が AVEC sim baseline にも適用されるため H1（序列入替り＝人工物）の定性結論は metric 選択に対し**頑健と推定されるが、実データ ADE の絶対水準・序列を published 数値と比較する記述は非正準 metric ゆえ不可**。

### RQ1b（robust 利得・CV 危険性）
**主張①（robust 利得）は全4 GT で頑健に成立し（verdicts.csv 実機確認）、MinDist gate が Time/censoring に先行して効くため打切りバイアス非依存（M10 裏取り）＝最も強い結果**。**主張②（CV 危険性は反応モデル依存）は実装上支えられるが提示に難**: 集計 verdict は単一桁カウント・不均等シードで Fisher 有意性なし（M8）＝集計 flip はノイズ grade。**ただし per-scenario S2 はクリーンな discriminator**（avec で single 衝突→calib で消失、Fisher p=0.0078）。**満たすべき条件**: per-scenario S2 を主証拠に据え、集計 cv_danger と calib_hi flip を undetermined 扱い、seed budget 均等化 or Fisher CI。REPORT narrative を4 GT に整合（M9）。**残る限界（本質的）**: GT 歩行者が SFM（real でなく較正 SFM、standoff ~0.7m 過小再現）＝結論は「SFM family 内の感度」に限定され外的妥当性ではない（M7）。clearance 原点 0.35/0.30 不一致（M6）も接地厳密性を損なう。

### RQ2（SFM 自車斥力の実データ較正）
**最も主張が過大な RQ**。(σ,v0) の identifiability が弱く、(1) held-out KS が n=1 縮退で忠実度を弁別しない（C1）、(2) ADE が v0=1.68 と 3.5 を区別できない（C2）、(3) standoff を ~30% 過小再現（M4）、(4) 低 v0 が goal 汚染/ped-ped 交絡で下方バイアス（C3）、(5) LOCO ±std が訓練共有で過小精度（M3）。**満たすべき条件**: ①KS は pooled CITR（n≥26）で1本に集約、②「AVEC v0 too high」を ADE でなく有効 pooled standoff KS に依拠 or「斥力は必要だが強さは弱 identifiable」へ緩和、③安定性は LOSO で headline・per-scenario v0 spread（1.2–8.1）を明示、④goal/ped-ped/cruise 感度を identifiability エンベロープとして併記、⑤速度外挿（M6）と clearance convention（m7）を limitation 化。**残る正直な結論**: 「実データは斥力の存在を支持するが、その強度 (σ,v0) は CITR 群衆・低速・goal 汚染下で弱 identifiable であり、AVEC v0=3.5 が高すぎるとは ADE では言えない」。

---

## 検討したが棄却・影響軽微とした点

- **one-step 診断の v0→0 が metric artifact**（当初 Major）: synthetic recovery（既知 v0=3.0 データ）で one-step が**内部最小 v0~2–3 を回復**し monotone でない＝finding の load-bearing 主張が refuted。Minor へ格下げ。
- **cruise de-bias で v0 が 3.1–5.3 に振れ過調整反転**（当初 Critical）: `loss_ade` が v0 上昇で**単調悪化**（baseline 0.657 最良）＝データは低 v0 を好み、de-bias 推定はむしろ ADE を悪化。「反転」主張は repo 自身の損失で反証され Minor へ。
- **interaction_distance=None で遠方 ped 希釈**（Minor）: 実 CITR で id=5m に絞ると v0 が UP（1.62→1.69）＝dilution は v0 を上げ下げ両方向で claim-impact 軽微。NotAnIssue 寄り。
- **KDE bandwidth が N=2 で noisy**（Minor）: 実 RQ1a run は全行 num_samples=20＝N=2 worst case 不発。NotAnIssue。
- **ETH/UCY 0.4s 仮定がフレーム穴無視**（Minor）: 固定母集団交差が hole-straddling 窓を構造的に全除外（実測 0/1950 ped が hole を跨ぐ）＝実害ゼロ。NotAnIssue。
- **開ループ vs 閉ループが cadence/sample policy で別評価**（dismissed）: 閉ループ ADE も best-of-20・0.4s grid で評価（`metrics.py:82` 同一、stride=4 で subsample）＝finding の前提が code で矛盾。NotAnIssue。
- **`_hits_dynamic` 凍結**（low→NotAnIssue）: t=0 prepend で CV フォールバック n_time=51＝実 config で clip は no-op。

これらは「mechanism は実在するが claim-impact が軽微」または「finding の load-bearing 経験的主張が repo データで反証」されたもので、研究結論を崩さない。

---

## カバレッジ

**見たもの（6次元・検証済 findings に裏取り再実行あり）**: 較正 harness/optimizer（`calibration_harness.py`, `optimize.py`, `run_rq2_*.py`）、忠実度メトリクス（`metrics.py`, `test_metrics.py`, `test_fidelity_metrics.py`）、データ読込/単位/fps（`vci_loader.py`, `vci_encounter.py`, `eth_ucy_loader.py`, `inspect_vci_data.py`）、実験設計（`run_openloop_prediction.py`, `run_rq1b_sensitivity.py`, `run_da_poc.py`, `make_margin_report.py`）、Replay 経路（`replay_source.py`, `observer.py`, `trajectory_predictor.py`）、AVEC コア（`integrated_simulator.py`, `frenet_planner.py`, `state_machine.py`, `coordinate_converter.py`, `footprint.py`）。本レビューで `summary_loco.txt`・`cruise_sensitivity.csv`・`verdicts.csv`・`scenario_rand.csv`・`REPORT.md` の主要数値を実機再取得し確認。

**見ていない/限界**: (1) SGAN ベンダ実装（`sgan_vendor/`）の予測正しさ自体は未監査（学習済モデル前提）。(2) 可視化層（`src/visualization/`）はベンチで無効化されるため未確認。(3) DUT/ETH/UCY の**生データ取得経路の正しさ**（CSV→m 変換の上流）は loader の m/s 整合確認に留まり、元動画キャリブレーションは未検証。(4) clustered bootstrap 等、推奨した統計手法の**再実装結果**は本レビュー範囲外（findings は欠如の指摘に留まる）。(5) 1980-run RQ1b キャンペーンの全 master_runs は spot-check のみで全 fold の悉皆再現はしていない。

---

> **完全性クリティックについての注記**: 下記クリティックは合成レポートの先頭12,000字（M8 まで）のみを参照して生成されたため、点7（M8 のシード不均衡）など一部は本文の M8/M9 で既出。**追加価値が高いのは点1–6・8–9（特に ETH/UCY の座標系・スケール・2ファイル連結・欠損フレーム・grid 整合の未検証）** で、これらは本文 findings に含まれない残存ギャップ。

# 完全性クリティック: 未カバー/検証不足の指摘（最大10点）

1. **ETH/UCY の座標系・スケール検証が完全に欠落（RQ1a 全体の前提）** — `eth_ucy_loader.py:1-18` は「world-frame metres」と宣言するだけで、SGAN 配布ファイルの座標が実際に m か（特に `eth` シーンは homography 起因で回転・スケールが他シーンと異なることが知られる）を一切検証していない。レポートは ADE の best-of-N 非対称（M1/M2）に集中するが、ADE の絶対値（cv 0.534 等）の単位妥当性そのものが未検証で、シーン間 ADE 比較（H1 の核心）が座標スケール差の混入を排除できていない。これは ADE プロトコル以前の妥当性問題で研究結論に直結する。

2. **`univ` の2ファイル連結とウィンドウ境界の取り扱いが未検証（RQ1a）** — `SCENE_TEST_FILES["univ"]` は students001/003 の2録画を持つ（loader:31-37）。2録画を跨いだ fixed-population window が生成されると、別録画のフレームが連続扱いされ偽の軌道が混入する。レポートは univ で per-agent 反転（M1）を最重要根拠に挙げるが、その univ サンプルの window 境界健全性（録画跨ぎ排除）を確認した形跡がなく、反転証拠自体が連結アーティファクトの可能性を排除できていない。

3. **`frame_step`（mode）と「missing frame を無視」の相互作用が未検証（RQ1a）** — loader:50-60 は最頻ギャップを 0.4s グリッドとし、欠損フレーム実時間を無視する旨を明記する。fixed-population window が欠損を跨ぐと、物理的に 0.8s 離れた2点を 0.4s として GT に使う＝ADE/FDE の time horizon が暗黙に伸縮する。tests でこの「欠損跨ぎ window で実時間が伸びるケース」が検証されているか不明で、stochastic 手法の外挿が不利化する系統誤差になり得る。

4. **RQ1a の観測履歴→GT 時刻アンカー整合がレポートで未追跡** — カバレッジに `test_prediction_anchor.py`/`test_observer.py` が挙がるが、生成レポートは observer のダウンサンプリング位相（0.1s sim → 0.4s SGAN グリッド）と `_standard_ade_fde_details` の `pred_indices`/`future_offsets`（metrics:38-40）の位相が厳密一致するかを論じていない。1ステップずれれば全 ADE が系統バイアスを受けるため、best-of-N 議論より先に検証すべき土台。

5. **closest-approach KS の n=1 縮退（C1）の波及先＝onset KS の独立性が未点検** — レポートは closest KS の n=1 を正しく指摘するが、`fidelity_report:474-477` の onset KS は per-ped プールで n>1 になる一方、同一 encounter 内の複数 ped は強く相関（同じ ego 軌道に反応）。`compare_distributions_ks` の docstring（metrics:406-408）自身が autocorrelated pooling を警告しているのに、onset 側がこの i.i.d. 仮定違反に該当しないかの検証が欠けている。closest を廃止しても onset KS が同種の anti-conservative p を出す可能性。

6. **RQ1b の dist-aware（robust）プランナ自身の衝突判定健全性が GT-artifact 分類の前提として未検証** — `rand_scenario_rows:201-208` は robust_tot >= single_tot を "GT-artifact" と分類するが、これは「robust プランナの chance-constrained 衝突回避が正しく全 SGAN サンプルを評価している」前提に立つ。M7（SFM 自己整合性）は GT 側を問題にするが、robust プランナ側の `chance_epsilon=0.0`（worst-case）実装が S3 で本当に全サンプル回避を試みたか（＝GT-artifact が真に幾何的不可能か、プランナ側の劣化か）の切り分けが未着手。主張②の S3 結論の頑健性に効く。

7. **M8 の seed 不均衡（20 vs 10）が衝突カウント比較を非正規化のまま行う設計欠陥** — `_seeds_for:237-240` で avec/calib のみ20、±1SD corner は10。`rand_verdict:159-167` は `collision_count.sum()` を seed 数で割らずに直接比較する。corner GT（10 seed）と headline GT（20 seed）を跨ぐ集計 verdict があれば母数差で衝突カウントが機械的に半減し、verdict が seed 予算の artifact になる。レポートは M8 で触れる方向だが（途中切断）、verdict 関数が母数正規化を一切持たない点を実装レベルで pin すべき。

8. **RQ スイート全体の多重比較・検定力の管理が不在（横断的統計妥当性）** — RQ1a（5シーン×3手法）、RQ2（LOCO 26 fold + LOSO + cruise sweep + DUT）、RQ1b（4 GT×2主張）で多数の対比較・KS・verdict flip を報告するが、family-wise error や検定力（n=26、衝突カウント単桁）の管理が見当たらない。単一桁の衝突差（M8）や p=0.94 の non-significance（C2）を「区別不能」と解釈する一方、有意所見側の多重性補正がなく、結論の非対称な厳密性が研究全体の信頼性を損なう。

9. **`compare_distributions_ks` の有限値ドロップが sim/real で非対称になり得る点が未検証** — metrics:416-419 は sim/real 双方から個別に非有限を除去する。sim 側で ped が ego に重なり onset が空配列（avoidance_onset:368-369 が `[]` 返し）になる頻度が GT パラメータ依存で変わると、KS の有効 n が arm 間で変動し、KS 比較が「分布差」でなく「有効サンプル数差」を測る。fidelity_report の `n_onset_sim`/`n_onset_real` が arm 間で大きくずれていないかのチェックがレポートに無い。

10. **生成レポート自身の内部矛盾（narrative「両 GT」vs テーブル「4 GT」）が未解決のまま提出されている** — レポート preamble がこの矛盾を「confirming the per-scenario narrative contradiction」と自認しながら、本文（C/M 群）で主張①「全 GT で True」と整合する形に統一していない。これは完全性の観点で、RQ1b の主張①の GT カバレッジ記述（2 か 4 か）が確定しておらず、`build_verdicts:215-231` が実際に何 GT をループしたか（`gt_labels` 引数の実値）をコミット済み `verdicts` CSV と突合して確定させる検証が残っている。

補足: レポートが健全と判定した土台（ReplayPedestrianSource の ego 無視、fps=29.97 整合、座標変換）は妥当だが、上記1-4の通り**ETH/UCY 側の単位・グリッド・連結の妥当性検証が VCI 側に比べ手薄**で、RQ1a の H1 結論はこのデータ前処理層の検証強化なしには確定できない。
