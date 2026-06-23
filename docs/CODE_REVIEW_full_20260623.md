# 全体コードレビュー（軸A 研究コードベース監査）

**対象**: Social-GAN 歩行者軌道予測 × Frenet 経路計画 統合シミュレーション（AVEC'26 論文／修士論文向け）
**日付**: 2026-06-23
**性質**: READ-ONLY 監査（ファイル変更なし）。本レビューは検証者合議（adversarial verification）を通過した確定指摘のみを掲載する。
**結論の一言**: コードベースは概ね健全。安全に関わる中核（同時刻のみの衝突判定、予測フォールバックの全ホライゾン生成、設定の単一情報源、決定論的再現）は実装上正しく機能している。**研究結論を覆す「現に発生しているバグ」はゼロ**。最大の研究リスクは（a）バージョン管理外で手計算される RQ1a の横断集計（集計方法しだいで H1 の序列が反転しうる）と（b）研究主張を支える経路に対するテスト欠如（再現性・安全分岐・統計family の番人がない）に集中する。

---

## エグゼクティブサマリ

- 確定指摘は **Critical 0 / Major 4 / Minor 18 / Nit 4**（計26）。Major のうち3件は**テスト欠如**、1件は**横断集計の単位混在**で、いずれも「現在の数値が間違っている」ではなく「将来の主張・再現性を守る番人がない／単位非整合な集計を見出しに使うと危うい」という性質。
- **現に出ている数値・コミット済み成果物（master_runs / verdicts / folds / summaries / openloop CSV）は妥当かつバイト安定**。確定指摘の大半は「現データでは発火しない潜在バグ」または「テスト・文言の欠落」である。指摘自身がこの限定を honestly 明記している。
- 唯一、執筆前に必ず手当てすべき研究主張リスクは **RQ1a の横断 ADE 集計**：5シーン非加重平均は eth シーン（加速動画 ~0.8s ケイデンス）の単位非整合で歪み、eth を除く／trajectory 加重／per-agent に変えると CV が「最悪」⇄「最良」と序列が反転する。per-scene の序列（=H1 の本体）は不変だが、横断平均を見出しにするのは危険。
- 数値プリミティブで唯一の実バグは `calc_curvature_rate` の指数誤り（chord-length パラメータ化で d^2/d^3 とすべきが本来 d^1.5/d^2.5）。曲線シーン S3 の絶対 comfort/jerk を最大 ~8% 歪めるが、全予測手法が同一プランナを共有するため**手法間序列（中心主張）は保存**＝二次的。

---

## 研究主張リスク（最優先・執筆前に判断が必要）

ここに挙げる項目は、放置すると修論／論文の結論や数値表に直結しうる。**現時点でコミット済みの結論を覆すものは無い**が、執筆時の集計・記述・追加テストで明示的に手当てすべきもの。

### R1. RQ1a 横断 ADE 集計が「単位非整合」かつ「バージョン管理外」で、集計方法しだいで H1 の序列が反転する 〔Major〕
- 証跡: `examples/run_openloop_prediction.py:111` (`evaluate_scene` は per-scene 集計のみを CSV 出力。横断平均を出すスクリプトは examples/ にもテストにも存在しない)。
- メカニズム: eth シーンは加速動画由来で ~0.8s/step（`src/datasets/eth_ucy_loader.py:60-75`、`docs/GAPB_eth_ucy_cadence_resolution.md`）。ADE は dt 非依存な位置差（`src/core/metrics.py:88-92`、stride=1）として**シーン内では正しい**が、1 プロトコルステップの空間スパンがシーン間で異なるため**絶対 ADE はシーン横断で比較不能**。eth の CV ADE=1.075 は他シーン ~0.32–0.52 に対し突出（CV の直線外挿が 2x 長い実時間ステップで誤差を蓄積）。
- 影響: MEMORY 記載の見出し「5シーン平均 cv0.534/lstm0.457/sgan0.510＝序列はシーン依存／CV は最良でない」は、この非整合な非加重平均に依存。検証で再現済み：**eth を除くと cv0.399 < lstm0.414（CV が最良に反転）**／**canonical な per-agent minADE では cv0.534 / lstm0.333 / sgan0.363（学習モデルが明確に優位）**。約1%のデータ量のシーン（eth）が等シーン重みで「CV competitive / 序列入替」主張を支えている。**この見出し集計を生成する成果物がリポジトリに存在しない**＝再現不能・テスト不能。
- 修正: 横断「非加重平均」を見出しに使わない。(1) per-scene のみ報告（H1 証拠は元来 per-scene）、または (2) 集計スクリプト `examples/aggregate_rq1a.py` をコミットし、trajectory 加重／非加重・eth 含む／除く・scene-level／per-agent を並置、回帰テストで数値を pin。修論には「どの集計が H1 を支えるか」を明記し、集計選択への不変性を示す（不変でなければ主張を弱める）。eth は等重み平均ではなく sensitivity 行として扱う。
- 注: per-scene の序列（H1 の本体）と eth ケイデンス caveat は `docs/GAPB` で既に開示済み。本リスクは「執筆時の横断見出しの作り方」に限定される。

### R2. 確率的（SGAN/LSTM）クローズドループ実行のバイト安定性を保証する決定論テストが無い 〔Major / テスト欠如〕
- 証跡: `examples/run_statistical_benchmark.py:21`（`set_seed()` が run 毎に random+numpy+torch をリセット）、`src/simulation/integrated_simulator.py:100`（INIT で `np.random.normal` による v0 抽選）、`src/prediction/sgan_vendor/models.py:29`（step 毎 `torch.randn`）。同一 (scenario, method=sgan, seed) を二度走らせて出力一致を検証するテストは無い（決定論テストは v0／ped-sim 層の `test_calibration_harness.py:222`・`test_pedestrian_simulator.py:209` のみ）。`torch.use_deterministic_algorithms`／cudnn フラグはどこにも設定されていない。
- 影響: MEMORY/CLAUDE が繰り返し主張する「master_runs/verdicts/folds はバイト安定」は脆い RNG 順序契約に依存。RNG 抽選を並べ替える将来のリファクタ（例: v0 ノイズをモデルロード後に移動、warmup に np.random 呼び出しを追加）が**テスト失敗なしに**再現性主張を破壊しうる。
- 修正: method=sgan・同一 seed で IntegratedSimulator を2回走らせ軌道／指標のビット一致を assert するテスト（distribution_aware_planning 版も）。set_seed を挟んだ再実行の一致テスト。可能なら trajectory.npz の golden hash を pin。「バイト安定」を手動 spot-check から強制 invariant に格上げする。

### R3. 予測失敗 CV フォールバックと persistent-failure 安全ガード（C-2）が完全に未テスト 〔Major / テスト欠如〕
- 証跡: `src/simulation/integrated_simulator.py:468-496`（フォールバック）、`:469-476`（5連続失敗で RuntimeError）。tests/ に `_update_prediction`／`_consecutive_prediction_failures`／"times in a row" を検索しても 0 ヒット。`test_trajectory_predictor.py:36` は predictor 単体の raise を見るだけで、シミュレータの except 分岐・捏造時系列の全ホライゾン長・5-strike RuntimeError には到達しない。
- 影響: CLAUDE.md の critical invariant「予測失敗→等速直線、フォールバックは全ホライゾン時系列を生成」と「5連続失敗で raise（沈黙の CV 退化を禁止）」が無防備。将来の変更が (a) フォールバックを単一ステップに平坦化、(b) 誤ホライゾンに外挿、(c) 連続失敗ガードを除去して**壊れた sgan 実行が毎ステップ沈黙裡に CV へ退化**しても全テスト green。(c) は cv/lstm/sgan 手法比較を直接無効化する（壊れた sgan run が CV として採点される）。実装自体は検査上正しいが、研究主張を守る回帰ガードが無い。
- 修正: `predict_single_best` を raise させ、`dynamic_obstacles.shape[1] == int(max_t/dt)`（現在位置 prepend 込み）と `positions + velocities*t` 一致を assert。5連続失敗で RuntimeError／4回では発生しない（成功でカウンタ reset）を assert。

### R4. RQ2 見出し pooled-KS と saturation/de-saturation family ロジック（多重比較の family サイズ）が未テスト 〔Major / テスト欠如〕
- 証跡: `examples/run_rq2_evaluation.py:281-296`（`_pooled_ks_stat`）、`:310-381`（`headline_tests`、`:346` の 1e-12 saturation ガード、`:348-358` の de-saturation 分岐）、`:_standoff_gap`。`tests/test_rq2_evaluation.py` は COLUMNS/RAW_KEYS/evaluate_fold/make_folds のみ import。pooled-KS・headline_tests・standoff_gap を呼ぶテストは皆無。
- 影響: RQ2 fidelity 見出し（pooled closest-approach KS=0.462, p=0.0071, standoff +0.68m）と、**どの arm を family の独立仮説に数えるか**（saturation ガードが family_size m を決定）が `src/core/multiplicity.py` の BH-FDR/Holm へ直結。m が変われば q 値が動き、見出し fidelity gap が補正後に生存するか否か（現在 family q=0.007 の境界）が変わる。検証で family_size が 1（saturated）⇄ 3（de-saturated）に切り替わることを実証済み。現ロジックは正しく現出力もバイト安定だが、claim 直結経路に番人ゼロ。
- 修正: 合成 pools を `_pooled_ks_stat` に与え既知 ks/p・空側→None を assert。`headline_tests` に saturate ケース（同一配列→family 1 + controls 充填）と de-saturate ケース（差異配列→family 2）を与え family 帰属と見出しフラグを assert。

### R5（数値プリミティブの実バグ）. `calc_curvature_rate` の指数誤りが曲線シーン S3 の絶対 comfort/jerk を最大 ~8% 歪める 〔Minor〕
- 証跡: `src/planning/cubic_spline.py:269` は `(b*d - 3*a*c)/(d*d*d)`（= d^2/d^3 分母）。chord-length パラメータ化（`_calc_s`、`:206-213`）では |r'(s)|≠1 なので、曲率 κ=a/d^1.5 の正しい導関数は `b/d^1.5 - 3*a*c/d^2.5`。コードは unit-speed（d==1）でしか正しくない。
- 影響: 3名の検証者全員が math 誤りを再現（S3 右折スプラインで |r'-1| 最大 0.084、code 式 vs 中心差分の最大絶対誤差 0.0022／相対 ~8%）。`rdkappa`/`i_dkappa` が `kappa_r_d_prime = rdkappa*d + rkappa*d_dot`（`src/core/coordinate_converter.py:74,142`）へ流れ、d_ddot 初期条件と変換後 kinematics を摂動。**全予測手法が同一プランナを共有するため手法間序列（中心主張）は保存**。S1/S2 は直線で影響ゼロ、blast radius は S3 のみの絶対 comfort/jerk に限定。決定論は不変（同じ誤値を毎回生成）。検証者の severity は major×2／minor×2 に割れたが、**impact が二次的（序列・安全・決定論いずれも非影響）であることは3者一致**のため minor に整える。
- 修正: `:269` を `b / d**1.5 - 3.0 * a * c / d**2.5`（= `(b*d - 3*a*c) / d**2.5`）に。曲線スプラインで `calc_curvature_rate` が `calc_curvature` の有限差分に ~1e-6 一致する回帰テストを追加（既存テストは直線のみで未防護）。

---

## テーマ別 確定指摘

### A. 衝突判定・予測パイプライン（safety invariant 群）
中核 invariant（同時刻のみ評価・全ホライゾン時系列）は実装上正しい。残るのは潜在的な不整合のみ。

| ID | 指摘 | Sev | file:line |
|----|------|-----|-----------|
| A1 | 予測失敗 CV フォールバックが plan_horizon のみのグリッドを使い、成功時の `max(plan_horizon, pred_len*sgan_dt)` グリッドと不整合。現シナリオでは max_t=5.0≥4.8 のため長さ同一で観測不能。将来 max_t<pred_len*sgan_dt の config で長さ差が出るが、プランナは max_t までしか参照しないため衝突カバレッジは正しいまま。 | nit | `src/simulation/integrated_simulator.py:482-493` |
| A2 | `process_prediction` の per-axis ガード `np.allclose(coords, coords[0]) or np.allclose(coords, 0.0)`。第2項は軸全体が原点 1e-8m 以内のときだけ発火＝物理的に動く歩行者では非発火。warmup 用途は第1項で足り冗長。実害なし（クラリティ nit）。 | nit | `src/prediction/trajectory_predictor.py:289-291` |
| A3 | `ObstacleSet.get_all_obstacle_points` が予測時間軸を `.reshape(-1,2)` で平坦化＝CLAUDE.md 禁止操作。ただしリポジトリ全 grep で衝突判定・プランナ・シミュレータから未呼出の**デッドコード**。ライブ経路（`frenet_planner.py:1200-1233` の `_hits_dynamic`）は per-time-index で同時刻比較しており正しい。 | minor | `src/core/data_structures.py:236-252` |
| A4 | Observer は sgan_dt(0.4) が sim dt の整数倍でないと SGAN 入力を沈黙裡に破壊（非一様サンプル間隔＋warmup 後 is_ready=False）。全クローズドループは dt=0.1（4:1）、open-loop は sgan_dt=dt 強制（`run_openloop_prediction.py:82`）で発火せず。整数倍ガードが validate_config に無い。 | minor | `src/pedestrian/observer.py:62-86`、`integrated_simulator.py:325`、`config/__init__.py:206-219` |

- A1 修正: フォールバックでも `max(plan_horizon, pred_len*sgan_dt)` を使う、または失敗経路を `predictor.predict_cv` に通してホライゾン定義を一本化。
- A2 修正: `or np.allclose(coords, 0.0)` 項を削除（coords[0]-equality で warmup/static は被覆）。
- A3 修正: `get_all_obstacle_points` を削除（`__init__` export も）か「可視化専用・衝突判定で使用禁止」と明記＋プランナ/シミュレータが import しない回帰テスト。
- A4 修正: validate_config（and/or `PedestrianObserver.__init__`）で `sgan_dt/dt` の整数倍を assert。warmup 後 is_ready=False なら loud に fail。

### B. Frenet コア・幾何
| ID | 指摘 | Sev | file:line |
|----|------|-----|-----------|
| B1 | `calc_curvature_rate` 指数誤り（**R5 参照**）。 | minor | `src/planning/cubic_spline.py:249-269` |
| B2 | `CubicSpline1D` が重複/非増加 x（h==0）を受理し inf/nan 係数を沈黙生成（`if np.any(h < 0)` のみガード）。全シナリオは waypoint distinct のため非発火。将来の coincident/out-of-order waypoint で参照経路が沈黙破壊。 | minor | `src/planning/cubic_spline.py:26-45` |
| B3 | `find_nearest_point_on_path` が自己接近/ループ参照経路で stale `_prev_s` のとき誤ブランチを返しうる（±10m 窓内に2区間が入ると edge fallback が発火せず内部極小を採用）。現シナリオは非自己接近かつ連続前進のため非発火。 | minor | `src/core/coordinate_converter.py:220-248` |

- B2 修正: `if np.any(h <= 0): raise ValueError(...)`、and/or CubicSpline2D で ds==0 の coincident waypoint を dedupe。
- B3 修正: local 探索後に粗い global probe と比較し materially 近ければ `_global_search` へ fallback、ループスプラインの回帰テスト。

### C. 設定・バリデーション（単一情報源 invariant）
| ID | 指摘 | Sev | file:line |
|----|------|-----|-----------|
| C1 | `validate_config` が文字列型数値 YAML（例 `dt: "0.1"`）で生 TypeError を送出（ConfigValidationError でない）。型 coercion も isinstance チェックも無い。全シナリオは正しい型のため非発火。 | minor | `src/config/__init__.py:489-501,206-211` |
| C2 | `save_config` がデッドコードかつ ~21 キーを沈黙ドロップ（手書き dict、`dataclasses.asdict` 不使用）。k_*・footprint・chance_epsilon・sfm_v0_* など安全/手法比較/分布GT 鍵を含む。save→load で dataclass デフォルトへ revert。未呼出のため現影響なし。 | minor | `src/config/__init__.py:508-569` |
| C3 | recovery-gate ordering チェック（emergency≥caution）が mixed clearance-override/legacy 構成でスキップ（両 None または両 set の2分岐のみ）。全シナリオは両 override を等値設定のため非発火。mixed config で復帰階層が反転しうる。 | minor | `src/config/__init__.py:299-305` |
| C4 | `social_force_config` のパス存在チェックが無い（sgan_model_path とは非対称）。未設定シナリオのみのため非発火。誤パスは pysocialforce 初期化で生 FileNotFoundError。 | nit | `src/config/__init__.py:440-443` |

- C1 修正: validate_config の冒頭で数値/bool/str フィールドに型チェック（coerce or assert isinstance）、または load_config で TypeError も ConfigValidationError へ変換。
- C2 修正: `config_dict = dataclasses.asdict(config)`（config_path を pop）でロスレス round-trip、または未使用なら削除。
- C3 修正: 有効 caution/emergency clearance（override 優先・無ければ legacy-距離−combined_radius）を一度計算し無条件比較。
- C4 修正: `if config.social_force_config and not Path(...).exists(): errors.append(...)`。

### D. RQ1b 感度分析ハーネス
| ID | 指摘 | Sev | file:line |
|----|------|-----|-----------|
| D1 | `cache_path` が `Path(scenario).stem` のみでキー化＝base と rq1b/ 同名 variant が**同一キャッシュセルを共有**。過去の DEFAULT_SCENARIOS バグと同クラス。公開 1980-run は variant のみで生成されバイト安定再現済み＝現影響なし。将来 base geometry を同一 --root へ re-baseline すると誤 geometry のキャッシュを沈黙再利用。 | minor | `examples/run_da_poc.py:159-160` |
| D2 | `margin_verdict` の dominance 判定で inflation の min_dist 平均 i_d は全 run、time 平均 i_t は collision-free run のみ＝非対称フィルタ。robust arm には collision-free ガードがあるが inflation arm には無い。衝突する inflation が「robust に勝つ」と判定され `robust_gain_holds` を誤って False に反転しうる。実データでは全4GT で no inflation dominates のため公開結果は正しい。 | minor | `examples/run_rq1b_sensitivity.py:183-200` |

- D1 修正: cache キーに geometry discriminator（親込み相対パス or 解決済みパスのハッシュ）を含める、各 cache JSON と master_runs に full path を記録、最低限「異 geometry の re-baseline は新 --root 必須」を文書化。
- D2 修正: per-(scenario,inflation) の collision-free ガードを追加（`_cond_collisions(df,sc,inf)>0` なら dom=False）、または i_d を collision-free run で計算して i_t に揃える。回帰テスト（colliding-but-fast inflation が dominating に入らない）。

### E. RQ2 較正・評価
| ID | 指摘 | Sev | file:line |
|----|------|-----|-----------|
| E1 | 無境界 Nelder-Mead が (sigma,v0) identifiability ridge を登り、cruise-sensitivity 診断で非可換な v0 を生成（`--scenario vci_front` で sigma=9.097/v0=0.568、--no-refine は grid edge sigma=2.0/v0=2.0）。**コミット済みの canonical `--scenario all` 出力は interior（sigma~0.83）で本見出しに ridge-escape は含まれない**。主 LOCO 較正も interior で健全。検証者は major×0/minor×2/refute×1＝impact 限定で minor に整える。 | minor | `src/calibration/optimize.py:72-92`、`examples/run_rq2_cruise_sensitivity.py:111-132` |
| E2 | `vehicle_speed_samples` が DUT で負の車速を報告（生 vel_est をそのまま、p5=-0.00）。RQ2「速度ドメイン」百分位の物理的妥当性を損なう。limitation #2 の band は CITR（全正）で計算されるため主張は不変。 | minor | `src/datasets/vci_loader.py:212-222`、`examples/run_rq2_dut_validation.py:148-152` |
| E3 | pooled 'calibrated' closest-approach KS が fold 毎に異なる (sigma,v0) を混合（cross-validated mixture model）する一方、control arm は固定パラメータ＝非対称比較。write_summary には開示があるが headline_tests sidecar の note には伝播せず、多重比較 ledger 読者には mixture caveat が届かない。saturation で family が単一統計に collapse。 | nit | `examples/run_rq2_evaluation.py:219-243,503-510,310-381` |

- E1 修正: Nelder-Mead penalty で sigma に上限（grid_sigma 上限 or box constraint / L-BFGS-B）。refined sigma が grid span 外なら v0-shift verdict を抑制/フラグ。
- E2 修正: vel チャネルに `np.abs()`、または `agent_speed_samples`（NaN-safe・既存）で magnitude 再計算＋非負 assert。
- E3 修正: mixture-vs-fixed 非対称 note を `headline_tests()` の note に追加、固定平均 (sigma,v0) を全 held-out へ適用した like-for-like KS を併記。

### F. データパイプライン（VCI / ETH-UCY ローダ）
| ID | 指摘 | Sev | file:line |
|----|------|-----|-----------|
| F1 | `_ped_velocities` の有限差分フォールバックが全 grid の最終 index だけパッチ＝録画スパンが grid 末尾前に終わる agent の最終present フレームで速度 NaN。`extract_encounters` の present マスクが位置＋速度の両方を要求するため、位置が完全でも encounter から沈黙除外。全120 VCI CSV が vx_est/vy_est を持つため**有限差分分岐はデッドコード**＝現影響なし。 | minor | `src/datasets/vci_encounter.py:106-120,232-239` |
| F2 | `extract_fixed_windows` がフレーム時刻でなくリスト位置でスライド＝フレーム欠損を跨ぐ窓を一様ケイデンスとして扱う。eth 32%/hotel 38% の候補窓が穴を跨ぐが、fixed-population 交差要件が**生成済み窓では穴跨ぎを 0 件**に除外（実証済み）＝現 RQ1a 影響なし。将来の別コーパス/緩い min_peds/補間充填で非一様時刻を一様として供給しうる。 | minor | `src/datasets/eth_ucy_loader.py:169-181` |

- F1 修正: agent 毎に末尾速度を per-column 充填、または present マスクを位置のみにして境界速度を imputation。no-vx クリップの回帰テスト。
- F2 修正: 各候補窓で `np.all(np.diff(scene.frames[start:start+seq_len]) == scene.frame_step)` を assert し skip/raise。

### G. RQ1a ベンチマーク（統計フレーミング）
| ID | 指摘 | Sev | file:line |
|----|------|-----|-----------|
| G1 | 横断 ADE 集計の単位混在（**R1 参照**）。 | major | `examples/run_openloop_prediction.py:111-151` |
| G2 | stride=1 の重複窓で `n_trajectories` が強く擬似反復化（連続窓が 19/20 フレーム重複）＝独立サンプル数でない。cm 規模の序列差（zara2 per-agent lstm0.190 vs sgan0.220、hotel cv0.319 vs sgan0.325）に CI/有意性なし。M10 として既知だがハーネスは clustered 不確実性を未出力。 | minor | `examples/run_openloop_prediction.py:118-151` |
| G3 | best-of-N が学習モデルにのみ利益（CV は N=1）＝N=20 vs N=1 の非対称がどちらの報告 variant でも中和されていない。N=1 では zara1 で CV が両学習器に勝つ（CV0.436 vs SGAN0.621/LSTM0.679）。同一プロトコルを AVEC sim baseline にも適用するため H1（序列反転）は保存。 | minor | `examples/run_openloop_prediction.py:181-196` |

- G2 修正: 小差の序列主張には clustered/block bootstrap CI（窓 or 歩行者軌道を resample）、または非重複 stride=seq_len の robustness 行。最低限「n_trajectories は評価数であり独立サンプル数でない」と報告箇所に明記。
- G3 修正: N-matched 感度行（全手法 single-forecast、または lstm/sgan 間で fair な NLL）を primary に併記。絶対 real-data ADE が published minADE と非可換である旨を明記。

### H. 多重比較 ledger・状態機械（narrative / bookkeeping）
| ID | 指摘 | Sev | file:line |
|----|------|-----|-----------|
| H1 | `_thesis_paragraph` が「全 12 セル走査」を hardcode しつつ同文で動的 `m={fs['m']}` を補間＝同一量の二重表現。今は一致（全12 finite）。将来 NaN-p セル（空/退化 arm＝モジュールが設計上扱うケース）が出ると「全 12 セル走査（m=10）」と自己矛盾する thesis-ready 文を生成。render_markdown のテーブルラベル（`:242`）も同 hardcode。 | minor | `examples/make_multiplicity_ledger.py:286` |
| H2 | retry 成功が state machine へ報告されない（成功した retry 計画後も consecutive_failures>0 が残る）。update() は最初の計画でのみ実 result で呼ばれ、retry ループ内は常に found_path=False。NORMAL-fail→CAUTION-retry-success が CAUTION/consecutive_failures=1 を残す。挙動は保守的（CAUTION→NORMAL 復帰が最大1ステップ遅延）で state skip/無限ループ/escalation 漏れ/安全後退なし。記録 ego_state.state は正しく研究指標は不変。 | minor | `src/simulation/integrated_simulator.py:591-644` |

- H1 修正: hardcode '12' を live 値へ（`f'弱パワー corner を含む全 {fs["m"]} セル走査'`、テーブルラベルも `sens['full_scan']['m']` から導出）。
- H2 修正: retry 成功時に break 前へ `self.state_machine.update(True, ...)` を呼ぶ、または「retry-rescued step は recovery-quarantine 上意図的に failure 扱い」と明記。

### I. テスト・再現性ギャップ（R2/R3/R4 以外）
| ID | 指摘 | Sev | file:line |
|----|------|-----|-----------|
| I1 | `DEFAULT_SCENARIOS` が scenarios/rq1b/ variant を指すことを pin する回帰テストが無い＝最も影響の大きい過去の再現性バグ（commit 8524708）が無防備。現値は正しいが、デフォルトを base へ revert する1行変更で 333 テスト全 green のまま silent-wrong-experiment が再発しうる。検証者は major×2/minor×1＝「現コード正・潜在 latent」のため minor に整える。 | minor | `tests/test_rq1b_sensitivity.py`（欠如）／cf. `examples/run_rq1b_sensitivity.py:47-51` |
| I2 | SimulationConfig dataclass ⇄ validate_config の parity を強制するテストが無い。新キーを検証なしで追加しても全テスト pass（save_config の config_dict も既に非網羅）。 | minor | `tests/test_config.py`（欠如）／cf. `config/__init__.py:10-187 vs 194-464` |
| I3 | シナリオ planner cost-weight 公平性（k_j/k_t/k_d/k_s_dot/k_lat/k_lon の全シナリオ統一）を pin する回帰テストが無い。現在は override 不在で「省略により成立」。将来の per-scenario 重み override が cv/lstm/sgan 比較を沈黙バイアスしうる。 | minor | `scenarios/`（+tests/ 欠如） |
| I4 | 同一ステップ escalate-and-retry ループ（実効最大2回・カウンタ上限3 backstop）が未テスト。`test_long_run_stability` は happy-path の有限性のみ。escalation 駆動 retry・cap-3 backstop・`_apply_emergency_stop` への fall-through が未 assert。SM 遷移と emergency-stop kinematics は別途テスト済み。 | minor | `src/simulation/integrated_simulator.py:382-383,595-604`（tests 欠如） |

- I1 修正: `run_rq1b_sensitivity.DEFAULT_SCENARIOS == ['scenarios/rq1b/scenario_01.yaml', ...]` と各 path が `scenarios/rq1b/` で始まり存在することを assert。base scenarios が default に無いことも optional に。
- I2 修正: `dataclasses.fields(SimulationConfig)` を introspect し各非内部フィールドが validate_config 参照 or curated set に在ることを assert＋save→load round-trip テスト。
- I3 修正: 全 `scenarios/*.yaml` と `scenarios/rq1b/*.yaml` を load し全 k_ 重みが共有定数（1.0）と等しいことを assert。
- I4 修正: NORMAL/CAUTION で fail・EMERGENCY で success する planner stub で step() を駆動し escalation・path 由来・`_replan_attempts<=3` を assert、全失敗で `_apply_emergency_stop` 起動を assert。

---

## 完全性ギャップ（深くレビューしていない領域／critic 指摘）

以下は本監査でユニットとして扱わなかった、または番人の不在が将来リスクとなる領域。**現結論への影響は無いか限定的**だが、執筆／流用前に確認価値がある。

1. **〔Major〕RQ1a 見出し集計のバージョン管理外性**（R1/G1 と同根）: H1 見出しは per-(scene,method,seed) CSV から手計算で再構成され、集計選択（非加重 vs trajectory 加重、scene-level vs per-agent、eth 含む vs 除く）が RQ1a 中で**最も結論を左右する決定**でありながらバージョン管理外。`examples/aggregate_rq1a.py` をコミットし全集計を並置＋回帰テストで pin。
2. **〔Major〕確率実行の決定論回帰テスト不在**（R2 と同根）: バイト安定性は脆い RNG 順序契約に依存し、手動 spot-check のみで強制 invariant 化されていない。
3. **〔Minor〕VCI ローダの内部フレーム穴を直線補間**: `_resample_agents`（`src/datasets/vci_loader.py:136-146`）はスパン内の欠損を np.interp で直線充填＝多秒オクルージョンを等速直線として描く。補間位置が `ped_vel`（`vci_encounter.py:106-120`）へ有限差分され cruise 速度と avoidance-onset 加速度に影響、onset KS と standoff gap が補間アーティファクトを吸収しうる。gap 長ガード／診断なし。閾値超え（例 2 grid step）の agent をフラグ/drop、fold 毎の補間フレーム割合を報告、合成穴の単体テスト。
4. **〔Minor〕多重比較 family サイズが実行時にディスク上の sidecar glob で決まる**: `make_multiplicity_ledger.py:62` の `glob('**/headline_tests*.json')`。stale/欠損/重複 sidecar で family m が動き、全 q/Holm-p がシフト＝「補正後生存」verdict が変わる。malformed はスキップするが**完全性/重複なし**を assert しない。期待 source 集合の assert か manifest、既知 sidecar 集合で family サイズを assert するテスト。
5. **〔Minor〕PedestrianObserver が窓内固定歩行者数を無検証で前提**: `update()` は最新フレームの count を n_peds とし deque に旧フレームを保持（`observer.py:75-77,123`）。窓内で population が変わると np.stack が raise（安全）or 形状一致時に**異なる歩行者を時刻間で誤対応**しうる。現データは固定 N 保証。窓内 n_peds 一致の assert＋population 変化で loud fail するテスト。
6. **〔Nit〕可視化・図版サブシステム未レビュー**: `src/visualization/animator.py:545-568` に4段の `except Exception` で描画エラーを沈黙吸収。`examples/plot_simulation_figs.py`／`plot_lateral_analysis.py` は AVEC 論文図を生成（→`~/Research/AVEC_FullPaper/figs/`）。誤ラベル/誤単位/スキップフレームを捕捉するテストなし。AVEC は凍結だが修論図のテンプレ。流用時は fixture npz での smoke テスト＋except 句の特化。

---

## 反証された指摘（非問題・付録）

検証で「実バグでない／過大主張」と判定された項目。**再掲・再昇格しない**。

- **min_separation の NaN 伝播**（`src/core/metrics.py`）: 上流（VCI ローダ）が present-throughout・有限の固定N配列のみ供給するため NaN は到達しない。防御的提案＝nit 未満。
- **avoidance-onset KS が onset-RATE を捨てる**: 統計は正しく n_sim/n_real も隣接表示・caveat 済み。headline/ledger 非依存の diagnostic。新指標の enrichment 提案であり defect でない。
- **normalize_angle docstring の範囲表記**: 実装は `normalize_angle(-pi)` が正確に -pi を返す（[-pi,pi]）。指摘の「+pi にマップ」は誤り。機能影響ゼロ。
- **H1 序列が best-of-N variant で反転（scene-level vs per-agent は非等価証拠）**: 両 variant はコードで正しく計算・per-agent を canonical と明記・並置保存・テストで pin・docstring で開示済み。さらに指摘の数値（CV が eth で勝つ＝3/5）は CSV と不一致（eth で CV は最悪、真は 2/5）＝過大主張。
- **NLL が lstm/sgan のみ＝尤度指標が 2-way**: point forecast の CV に NaN を返すのは数学的に正しく意図的・回帰テスト済み・LaTeX は "--"・axisA review m11 で開示済み。
- **process_prediction の near-origin clamp が実動作を捨てる**: 10nm 以内でのみ発火＝物理的に動く歩行者で発生不能。指摘メカニズムは実在しない。
- **RQ1a open-loop ハーネスの coverage ゼロ（major 主張）**: コードは正しく（再構成は正確、N=1 恒等・per-agent best-of-N も正）、metrics 層は網羅テスト済み。本レビューでは「テスト欠如」の側面のみ I/R 群として扱い、active defect とはしない（一部検証者が再昇格を試みたが、現コード正・現出力バイト安定で active defect でないとの判定が優勢）。

---

## 優先アクション（執筆前推奨順）

1. **【最優先・研究主張】R1/G1**: RQ1a 横断集計スクリプト `examples/aggregate_rq1a.py` をコミットし、trajectory 加重・eth 除外・per-agent を並置。修論で H1 を支える集計を明示し、集計選択への（非）不変性を正直に記述。eth は sensitivity 行扱い。
2. **【最優先・テスト番人】R3/R4/I1**: (a) 予測失敗フォールバック＋5-strike ガード、(b) RQ2 pooled-KS／saturation family、(c) DEFAULT_SCENARIOS が rq1b/ variant を指す——の3回帰テストを追加。いずれも claim 直結経路で番人ゼロ、追加は trivial。
3. **【再現性】R2**: method=sgan・同一 seed のビット一致テスト＋trajectory.npz golden hash で「バイト安定」を強制 invariant 化。
4. **【数値プリミティブ】R5/B1**: `cubic_spline.py:269` を `d^1.5/d^2.5` に修正＋曲線スプラインの有限差分回帰テスト。S3 の comfort/jerk を再生成（手法序列は不変なので主結論の再計算不要）。
5. **【潜在ハード化】B2/A4/C1/C3/D1/D2/E1/E2**: loud-fail ガードと cache キー discriminator を追加（h<=0 raise、sgan_dt/dt 整数倍 assert、文字列数値→ConfigValidationError、recovery-gate 無条件比較、cache キーに geometry、inflation collision-free ガード、Nelder-Mead sigma 上限、DUT 車速 abs()）。
6. **【narrative / parity】H1/I2/I3**: hardcode '12' を live `fs['m']` へ、config parity テスト、k_* 公平性テストを追加。
7. **【完全性（任意）】**: VCI 内部穴の補間診断、ledger sidecar manifest、observer 固定N assert、図版 smoke テスト——修論図/データを流用する前に。
8. **【整理】A3/C2**: デッドコード `get_all_obstacle_points`・`save_config` を削除 or 用途明記（誤用防止）。

---

*本レビューは検証者合議を通過した確定指摘のみを掲載。Critical 0・現行コミット済み結論を覆す active bug 0。コードベースは健全で、残課題は「執筆時の集計判断」と「研究主張を守る回帰テスト」の整備に集約される。*
