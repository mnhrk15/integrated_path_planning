# コードベース徹底レビュー報告（2026-06-11）

多段マルチエージェントレビューの結果。構成: モジュール別8系統 + 横断3系統（時間軸/衝突安全/エッジケース）+ テスト品質 + pytest実実行 + 網羅性クリティックによる追加4系統（pysocialforceラッパ、run_simulation.py、論文図版、multi-circle footprint統合）。
生の指摘113件 + 追加ラウンド24件のうち、敵対的検証（critical/majorは反証・再現・設計意図の3視点多数決、minorは反証1票）を通過した**89件**を重複統合して記載する。反証された6件は末尾に記録。

**テスト実行結果: tests/ 全70件合格（14.66s）。警告なし。** ただしテストの検証力には重大な穴がある（§6）。

数値検算により**正しいことを確認した部分**: quintic/quartic多項式の係数行列と1〜3階微分、cubic spline係数（scipyと完全一致）、符号付き曲率式、Frenet⇄Cartesian往復変換（機械精度で一致）、KDE-NLL（Scott則・logsumexp・フロア）、SGAN流best-of-N ADE/FDE、multi-circle footprint幾何、chance_epsilonの意味論（ε=0.0でworst-case）、計画時衝突閾値と評価時衝突閾値の数値一致（1.2m）、プランナの同時刻衝突判定のインデックス整列（path_t/dt）、予測失敗時CVフォールバックのホライゾン分時系列生成、統計ベンチマークの同一シード統制とキャッシュの原子的書き込み。

---

## 1. CRITICAL（3件）

### C-1. PedestrianObserver のダウンサンプリングが経過時間を多重加算 — SGAN入力が0.4s間隔にならない
`src/pedestrian/observer.py:56-73`

> **[2026-06-11 修正済み]** `_last_update_timestamp` を導入して delta_t を「前回update呼び出し時刻」基準に変更、剰余を減算に置換。回帰テスト `tests/test_observer.py`（8件）を追加。同一シード（scenario_01, seed=0）での修正前後比較: サンプル間隔 0.2-0.3s→正確に0.4s、ADE 2.416→1.088、FDE 4.356→1.877、min_dist 3.15→3.58。**全ベンチマークの再計測が必要。**

`delta_t = ped_state.timestamp - self.timestamps[-1]` の `timestamps[-1]` は「最後にサンプリングした時刻」だが、`accumulated_time += delta_t` は毎ステップ実行されるため、非サンプリングステップで同じ経過時間が重複加算される（サンプル後 0.1+0.2+0.3=0.6 が3ステップで積もる）。**実行で再現済み**: dt=0.1 でのサンプリング時刻は 0.4, 0.7, 0.9, 1.1, 1.2, 1.4, … と間隔0.1〜0.3s（平均約0.2s）になり、0.4sには一度もならない。

影響:
- SGAN/LSTMへの8フレーム観測が想定3.2sではなく約1.5〜1.7s分・不等間隔となり、0.4s学習済みモデルには歩行者が約半分の速度に見える。
- `predict_cv`（trajectory_predictor.py:195）は隣接サンプル差分を 0.4 で割るため速度を25〜75%過小評価（実測で見かけ速度 0.3〜0.9 m/s に振動、真値1.2 m/s）。
- **cv/lstm/sgan 全手法の予測入力を歪めるため、論文のADE/FDE/KDE-NLL・回避挙動・安全指標すべてに影響。** テストは存在しない。

修正: 「前回のupdate呼び出し時刻」を別途保持して delta_t を計算する。あわせて `accumulated_time %= sgan_dt` を `accumulated_time = max(accumulated_time - sgan_dt, 0.0)` に（浮動小数点エッジで直後再サンプリングする問題）。**修正後は全ベンチマーク数値が変わるため再計測必須。**

### C-2. run_simulation.py の --method がモデルパスを切り替えない — 「lstm」ラベルでサイレントにCVが走る
`examples/run_simulation.py:104-106` + `src/prediction/trajectory_predictor.py:151-158` + `src/simulation/integrated_simulator.py:446-466`

`--method` は `prediction_method` のみ上書きし `sgan_model_path` を切り替えない。`--method lstm` をベースシナリオに渡すと pooling付き重み（sgan-p-models）のまま lstm モードになり、実行時の `generator.pooling_type = None` 差し替えが `mlp_decoder_context`（構築時 input_dim=40）との次元不一致で**毎ステップ RuntimeError** → integrated_simulator.py:446 の広域 except が握り潰して**CVフォールバック**。シミュレーションは正常終了し「lstm」ラベルの成果物が実体CVで生成される（**実行で実証済み**）。逆方向（`--method sgan` + _lstm シナリオ）はクラッシュすらせず pooling なし重みのまま 'sgan' と記録される。

**既存成果物の汚染調査済み**: output/scenario_0X 系の metrics_report.txt は全件 method とモデルパスが整合しており、_lstm/_cv 派生YAML経由で生成されたもの。現時点で汚染なし。ただし CLAUDE.md がこのコマンドを主要導線として案内しているため今後の再生成で発火する地雷。

関連（同根の構造問題）:
- 実行時 pooling トグル自体が無意味: no-pooling チェックポイントでは no-op、pooling 付きでは例外（trajectory_predictor.py:151-158）。load_model 時に checkpoint args の pooling_type と method の整合検証 + 不一致で raise に置き換えるべき。
- `resolve_model_path` のフォールバック（run_statistical_benchmark.py:29-40）: sgan-models 不在時に warning のみで pooling 重みのまま LSTM 続行。FileNotFoundError で停止すべき。
- `--method` 上書きが validate_config 後のため整合チェックを迂回（run_simulation.py:97,104-106）。
- 広域 except が恒常的失敗（モデル未ロード等）も毎ステップCVに縮退させ、メタデータは 'sgan'/'lstm' のまま（integrated_simulator.py:446-466）。連続N回失敗で raise すべき。

### C-3. 緊急停止でEgo位置が更新されず「その場で瞬間停止」+ ego_state エイリアシングで履歴が遡って書き換わる
`src/simulation/integrated_simulator.py:676-688, 594-610, 641-664`

(a) **制動距離ゼロ**: `_apply_emergency_stop()` は v/a/jerk/timestamp のみ更新し x/y/yaw を据え置く。計画失敗ステップでは v>0 のまま位置が凍結され、実車なら v²/(2·a_dec)（v=8.33 m/s, 4 m/s² で約8.7m）前進して歩行者に到達しうる局面で到達距離ゼロ。**フェイルセーフ作動時の衝突を系統的に見逃し、collision_count・min_dist が楽観側に歪む。** S2狭路など計画失敗が起きるランの結果に直結。なお減速度4 m/s²固定は EMERGENCY 制約緩和（6 m/s²）とも不整合。

(b) **履歴エイリアシング**: 計画成功時は `get_state_at_index(1)` が新オブジェクトを返すが、緊急停止は `self.ego_state` をインプレース変更する。`SimulationResult` は参照を保持するため、計画失敗が2ステップ以上連続すると**過去ステップの履歴 v/a/jerk が最後の値で塗りつぶされる**。急制動開始時のjerkスパイクが消え max_jerk/rms_jerk（乗り心地指標）が過小評価、trajectory.npz・dashboard・アニメーションにも波及。

修正: 緊急停止で `x += v·cos(yaw)·dt, y += v·sin(yaw)·dt` を積分し、冒頭で `dataclasses.replace(self.ego_state)` で新オブジェクトに差し替える。

---

## 2. MAJOR: 時間軸の系統ズレ群（同根3層 + 関連minor）

最重要テーマ。予測パイプラインの時刻アンカーに3つの独立したズレが重なっている。**全手法共通バイアスのため手法間ランキングは概ね保たれるが、論文に載せる絶対値（ADE/FDE/NLL/min_dist）を歪め、SGAN文献値との比較を不能にする。**

### M-1. 予測の時刻原点が「最終観測サンプル時刻」（最大0.3s過去）なのに「現在時刻」として消費される
`src/prediction/trajectory_predictor.py:231-234` / `src/simulation/integrated_simulator.py:472-495`

Observer は0.4s毎にしかサンプルしないが予測は毎0.1sステップで実行され、`time_src = (1..12)*0.4` は最終観測フレーム基準。4ステップ中3ステップで予測軌道全体が最大0.3s分シフトし（歩行者1.4 m/sで約0.4m、結合半径1.2mと同オーダー）、(a) プランナの同時刻衝突判定、(b) predict_cv のアンカー、の両方に入る。修正: staleness = 現在時刻−最終観測時刻 を predictor に渡し time_src をシフトして再アンカー。

### M-2. ADE/FDE/KDE-NLL の GT 対応付けが同じアンカーずれを共有
`src/core/metrics.py:36-77, 111-150` （calculate_planning_ade_fde 含む）

`pred_indices = 4k-1` の予測点を `history[i+4k]` のGTと比較するが、予測アンカーは最大0.3s古い。評価原点の3/4で「アンカー+0.4k秒の予測」vs「現在+0.4k秒のGT」の比較になり、約0.2m級の系統誤差がADE/FDE/NLLに上乗せ（悪化方向）。標準SGAN評価プロトコル（最終観測フレーム=共通t=0）から逸脱。修正: Observerがサンプルしたステップ（staleness=0）のみを評価原点にするか、GT抽出オフセットを補正。

### M-3. 歩行者を先に t+dt へ進めてから時刻 t のEgoで計画する恒常的 +0.1s 位相ずれ
`src/simulation/integrated_simulator.py:612-638`

障害物時刻−Ego時刻 = (1−j)·0.1s（jは観測位相）となり、M-1と合成すると -0.2s〜+0.1s の周期的ジッタ。「同時刻位置のみ評価」の不変条件を厳密には破る。

### 関連minor（同テーマ）
- **補間の左端クランプ**: time_src が0.4s始まりのため、密予測の index 1〜3（0.1〜0.3s）が「0.4s先の予測位置」で埋まり、index 0→1 で最大約0.6mのジャンプ（trajectory_predictor.py:256）。現在位置を t=0 アンカーとして補間に加えるべき。M-1とは逆方向のずれで部分相殺あり。
- **trajectory.npz の times が状態より1ステップ古いラベル**（integrated_simulator.py:641-667）: ego_state.timestamp=t+dt なのに result.time=t。全図版・アニメ・衝突ログの時刻が一律0.1s早い。Start マーカーも真の初期位置から約0.5mずれる（履歴に t=0 レコードがない。plot_simulation_figs.py:70、実npzで確認済み）。
- **dt が0.4の約数でない場合の検証なし**（config）: dt=0.15等で実サンプル間隔0.45sがSGANに0.4sとして渡る（現行dt=0.1のため潜在）。

---

## 3. MAJOR: 設定・公平性

### M-4. load_config() は default_config.yaml を一切読まない — CLAUDE.mdの「マージ」記述と乖離、3点同期が実質崩壊
`src/config/__init__.py:337-375`

default_config.yaml を読むコードはリポジトリに皆無。実際のデフォルトは dataclass 初期値で、既にドリフト: pred_len（yaml 12 vs dataclass 8）、d_road_w（0.3 vs 0.5）、k_j/k_lat（0.8 vs 1.0）。yaml側には num_samples / distribution_aware_planning / chance_epsilon が存在しない。新規シナリオで pred_len を省略すると 8 が適用されSGANの12ステップ前提と矛盾（現行9シナリオは明示しているため未発火）。修正: マージを実装するか、default_config.yaml を削除して dataclass を唯一の情報源と明記。

### M-5. _cv/_lstm 派生シナリオの設定差分 — 直接実行すると公平比較が崩れる
- **旧コスト重みの残存**: scenario_02_cv/_lstm は k_j=0.5,k_d=0.5,k_lat=0.5、scenario_03_cv/_lstm は k_j=0.5,k_s_dot=0.5,k_lat=1.5 を保持（ベースは全1.0に統一済み）。
- **num_samples=1**（全6派生、ベースは20）: lstm が分散低減なしの生サンプル、sgan が best-of-20 となり、pooling以外の要因が混入。

**run_statistical_benchmark 系（ベースYAML + prediction_method 上書き方式）は影響なし**だが、派生YAMLの直接実行（run_simulation.py / benchmark_prediction.py）で発火する。派生をベースと揃えるべき。

### M-6. chance_epsilon の範囲バリデーション欠如 — ε≥1.0 で動的衝突制約が無警告で完全無効化
`src/config/__init__.py` / `src/planning/frenet_planner.py:808`

`max_violations = floor(ε·n)` のため ε≥1.0 で全サンプル衝突でも「衝突フリー」。パーセント表記のつもりの誤記（5, 20等）で衝突を見逃す。`0.0 <= ε < 1.0` のチェックを追加。関連: distribution_aware_planning=true かつ num_samples=1 だと分布がNoneになり無言で単一サンプル計画に退化（validate で要求すべき）。

---

## 4. MAJOR: プランナ・状態機械・幾何

### M-7. S1/S3 の横方向サンプリング格子に d=0 が含まれない — 恒常的 −0.1m オフセット
`src/planning/frenet_planner.py:288`

`np.arange(-max_road_width, max_road_width, d_road_w)` は max_road_width=7.0, d_road_w=0.3（S1/S3）で {−7.0,…,−0.1, 0.2,…,6.8} となり d=0 が存在しない（実測確認）。障害物がなくてもエゴは参照線から約0.1m左にずれて走り続け、S2（d_road_w=0.2、d=0を含む）と格子構造が異なるため、**横方向解析（論文Fig）・乗り心地指標のシナリオ間比較に系統バイアス**。回避コリドーも左右非対称（−7.0 vs +6.8）。0中心の対称格子（linspace）に変更すべき。

### M-8. Frenet微分の規約混在（空間微分 d' vs 時間微分 ḋ）により fp.yaw が経路接線と乖離
`src/planning/frenet_planner.py:484-512` / `src/core/coordinate_converter.py:72,137`

cartesian_to_frenet は Apollo 規約の空間微分 d'=dd/ds を返すが、横方向 quintic は時間グリッド上に構築され、frenet_to_cartesian はその時間微分を再び空間微分として解釈する。**数値検証済み**: 直線参照・yaw=15°・v=5 m/s で格納 yaw とポリライン接線の乖離が最大14〜27°。multi_circle footprint はこの yaw で円を展開するため、回避操舵中（まさにfootprintが効く局面）にノーズ円が最大0.36〜0.7m横にずれる。初期横速度も d'（tan15°≈0.27）を ḋ（v·sin15°≈1.29 m/s）の代わりに使うため横応答が約 s_d 倍鈍く計画される。計画と評価は内部整合するが、車両は非ホロノミック制約違反の横滑り運動として記録される。修正: 初期条件に ḋ=v·sinΔθ を渡し、逆変換へは d'=ḋ/ṡ を渡す（規約統一）。fp.yaw とポリライン接線の一致を回帰テスト化。

### M-9. 1−κd≤0 の特異点ガードなし — S3で特異点越えの退化候補が妥当性チェックを通過
`src/core/coordinate_converter.py:71,134-156`

S3 の最大曲率 |κ|=0.185（曲率中心まで5.41m）に対し max_road_width=7.0 のため、|d|∈(5.41,7.0) の候補が実際に生成される。**数値検証済み**: d=−5.5以降で theta がπ反転、指令3 m/s に対し変換後 v=0.05〜0.6 m/s となるが、_check_paths は変換後の値しか見ないため全制約を通過し 'ok' に分類される。通常は横偏差コストで選ばれないが、EMERGENCY の制約緩和で他候補全滅時には選択され得る。1−κd≤ε の点を含む経路を invalid にすべき。

### M-10. 状態機械の復帰クリアランス閾値がデフォルトで負 — 距離ゲートが実質無効
`src/core/state_machine.py:38-46,70-87`

clearance_caution = 0.5−(1.0+0.2) = **−0.7m**、clearance_emergency = **−0.2m**。歩行者が車両に食い込んでいても計画が1本見つかれば EMERGENCY/CAUTION から復帰できる。全シナリオがデフォルトを使用しているため**全実験でフェイルセーフ復帰が実質 plan_found のみで駆動**されている。論文でフェイルセーフ階層を主張する場合は記述と齟齬。safe_distance を結合半径より大きく（例 2.0/3.0m）し、validate_config で `safe_distance > ego_radius+ped_radius` を要求。multi_circle 時は footprint.radius(1.25m) 基準への換算も必要。

### M-11. 「再計画3回/ステップ」は実装に存在しない（最大1回/ステップ、上限カウンタはデッドコード）
`src/simulation/integrated_simulator.py:550,581-585,670-672`

`_replan_attempts` は毎ステップ末尾で0リセットされ、リトライはループでなく単発if。581行のelifは到達不能。挙動自体は安全側（停止フォールバック）だが、CLAUDE.md・論文記述と実装が不一致。whileループ化するかドキュメントを「状態昇格時に1回のみ再計画」へ修正。

---

## 5. MAJOR: ベンチマーク・再現性

### M-12. 衝突でランが即 break され、time_s/speed_ms の平均を歪め goal_reached を誤判定
`run_da_poc.py:93-106` / `run_footprint_benchmark.py:160-180` / `benchmark_prediction.py:81-96`

`goal_reached = end_time < total_time − 1.5dt` は衝突早期終了も「ゴール到達」と判定（衝突時刻 < cap で必ずTrue）。衝突ランほど time_s が短く「高速」として平均に混入。現行 comfort_s* キャンペーン（衝突0件）には影響しないが、chance_epsilon>0 等の追加実験で系統的に誤る。終了理由（goal/collision/timeout）をシミュレータから返して記録すべき。

### M-13. run_statistical_benchmark: 失敗ランが無言で平均から脱落、LaTeXキャプションは「over 20 runs」固定
`run_statistical_benchmark.py:81-83,106,216-230`

難しいシードほど失敗しやすい場合に安全指標が楽観側へ選択バイアス。失敗数の警告・非ゼロ終了なし（run_da_poc とは非対称）。キャプションの n を summary の n_runs から動的生成すべき。

### M-14. シード制御の欠落
- `benchmark_prediction.py`: シードなし + cv→lstm→sgan を同一プロセス順次実行で乱数状態が手法間に持ち越し。単発比較表を論文根拠にするのは危険。
- `run_simulation.py`: シードなし。**論文図版の元データ trajectory.npz が非決定的**（output/ は gitignore のため成果物も残らない）。--seed 引数と metrics_report.txt へのシード記録を追加すべき。

### M-15. 計画時間計測がリトライの plan() を含まない
`integrated_simulator.py:514-523,565-572` / `measure_proc_planning.py`

t_plan は初回 plan() のみ計時。状態機械がエスカレーションする最も重いステップで最大2倍の過小評価。robust/multi_circle 条件はリトライが起きやすく、「100x check points → 2.06x」（コミット 923b2aa）の条件間比較が不公平に有利化される恐れ。リトライ分を t_plan に加算すべき。

---

## 6. MAJOR: テストの検証力の欠陥（全70件合格の裏で）

### M-16. 中核不変条件「同時刻位置のみの衝突判定」がどのテストでも判別不能
`tests/test_frenet_planner.py:134-189`

全動的衝突テストが「時刻インデックス0で衝突する」配置のため、**平坦化実装や常時インデックス0参照への退行でも全テスト合格**（各ケースの座標で検算済み）。「障害物が経路点Pの位置を別時刻に通過する→衝突なし」「同時刻に通過する→衝突あり」の判別ケースを追加すべき。

### M-17. 0.4s→0.1s 補間・外挿コードがテストで一度も実行されない
`tests/test_trajectory_predictor.py:40-96`

ダミー生成器がゼロ軌道を返すため定数分岐に入り、np.interp・テール外挿・MAX_WALKING_SPEEDクランプが未実行。アサーションも自己充足的。C-1（Observer）も完全未テスト。非自明な等速軌道での厳密assert + Observer のダウンサンプリングテストを追加すべき（C-1修正の回帰テストにもなる）。

その他のテスト品質: test_animator はモックが実構造と不一致の純粋煙テスト（src/core/state.py はこのテスト専用のレガシー）/ test_pedestrian_simulator に名前が示す対象を検証しないテスト2本 / test_state_machine は「成功2回で回復」ゲート未検証（条件削除でも全合格）/ test_long_run_stability はSGANモデル不在環境で skip せず失敗 / 設定3点同期・コスト関数・曲線参照経路での座標変換・chance-constrained統合経路・plan()経由のmulti-circleが未カバー。

---

## 7. MINOR（統合・抜粋）

**プランナ**:
- np.arange の端点排除と浮動小数点誤差で候補数が target_speed の値依存に±1変動（frenet_planner.py:277-284）。linspace化推奨。
- 時間格子が終端 Ti を含まず、終端境界条件・終端コストが Ti−dt で評価。実効ホライゾンは max_t−2dt=4.8s（論文で「5.0s」と書く場合は注意）。
- CAUTION の max_speed 締め付けが経路 index 0（現在速度）にも適用され、現在速度超過時に全候補棄却→不要なEMERGENCYエスカレート（frenet_planner.py:711）。
- スプライン定義域外で打ち切られた経路の未チェック区間が候補に残る / 終端付近で長さ1経路が「成功」扱い（状態機械に失敗が伝わらない）。
- 制約チェックが NaN を素通し（`any(v > limit)` は NaN で False）。
- エゴ曲率を常に0と仮定（_cartesian_to_frenet_state）→ 旋回中の初期 d_dd/s_dd が不正確、ジャーク指標へのリプラン不連続汚染。

**スプライン**: 重複ウェイポイントが警告のみで全NaN化（h==0 を許す）/ 弧長が弦長近似のため S3 旋回部で単位速さ仮定が最大約8%破れる（S1/S2 は直線で影響なし）。

**設定**: save_config() が多数フィールドを書き出さずラウンドトリップでデフォルトに戻る（現在未使用の潜在バグ）/ obs_len・pred_len と checkpoint args の突合なし / sgan_model_path が CWD 相対。

**pysocialforce ラッパ（地上真値生成器）**:
- social_force_params のパススルーは factor・scene系キーで**サイレント no-op**（Force.init で初期化時キャッシュ済み）。将来の感度実験で無効なまま解釈する危険。
- `Config.__call__` の `or default` により 0/0.0/False の上書きがデフォルトに差し替わる（アブレーション実験が無警告で失敗）。
- ゴール0.5m以内に到達した歩行者は ego 斥力を完全無視して凍結（psf仕様）。現行シナリオでは sim 時間内に未発火だが、シナリオ追加時に「絶対回避しない静的障害物」が生まれる。
- 退化box（幅0）で同一線分が2重登録され斥力2倍 / len≠4 要素はサイレント破棄。

**メトリクス/ベンチマーク**: 衝突中（食い込み中）のTTCがinfになり min_ttc から最悪ケースが欠落 / min_ttc=inf のランが1つでもあると mean が inf で表が壊れる / LaTeX表の決定論判定が time_s_std==0 だけで全カラムの±stdを抑制 / footprint円のTTCがヨーレート項 ω×r を無視 / run_footprint_benchmark の --total-time がキャッシュキー外でガードなし。

**可視化/図版**: dashboard・animator の安全しきい値線が1.0m固定（実衝突しきい値1.2m）で図とメトリクスの安全/衝突の読みが食い違う / plot_lateral_analysis の ψ ペインが他ペインと1サンプルずれ（ego_yaw キーを使えば解決）/ make_margin_report の sanity チェックが inf−inf=NaN で完全一致ランをFAIL判定 / experiment_a の有意性判定でデータ欠損条件が「有意差なし」側に分類 / animator の transpose ヒューリスティックが n_peds==pred_len で誤転置 / run_simulation.py のアニメ生成失敗が握り潰され終了コード0 / --steps 0 で IndexError。

**シミュレータ**: 現在位置 prepend の np.allclose 判定がデータ依存（静止歩行者で時刻インデックスが1ステップずれうる）/ 観測未準備時のフォールバックが単一時刻スライス（warmupにより現行は到達不能）。

## 8. INFO（設計上の留意点、42件から抜粋）

- 時間コスト Jt が lat/lon 両方に加算され実効重み k_t·(k_lat+k_lon)（PythonRobotics と同構造。論文で重みを報告する際の解釈注意）。
- ObstacleSet.get_all_obstacle_points は時間次元を平坦化する**罠コード**（現在デッドコードだが不変条件と正面矛盾。削除推奨）。
- ウォームアップにより歩行者は t=0 までに3.2s移動済み — 遭遇ジオメトリが dt・obs_len に結合（アブレーション時の注意）。
- num_samples>1 のとき毎0.1sステップで N 回順伝播（観測は0.4s毎しか更新されないのに再サンプル）— 計算4倍冗長 + 分布のチャタリング + 計画時間報告値への影響。
- ego 斥力は等方・速度非依存で「常に譲る歩行者」に偏る（S2 のゼロ衝突天井を楽観側に支える設計要因）。v0/sigma の命名は Helbing ポテンシャルを模すが 1/σ 係数が欠落しており、S2 の v0=2.1, sigma=0.3 はポテンシャル解釈なら実装の約3.3倍 — 地上真値キャリブレーションの意図と要照合。
- 歩行者の実効希望速度は「初期速度ノルム×1.3」（psf の max_speed_multiplier）— 論文で歩行者速度を報告する際は実効値で。
- S1/S3 の図中道路境界は衝突判定上の実体なし（map_config は可視化専用、static_obstacles は空）。
- predict_cv（手法としてのcv）と予測失敗時CVフォールバックは時刻原点・速度推定が異なる別物。
- min_distance の意味が footprint モード間で異なる（モード横断比較は clearance を使う）。
- 歩行者対応は配列インデックス順序のみ依存（動的追加・離脱を入れると壊れる）。

## 9. 反証された指摘（6件）

検証エージェントの多数決で棄却:
1. 観測未準備時の単一タイムステップ障害物のフォールバック不変条件違反 → warmup により到達不能（infoに降格して記録済み）。
2. collision_count が「イベント数」でなく「ステップ数」 → 設計意図と判断。
3. ped_initial_states 空のときの ped_groups 検証スキップ → 実害なし。
4. 分布サンプル prepend の非対称 → 実害は限定的（minorとして別途記録あり）。
5. step_width 直接代入の hasattr ガード → 現行psfバージョンで発火せず。
6. scenario_03 の lateral shift 注釈のハードコード → 図注釈であり意図的。

## 10. 推奨対応順

1. **C-1（Observer）を修正 → 全ベンチマーク再計測**。論文数値の土台。修正は数行、影響は全表。
2. **C-3（緊急停止の位置凍結 + エイリアシング）を修正** → 安全指標の再計測（衝突数は増える方向に変わりうる）。
3. **C-2 と M-5（method/モデルパス整合、派生YAML統一）** — 今後の実行ミスによるデータ汚染を構造的に防止。
4. **M-1〜M-3（時刻アンカー群）** — C-1 修正後に staleness 補正を入れ、ADE/FDE/NLL を標準プロトコル（staleness=0 起点のみ）で再評価。
5. **M-16/M-17 の判別テスト追加** — 上記修正の回帰防止とセットで。
6. M-7（d=0格子）、M-10（復帰閾値）、M-8/M-9（Frenet規約・特異点）は実験への影響を見つつ。
7. 残り minor は次回キャンペーン前にまとめて。

---
レビュー実施: Claude Code マルチエージェントワークフロー（エージェント179体、レビュー17系統、敵対的検証延べ約150票）。指摘原文・検証票の全文は `/private/tmp/claude-501/-Users-mnhrk-Research-integrated-path-planning/d92560f8-6741-4b4d-957e-35d173715e75/tasks/wiy11cndd.output` に保存。
