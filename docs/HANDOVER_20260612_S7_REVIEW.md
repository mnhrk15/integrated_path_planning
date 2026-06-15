# 引き継ぎ S7: 第5世代改訂後の AVEC 論文の徹底レビュー（2026-06-12 作成）

前セッション（S6）で AVEC 論文を第5世代（`_g5a2`）結果へ全面改訂した。**本ミッションは改訂後論文の徹底レビュー（独立検証 + 明確な誤りの修正適用）**。S6 セッションの作業報告・本引き継ぎの数値も鵜呑みにせず、生データから再導出して検証すること。

## 0. リポジトリ状態と制約

- **論文**: `~/Research/AVEC_FullPaper`（main、**リモートなし**）。レビュー対象 = HEAD `059e355`。
  - `059e355` 本体改訂（main.tex / main_ja.tex / NUMERICAL_VERIFICATION.md / PDF）
  - `4b8c5cc` keywords 短縮・謝辞節番号化（ユーザー編集）+ **図4枚の再生成も同コミットに混入**（メッセージに明記済み）
  - それ以前（`90b3371`〜）は旧世代の論文。
- **実装**: `~/Research/integrated_path_planning`（dev、origin へ未 push）。**シミュレーションコードは凍結**（挙動変更は `*_g5a2` 全結果を無効化する。実験キャッシュはコードをキーに含まない）。`0c0b489` はプロットスクリプトのみ（安全）。プロット・ドキュメント変更のみ可。
- **push は行わない**（sim 側。論文側はリモート自体なし）。修正コミットは段階ごとに可。
- ビルド: EN `pdflatex main.tex` ×2 → 現状 **6ページ・Overfull 0・undefined 0。6ページは厳守**（修正後も維持・再確認）。JA `lualatex main_ja.tex` ×2 → 現状6ページだが **JA はページ制約なし**（ユーザーの内部確認用。EN の6頁圧縮をミラーして JA まで切り詰めた箇所があるため、不自然に簡潔な箇所は読みやすく展開してよい。ただし主張・数値の追加・変更は不可＝EN と同期維持）。
- 環境: 再計算は `~/Research/integrated_path_planning` で `.venv/bin/python`（pandas / scipy / numpy 利用可）。

## 1. S6 改訂の要約（= レビュー対象の構造）

**ユーザー決定事項（再交渉しない）:**
1. `tab:benchmark` = **rand（v0 ランダム化 GT）** ベース。CV も n=20、衝突ラン数 `Coll.` 列追加。決定論 GT（comfort、123ラン衝突ゼロ）は本文言及に格下げ。
2. NLL 不掲載。評価は単円構成（multi_circle / footprint は §5.5 limitations の定性記述のみ、数値なし）。
3. 旧 §5.3(ii) の ε=0.1 記述は削除（`_g5a2` に条件が存在しない）。`tab:domain` は本文1文に畳んで削除（6頁化のため）。
4. §5.1 は「lateral artifact 解析」→「右折の譲り挙動解析」へ転換（第5世代では旧 artifact が再現しない: 右折後 |d|≤0.3 m・ψ 滑らか・4.1 s の譲り停止）。
5. 衝突論拠は rand 側（S1: CV 4/20 vs 学習系 0/20）に一本化。安全構成の推奨 = robust(ε=0)（S2/S3 で MinDist・Time 両軸支配、§5.3 = margin_control 実験として全面書き換え、`tab:da` は SGAN/LSTM × ΔMinDist/ΔTime の4列）。

**主な書き換え箇所**: Abstract / 貢献 (i)–(iv) / §2.3+§2.5（Gen-5 アーキテクチャ: 安全速度エンベロープ・全域速度グリッド・ブレーキ候補ラダー・停止指令・予防 CAUTION・速度対応回復ゲート・適応緊急減速・曲率非緩和）/ tab:params（現 YAML 同期）/ §3.1（SFM 身体半径 0.3 vs 判定半径 0.2 の透明化）/ §3.2（S1 ホライズン 30 s・S3 目標 5.0 m/s・S2 エンベロープ無効の根拠）/ §3.4（v0 ランダム化 N(0, 0.19)・下限 0.3・180ラン、決定論補助 123ラン、CV S1 の ADE n=18 注記）/ §4 全節 / §5.2（「ADE 3–4×」撤回 → 序列逆転を新しい兆候に）/ §5.4（「well within」撤回 → plan 100 ms・64% 超過・robust20 3.67×）/ §5.5（単円限定・同時刻判定の設計根拠・t±Δ 将来課題）/ §6。**6頁化のため多数の圧縮あり**（コスト式の1行化・§2.1 小見出し削除・図幅 0.72/0.70・キャプション短縮・文単位の削除）。

## 2. 数値の正（検証の起点）

- **`NUMERICAL_VERIFICATION.md` 第9段階** = 論文中の全数値 → 出典パスの台帳。ただし**台帳自体の転記ミスも疑い、必ず生データから再計算して三者照合**（本文 ⇔ 台帳 ⇔ 生データ）すること。
- 生データ（実装リポジトリ）: `output/benchmark_rand_s{1,2,3}_g5a2/all_runs.csv` / `output/comfort_s{1,2,3}_g5a2/` / `output/exp_margin_control_g5a2/{summary,welch_tests,welch_vs_baseline,all_runs}.csv` / `output/exp_proc_planning_g5a2/results.json` / `output/verify_g5a_s8/s{1,2,3}_{cv,lstm,sgan}/trajectory.npz`（seed1、図の出典）。
- 上位レポート: `docs/FINAL_BENCHMARK_REPORT.md` §8、`docs/HANDOVER_20260612_S6.md` §1（ともに二次情報として扱う）。
- 再計算の要点: 平均±標準偏差は pandas groupby、検定は `scipy.stats.ttest_ind(equal_var=False)`、**rand S1 の CV ADE は NaN 2件（seed 13/15、衝突早期終了）→ `nan_policy="omit"`・n=18**。npz の d(t) は `examples/plot_lateral_analysis.py` と同一の Frenet 変換（最近傍点+外積符号）、加速度は ego_v の差分/0.1 s、停止判定は v<0.05 が 0.5 s 以上。
- 図の再生成（必要時）: `examples/plot_simulation_figs.py --inputs output/verify_g5a_s8/s1_sgan output/verify_g5a_s8/s2_sgan output/verify_g5a_s8/s3_sgan` / `examples/plot_lateral_analysis.py --input output/verify_g5a_s8/s3_sgan`（出力先は `~/Research/AVEC_FullPaper/figs/`）。

## 3. レビュー観点（番号順 = 優先度順）

1. **[最優先] 数値の全件突合**: 本文・表・キャプションの全数値（tab:benchmark 全セル、§4 の p 値、tab:da と §5.3 のデルタ・p、§5.4 計時、§5.1 挙動値、tab:params、Abstract/結論中の数値）を生データから独立再計算して照合。太字付与（列内最良・同値は無太字）の正誤も全列確認。
2. **[最優先] 実装記述の正確性**: §2.3/§2.5/tab:params の全主張をコードと照合 — `src/core/state_machine.py`（`_envelope_speed` の式、予防トリガ c_trig+τv、回復ゲート max(gate, trigger+τv)、EMERGENCY 回復、適応減速 v²/(2·max(c−0.2, 0.05)) と clip 範囲・非有限時フォールバック、accel ×1.5/×3.0、曲率非緩和）、`src/planning/frenet_planner.py`（終端速度グリッド target→0、ブレーキラダー BRAKE_*、`max_stop_distance` フィルタ）、`src/config/__init__.py`（envelope デフォルト 0=無効、min_t/max_t=4/5、d_t_s、sfm_v0_std/min）、`scenarios/scenario_0{1,2,3}.yaml`。
3. **統計主張の境界**: §3.4 の Holm 主張（ファミリ=変位誤差序列、最小 p が本当に 10⁻⁸ 未満か）/ §4.1「快適性 all pairwise p>0.07」（実最小 0.076 のはず）/ §4.4「MinDist pairwise p≥0.24」/ 生存バイアス・右側打切り（S1 timeout: single 4 vs robust 5）の記述が誇張なしか / CV ペア検定の扱い（rand では CV も確率的＝検定対象、決定論側では対象外）の一貫性。
4. **論理整合**: 「決定論 GT で CV 最良 ADE」⇔「rand で CV のみ S1 衝突」⇔「ADE は安全性の代理にならない」⇔「robust(ε=0) 推奨」の筋に飛躍・過剰一般化がないか。§5.3 の inflation 非支配の言い回し（S2 では λ≥1.35 が MinDist 単独では robust を上回る — 「両軸では並べない」が正確な主張）が welch_tests.csv / summary.csv の実数と整合するか。
5. **図⇔本文⇔キャプション**: 図は決定論 GT・seed1（rand ではない）→ キャプションの「deterministic ground truth」明記を確認。S3 の 4.1 s 停止が図注釈・§4.4・§5.1 で一致。fig:lateral のパネル構成（v/d/ψ）とキャプション一致。
6. **EN↔JA 同期**: 節単位で主張・数値の一致（JA の展開は可、内容の乖離は不可）。
7. **6頁圧縮の副作用**: 文削除による論理の飛び・英文の文法/自然さ（多数の圧縮編集が入っている）・`tab:da` の `\textbf{$-1.42$}`（math を含む太字）が PDF で実際に太字に見えるか目視確認。
8. **相互参照・引用**: \ref/\label 全件解決、bib 13件の使用状況（bib11/bib01 は結論のみの引用になった）、表番号（tab:domain 削除で繰り上がり）。
9. **旧値残存の再 grep**: パターンを独立に設計し直して再確認（例: 旧目標速度・旧ホライズン・旧計時・旧 ADE 値・artifact 語彙）。
10. **査読者視点の自由レビュー**: 新規性主張の強度、タイトル「Evaluation of Deep Learning-Based Trajectory Prediction…」と新しい物語（CV が ADE 最良）の整合、Abstract と本文の一致、その他。

## 4. 既知の注意点・落とし穴

- **S6 引き継ぎ文書には誤記が確認済み**: §3-16/§4 の「max accel 全シナリオ 2.0」は誤りで S2 は 1.5（YAML が正。論文は 2.0/1.5/2.0 で記載済み）。S6 文書の他の記述も一次情報（YAML/コード/CSV）と突合すること。
- `exp_margin_control_g5a2/REPORT.md` 冒頭の Sanity「per-seed 照合 FAIL」は旧世代 PoC との照合であり期待どおり（慌てない）。
- 第4世代（`*_final`）の数値は表・本文の定量値として混ぜない（limitations の定性記述のみ可）。世代間絶対値は非比較。
- `output/scenario_0{1,2,3}/` は旧世代の遺物（図の出典ではない）。図の出典は `verify_g5a_s8`。
- コードに触る場合の禁止事項（S6 文書 §7）: ゼロラグ clearance 再適用禁止 / 前方 clearance 必須 / S2 エンベロープ無効維持 / 同時刻衝突判定維持 / 曲率非緩和。プロット・ドキュメントのみ安全。

## 5. 成果物

1. **重大度付きレビュー報告**（CRITICAL / MAJOR / MINOR、各指摘に根拠と出典パス・行番号）。
2. **明確な誤りは修正を適用**（数値転記ミス・文法・EN↔JA 不整合・参照切れ・太字誤り）。修正後は EN 6ページ・Overfull 0・undefined 0 を再ビルドで確認し、PDF も含めてコミット（リポジトリ慣行）。
3. **判断を要する指摘**（主張のトーン・構成・追加削除）は修正せずユーザーに選択肢として提示。
4. 数値に触れた場合は `NUMERICAL_VERIFICATION.md` 第9段階も更新。push はしない。
