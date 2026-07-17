# RQ3 REPORT: 実データ接地閉ループ（Closing the Loop の歩行者版）

生成: `examples/make_rq3_report.py`（全 verdict は純関数、prose は表から機械生成）。入力: `outputs/rq3_realloop/all_runs.csv`（`run_rq3_realloop.py --report-only` でキャッシュから再構築可能）。

判定しきい値: ALPHA=0.05（tri-state の有意性ゲート）・主指標=min_dist_m（同時刻 ego-歩行者中心間距離の最小値）・対単位=encounter（seed は encounter 内平均で先に潰す＝擬似反復の排除）。

## 0. 実験構成と器具開示

プランナ駆動 ego を録画 CITR 遭遇ジオメトリに接地: 参照経路=録画 ego 軌道のスプライン（0.5 m 間引き）・目標速度=録画 ego 速度中央値・初期状態=録画開始フレーム・total_time=録画窓長（timeout=censoring）。歩行者アーム:

| arm | kind | sigma | v0 | speed_regime |
|---|---|---|---|---|
| replay | replay | - | - | replay |
| calib | sfm | 1.168 | 1.712 | median_cruise |
| avec | sfm | 0.7 | 3.5 | median_cruise |
| norep | sfm | 1 | 0 | median_cruise |
| calib13x | sfm | 1.168 | 1.712 | initial_13x |

- 固定器具: scenario_01.yaml の verified フェイルセーフ/エンベロープ/プランナ定数（掃引はスコープ外）・SGAN/LSTM チェックポイント zara1_12_model.pt・衝突判定は同時刻位置のみ。
- 幾何: ego_radius=1.0 m / ped_radius=0.30 m（RQ2 較正整合）。実車寸法との差は制約 (limitation) 参照。
- 観測ウォームアップ: 全アームでフレーム0速度の等速バックキャスト（warmup_source=backcast、窓前実録画は ego NaN/在席非保証のため不使用）＝ t=0 の観測履歴と予測はアーム間で同一、差分は純粋にアーム動力学由来。

## 1. ランと打ち切りの census

- ラン総数 3120（アーム: avec, calib, calib13x, norep, replay）・遭遇 26/26（適格 26/26）。
- goal 到達 683/3120・timeout 打ち切り（censored） 2143/3120・衝突ラン 294/3120。完了時間の対比較は両アーム goal 到達の遭遇のみで行う（censoring 対処）。

| ped_arm | n_runs | censored_frac | goal_frac | progress_mean | ego_dev_mean_m | ego_dev_max_m |
|---|---|---|---|---|---|---|
| avec | 572 | 0.79 | 0.21 | 0.502 | 4.33 | 11.72 |
| calib | 702 | 0.711 | 0.231 | 0.513 | 4.18 | 11.38 |
| calib13x | 572 | 0.692 | 0.213 | 0.504 | 4.131 | 10.83 |
| norep | 572 | 0.703 | 0.208 | 0.499 | 4.239 | 11.55 |
| replay | 702 | 0.561 | 0.228 | 0.532 | 3.485 | 9.799 |

**打ち切りの帰結（review M2）**: プランナ ego は録画ドライバより保守的で、録画窓内に録画終端へ到達しないランが多数（progress 平均 0.51・録画 ego との時刻整合偏差 平均 4.05 m）。したがって V1/V3 の対比較は「録画窓で切り詰めた曝露・録画経路から乖離し得る ego」の下での測定であり、遭遇後半の相互作用は部分的にしか観測されない。対内では両アームが同一の打ち切り規則・同一の録画窓を共有するため比較自体は保存されるが、絶対値（衝突率・min_dist）は完走条件下の値ではない点に注意。

## 2. V1: 反応性交絡の直接測定（SFM アーム vs replay・対応差）

Δ = SFM アーム − replay（正= SFM 歩行者が譲るぶん ego の余裕が水増しされる方向）。canonical family = `rq3_v1_reactivity`（calib の全 pred×plan、BH-FDR は ledger 参照）。

| ped_arm | pred | plan | n_pairs | n_arm_gt_replay | n_arm_lt_replay | mean_delta_m | sign_p | wilcoxon_p | arm_coll_encs | replay_coll_encs | coll_arm_only | coll_replay_only | mcnemar_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calib | cv | single | 26 | 13 | 13 | -0.0494 | 1 | 0.9007 | 1 | 7 | 0 | 6 | 0.03125 |
| calib | cv | robust | 26 | 13 | 13 | -0.0494 | 1 | 0.9007 | 1 | 7 | 0 | 6 | 0.03125 |
| calib | lstm | single | 26 | 16 | 10 | 0.2719 | 0.3269 | 0.07957 | 1 | 8 | 0 | 7 | 0.01562 |
| calib | lstm | robust | 26 | 18 | 8 | 0.3715 | 0.07552 | 0.01401 | 1 | 3 | 0 | 2 | 0.5 |
| calib | sgan | single | 26 | 14 | 12 | 0.2446 | 0.845 | 0.1427 | 2 | 8 | 0 | 6 | 0.03125 |
| calib | sgan | robust | 26 | 14 | 12 | 0.1732 | 0.845 | 0.5317 | 2 | 4 | 0 | 2 | 0.5 |
| avec | cv | single | 26 | 13 | 13 | -0.0389 | 1 | 0.7644 | 0 | 7 | 0 | 7 | 0.01562 |
| avec | cv | robust | 26 | 13 | 13 | -0.0389 | 1 | 0.7644 | 0 | 7 | 0 | 7 | 0.01562 |
| avec | lstm | single | 26 | 16 | 10 | 0.2733 | 0.3269 | 0.06303 | 0 | 8 | 0 | 8 | 0.007812 |
| avec | lstm | robust | 26 | 17 | 9 | 0.3474 | 0.1686 | 0.02915 | 0 | 3 | 0 | 3 | 0.25 |
| avec | sgan | single | 26 | 15 | 11 | 0.2512 | 0.5572 | 0.07957 | 0 | 8 | 0 | 8 | 0.007812 |
| avec | sgan | robust | 26 | 14 | 12 | 0.1661 | 0.845 | 0.4082 | 0 | 4 | 0 | 4 | 0.125 |
| norep | cv | single | 26 | 12 | 14 | -0.1998 | 0.845 | 0.3941 | 2 | 7 | 0 | 5 | 0.0625 |
| norep | cv | robust | 26 | 12 | 14 | -0.1998 | 0.845 | 0.3941 | 2 | 7 | 0 | 5 | 0.0625 |
| norep | lstm | single | 26 | 13 | 13 | 0.1697 | 1 | 0.2914 | 4 | 8 | 0 | 4 | 0.125 |
| norep | lstm | robust | 26 | 15 | 11 | 0.2906 | 0.5572 | 0.1048 | 2 | 3 | 0 | 1 | 1 |
| norep | sgan | single | 26 | 12 | 14 | 0.112 | 0.845 | 0.5995 | 4 | 8 | 0 | 4 | 0.125 |
| norep | sgan | robust | 26 | 9 | 17 | 0.0856 | 0.1686 | 0.9403 | 2 | 4 | 0 | 2 | 0.5 |
| calib13x | cv | single | 26 | 15 | 11 | 0.0541 | 0.5572 | 0.5995 | 2 | 7 | 0 | 5 | 0.0625 |
| calib13x | cv | robust | 26 | 15 | 11 | 0.0541 | 0.5572 | 0.5995 | 2 | 7 | 0 | 5 | 0.0625 |
| calib13x | lstm | single | 26 | 15 | 11 | 0.2065 | 0.5572 | 0.4227 | 4 | 8 | 0 | 4 | 0.125 |
| calib13x | lstm | robust | 26 | 16 | 10 | 0.3355 | 0.3269 | 0.1499 | 3 | 3 | 0 | 0 | nan |
| calib13x | sgan | single | 26 | 16 | 10 | 0.3658 | 0.3269 | 0.1105 | 4 | 8 | 0 | 4 | 0.125 |
| calib13x | sgan | robust | 26 | 14 | 12 | 0.1325 | 0.845 | 0.7265 | 2 | 4 | 0 | 2 | 0.5 |

- `calib` vs replay: 6 セル中 4 セルで min-separation が replay より大（Δ範囲 [-0.049, +0.371] m）、符号検定 p<0.05 は 0/6 セル。衝突不一致の対（セル横断延べ・同一遭遇の重複含む）は replay 側のみ 29 件 vs calib 側のみ 0 件。McNemar family BH（m=5・cv/robust 縮退除外）: 0/5 件が q<0.05 で生存・最小 q=0.0521。
- `avec` vs replay: 6 セル中 4 セルで min-separation が replay より大（Δ範囲 [-0.039, +0.347] m）、符号検定 p<0.05 は 0/6 セル。衝突不一致の対（セル横断延べ・同一遭遇の重複含む）は replay 側のみ 37 件 vs avec 側のみ 0 件。McNemar family BH（m=5・cv/robust 縮退除外）: 3/5 件が q<0.05 で生存・最小 q=0.0195。
- `norep` vs replay: 6 セル中 4 セルで min-separation が replay より大（Δ範囲 [-0.200, +0.291] m）、符号検定 p<0.05 は 0/6 セル。衝突不一致の対（セル横断延べ・同一遭遇の重複含む）は replay 側のみ 21 件 vs norep 側のみ 0 件。McNemar family BH（m=5・cv/robust 縮退除外）: 0/5 件が q<0.05 で生存・最小 q=0.208。
- `calib13x` vs replay: 6 セル中 6 セルで min-separation が replay より大（Δ範囲 [+0.054, +0.366] m）、符号検定 p<0.05 は 0/6 セル。衝突不一致の対（セル横断延べ・同一遭遇の重複含む）は replay 側のみ 20 件 vs calib13x 側のみ 0 件。McNemar family BH（m=5・cv/robust 縮退除外）: 0/5 件が q<0.05 で生存・最小 q=0.167。
- 全 SFM アーム・全セルで `coll_arm_only`=0（方向は一貫して「SFM で衝突が消える」側）だが、canonical アーム（calib）の McNemar family は BH を生存せず（最小 q=0.0521）＝方向的示唆にとどまり確証的主張はしない（制御アームの family 生存状況は各行のとおりで、確証枠には算入しない）。

## 3. V2: ベンチマーク判定の保存性（tri-state）

replay アームの判定が headline 候補・SFM アームは感度幅。有意性ゲート（Wilcoxon p<0.05）を通らない不一致は『検出力限界』であり反転とは読まない（RQ1b の tristate 規律）。

| verdict_kind | pred_or_plan | ped_arm | value | significant | detail | gap_p |
|---|---|---|---|---|---|---|
| robust_gain_direction | cv | ALL | 全アームで縮退（Δ=0・robust ≡ single） |  | replay:0(縮退) calib:0(縮退) avec:0(縮退) norep:0(縮退) calib13x:0(縮退) |  |
| robust_gain_direction | lstm | ALL | 全アームで不変（反応性仮定に頑健） |  | replay:+ calib:+ avec:+ norep:+ calib13x:+ |  |
| robust_gain_direction | sgan | ALL | 全アームで不変（反応性仮定に頑健） |  | replay:+ calib:+ avec:+ norep:+ calib13x:+ |  |
| most_dangerous_predictor | single | ALL | 全アームで方向不変だが全アーム非有意（頑健・ただし検出力は限定的） |  | replay:cv<sgan<lstm(n.s.) calib:cv<sgan<lstm(sig) avec:cv<sgan<lstm(sig) norep:cv<sgan<lstm(sig) calib13x:cv<lstm<sgan(sig) |  |
| predictor_ranking | single | replay | cv<sgan<lstm | False | cv=2.956 sgan=3.077 lstm=3.209 | 0.135 |
| predictor_ranking | single | calib | cv<sgan<lstm | True | cv=2.907 sgan=3.322 lstm=3.480 | 0.007838 |
| predictor_ranking | single | avec | cv<sgan<lstm | True | cv=2.917 sgan=3.328 lstm=3.482 | 0.007066 |
| predictor_ranking | single | norep | cv<sgan<lstm | True | cv=2.756 sgan=3.189 lstm=3.378 | 0.00671 |
| predictor_ranking | single | calib13x | cv<lstm<sgan | True | cv=3.010 lstm=3.415 sgan=3.443 | 0.002354 |
| most_dangerous_predictor | robust | ALL | 全アームで不変（反応性仮定に頑健） |  | replay:cv<sgan<lstm(sig) calib:cv<sgan<lstm(sig) avec:cv<sgan<lstm(sig) norep:cv<sgan<lstm(sig) calib13x:cv<sgan<lstm(sig) |  |
| predictor_ranking | robust | replay | cv<sgan<lstm | True | cv=2.956 sgan=3.858 lstm=4.071 | 0.0006739 |
| predictor_ranking | robust | calib | cv<sgan<lstm | True | cv=2.907 sgan=4.031 lstm=4.442 | 0.0001195 |
| predictor_ranking | robust | avec | cv<sgan<lstm | True | cv=2.917 sgan=4.024 lstm=4.418 | 0.0001195 |
| predictor_ranking | robust | norep | cv<sgan<lstm | True | cv=2.756 sgan=3.944 lstm=4.361 | 0.000127 |
| predictor_ranking | robust | calib13x | cv<sgan<lstm | True | cv=3.010 sgan=3.991 lstm=4.406 | 0.0001623 |

## 4. V3: robust 利得の実ジオメトリ検証（replay 参照点つき・auxiliary）

Δ = robust − single（true-single draw、review F4 対応）。replay アーム＝『実際の歩行者がした行動』の下での利得。全アーム分を併記。

| ped_arm | pred | n_pairs | n_robust_gt_single | mean_delta_m | sign_p | wilcoxon_p | single_coll_encs | robust_coll_encs | n_time_pairs | mean_time_cost_s |
|---|---|---|---|---|---|---|---|---|---|---|
| replay | cv | 26 | 0 | 0 | nan | nan | 7 | 7 | 8 | 0 |
| replay | lstm | 26 | 20 | 0.8623 | 0.0004883 | 0.000127 | 8 | 3 | 3 | 0 |
| replay | sgan | 26 | 20 | 0.7808 | 0.0004883 | 0.0002067 | 8 | 4 | 4 | 0.065 |
| calib | cv | 26 | 0 | 0 | nan | nan | 1 | 1 | 8 | 0 |
| calib | lstm | 26 | 22 | 0.9619 | 3.588e-05 | 0.0001289 | 1 | 1 | 3 | 0 |
| calib | sgan | 26 | 19 | 0.7094 | 0.0008554 | 0.000177 | 2 | 2 | 4 | 0.005 |
| avec | cv | 26 | 0 | 0 | nan | nan | 0 | 0 | 8 | 0 |
| avec | lstm | 26 | 22 | 0.9364 | 3.588e-05 | 0.0001289 | 0 | 0 | 3 | 0 |
| avec | sgan | 26 | 20 | 0.6956 | 0.0001211 | 0.0001363 | 0 | 0 | 4 | 0.005 |
| norep | cv | 26 | 0 | 0 | nan | nan | 2 | 2 | 8 | 0 |
| norep | lstm | 26 | 20 | 0.9832 | 0.001544 | 0.0001819 | 4 | 2 | 3 | 0 |
| norep | sgan | 26 | 21 | 0.7543 | 1.097e-05 | 9.149e-05 | 4 | 2 | 4 | 0.01 |
| calib13x | cv | 26 | 0 | 0 | nan | nan | 2 | 2 | 8 | 0 |
| calib13x | lstm | 26 | 19 | 0.9913 | 0.01463 | 0.0004458 | 4 | 3 | 5 | 0.076 |
| calib13x | sgan | 26 | 20 | 0.5475 | 0.0001211 | 0.0002013 | 4 | 2 | 4 | 0.01 |

### 4.1 medoid 参考対比（review F4 の実測差・記述のみ）

| ped_arm | pred | n_pairs | mean_medoid_minus_draw_m | max_abs_diff_m | medoid_coll_encs | draw_coll_encs |
|---|---|---|---|---|---|---|
| calib | sgan | 26 | -0.1967 | 1.165 | 2 | 2 |
| replay | sgan | 26 | -0.1666 | 1.02 | 8 | 8 |

medoid（predict_single_best 既定＝分散抑制代表値）と true-single draw は同一 seed でも異なる閉ループ軌道を生む（AVEC/RQ1b の single 条件は medoid 相手の測定だったという F4 の定量的裏付け）。ledger 検定は張らない（新仮説ではなく計装の開示）。

## 5. 制約 (limitations)

- **円板近似**: ego_radius=1.0 m は実車の外接円近似で、録画済み実遭遇にも最接近 1.28 m（< 1.30 m 判定半径和・1 遭遇）の事例がある＝『衝突』は保守的な器具定義であり実接触ではない。
- **打ち切り窓上の測定**: §1 の通り、録画窓長を total_time とする設計は censoring を生む（対内で共有・開示済み）。
- **replay の非反応性**: replay 歩行者はプランナ ego に反応しない（Closing the Loop の log-replay と同じ設計選択を参照点として利用）。ego が録画から乖離した後の replay 軌道は反実仮想として読めない。バイアスの向きは既知: replay は歩行者側回避を過小評価（衝突を過大に）、SFM は過大評価（衝突を過小に）しうるため、両アームは実世界の挙動を挟み込む参照枠 (bracketing) として読む。
- **バックキャスト観測履歴**: 直線 8 フレームは SGAN の学習分布と異なるが、全アーム同一条件のため対比較は保存される。
- **フェイルセーフ定数**: S1-S3 で手調整された固定器具（未掃引）。
- **medoid 参考条件**: sgan × {replay, calib} のみ（計算量制御・ユーザー承認 2026-07-03）。
- **統計単位**: encounter（n<=26）。同一 encounter を共有するpred×plan セル間の検定は独立でない（family 補正は ledger の canonical/auxiliary 区分で処理）。

## 6. ledger 登載

`headline_tests.json`（namespace rq3.*）: canonical = `rq3_v1_reactivity`（6 検定）、auxiliary = `rq3_v1_reactivity_ctrl` / `rq3_v1_collision_mcnemar`（calib）/ `rq3_v1_collision_mcnemar_ctrl`（制御アーム） / `rq3_v3_robust_real` / `rq3_v3_robust_real_ctrl`（非 replay アームの V3 符号検定・cv は p 未定義で縮退開示） / `rq3_v3_robust_wilcoxon`（V3 全アームの Wilcoxon 併記） / `rq3_v2_ranking_gates`（V2 最危険予測器判定の有意性ゲート）。cv/robust の McNemar はビット同一縮退のため未登載（cv/single の note 参照）。V1 の Wilcoxon 併記は符号検定レコードの passthrough フィールドに記録し別仮説として数えない（canonical 6 検定設計）。`examples/make_multiplicity_ledger.py` の再実行で台帳へ自動編入。

**台帳への意図差分の開示**（静的記録: 2026-07-03 の台帳再生成時に機械検証した監査結果の転記であり、本 REPORT の再生成では再計算されない）: canonical family の追加により study-wide 補正（overall_* 列）は全既存行で再計算される（canonical 21→27 検定）。既存行の within-family 列は全行バイト不変を機械検証済み。overall 層の判定反転は 4 件・全て True→False（保守化方向・新規の主張は発生しない）: `rq1b.rand.fisher.avec.scenario_02`（既知の境界セル、within-family q=0.023 生存は不変）、`rq1b.rand.fisher_aggregate.avec.lstm`・`rq2.dut.multivehicle.closest_ks.avec_default`・`rq2cap.loco.closedloop.closest_sign.no_repulsion`（いずれも auxiliary 層）。

**追加登載の開示（静的記録: 2026-07-16、修論横断レビュー M2a 対応）**: 本文（修論 表8.2/8.3）に掲載していた p 値のうち台帳外だったもの（非 replay アームの V3 符号検定・V3 全アーム Wilcoxon・V2 判定ゲート Wilcoxon）を auxiliary family 3 つ（計 32 行・うち cv 縮退 4 行は p 未定義）として末尾追加。auxiliary は canonical の study-wide 補正プールに入らないため、canonical 27 行・研究横断生存 3 件・既存全行の within-family 列は不変（追加後の台帳再生成で機械検証済み。auxiliary 行の overall_* 列のみ aux プール内補正の再計算で変わるが、REPORT・修論とも非使用）。全新規 family は family 内 BH で `rq3_v2_ranking_gates` の replay/single ゲート（p=0.135・V2 の検出力限界開示と整合）を除き生存＝既存結論への影響なし。方向つき開示の規律に従い1点付記する: 非使用の auxiliary プール内 overall 層では、有限 p の拡大（87→117）で既存行 `rq2.dut.multivehicle.closest_ks.avec_default` の判定が1件だけ 非有意→有意（q 0.0575→0.0428）へ動く。この検定は KS 診断（p 非主張・付録Bで overall 列を掲載しないと宣言済み）であり、どの主張にも使われない。上の 2026-07-03 静的記録にある同 ID の True→False は canonical 21→27 拡大時の overall 層の別事象である。
