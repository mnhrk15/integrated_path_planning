# RQ スイート横断 多重比較 ledger

family-wise / FDR 補正（alpha=0.05）。BH-FDR を primary、Holm-Bonferroni を conservative sensitivity として併記。NaN（空 arm の Fisher・空プールの KS）は仮説ではないので family size に数えない。

> **RQ1a は検定を持たない**（開ループ ADE/FDE/NLL は点推定・有意性検定なし＝意図的、review M1/M10）。実行していない検定は補正対象にならない＝「やらなかった検定で p-hack できない」という多重性衛生そのもの。

## 収集した sidecar

- `outputs/rq1b/headline_tests.json` — source=RQ1b-rand, tests=30
- `outputs/rq2_dut_validation/headline_tests_dut_multivehicle.json` — source=RQ2-DUT-multivehicle, tests=3
- `outputs/rq2_dut_validation/headline_tests_dut_single.json` — source=RQ2-DUT-single, tests=3
- `outputs/rq2_evaluation/headline_tests_loco.json` — source=RQ2-loco, tests=6
- `outputs/rq2_evaluation/headline_tests_loso.json` — source=RQ2-loso, tests=6
- `outputs/rq2_instrument_audit/cap/headline_tests_cap_closedloop_loco.json` — source=RQ2-cap-closedloop-loco, tests=3
- `outputs/rq2_instrument_audit/cap/headline_tests_cap_median_loco.json` — source=RQ2-cap-median-loco, tests=3
- `outputs/rq2_instrument_audit/cap/headline_tests_cap_uncapped_loco.json` — source=RQ2-cap-uncapped-loco, tests=3
- `outputs/rq2_instrument_audit/distmatch/headline_tests_dm_pure_loco.json` — source=RQ2-distmatch-pure-loco, tests=3
- `outputs/rq2_instrument_audit/distmatch/headline_tests_dm_w0.5_loco.json` — source=RQ2-distmatch-w0.5-loco, tests=3
- `outputs/rq2_instrument_audit/distmatch/headline_tests_dm_w1_id8_loco.json` — source=RQ2-distmatch-w1_id8-loco, tests=3
- `outputs/rq2_instrument_audit/distmatch/headline_tests_dm_w1_loco.json` — source=RQ2-distmatch-w1-loco, tests=3

## 補正結果（canonical 研究横断 family）

| test_id | family | p_value | family_size | family_bh_q | family_holm_p | overall_bh_q | overall_holm_p |
|---|---|---|---|---|---|---|---|
| rq1b.rand.fisher.avec.scenario_01 | rq1b_claim2_fisher | 0.6000 | 18 | 0.7714 | 1.0000 | 0.7412 | 1.0000 |
| rq1b.rand.fisher.avec.scenario_02 | rq1b_claim2_fisher | 0.0078 | 18 | 0.1404 | 0.1404 | 0.0410 | 0.1404 |
| rq1b.rand.fisher.avec.scenario_03 | rq1b_claim2_fisher | 0.2116 | 18 | 0.7714 | 1.0000 | 0.5772 | 1.0000 |
| rq1b.rand.fisher.calib.scenario_01 | rq1b_claim2_fisher | 0.3576 | 18 | 0.7714 | 1.0000 | 0.7412 | 1.0000 |
| rq1b.rand.fisher.calib.scenario_02 | rq1b_claim2_fisher | 1.0000 | 18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| rq1b.rand.fisher.calib.scenario_03 | rq1b_claim2_fisher | 0.2199 | 18 | 0.7714 | 1.0000 | 0.5772 | 1.0000 |
| rq1b.rand.fisher.calib_hi.scenario_01 | rq1b_claim2_fisher | 0.3551 | 18 | 0.7714 | 1.0000 | 0.7412 | 1.0000 |
| rq1b.rand.fisher.calib_hi.scenario_02 | rq1b_claim2_fisher | 1.0000 | 18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| rq1b.rand.fisher.calib_hi.scenario_03 | rq1b_claim2_fisher | 0.4716 | 18 | 0.7714 | 1.0000 | 0.7412 | 1.0000 |
| rq1b.rand.fisher.calib_lo.scenario_01 | rq1b_claim2_fisher | 0.6000 | 18 | 0.7714 | 1.0000 | 0.7412 | 1.0000 |
| rq1b.rand.fisher.calib_lo.scenario_02 | rq1b_claim2_fisher | 0.6000 | 18 | 0.7714 | 1.0000 | 0.7412 | 1.0000 |
| rq1b.rand.fisher.calib_lo.scenario_03 | rq1b_claim2_fisher | 0.0673 | 18 | 0.4038 | 1.0000 | 0.2355 | 1.0000 |
| rq1b.rand.fisher.calib_loso_smin.scenario_01 | rq1b_claim2_fisher | 0.6000 | 18 | 0.7714 | 1.0000 | 0.7412 | 1.0000 |
| rq1b.rand.fisher.calib_loso_smin.scenario_02 | rq1b_claim2_fisher | 0.6000 | 18 | 0.7714 | 1.0000 | 0.7412 | 1.0000 |
| rq1b.rand.fisher.calib_loso_smin.scenario_03 | rq1b_claim2_fisher | 0.0673 | 18 | 0.4038 | 1.0000 | 0.2355 | 1.0000 |
| rq1b.rand.fisher.calib_loso_vmax.scenario_01 | rq1b_claim2_fisher | 1.0000 | 18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| rq1b.rand.fisher.calib_loso_vmax.scenario_02 | rq1b_claim2_fisher | 1.0000 | 18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| rq1b.rand.fisher.calib_loso_vmax.scenario_03 | rq1b_claim2_fisher | 0.4716 | 18 | 0.7714 | 1.0000 | 0.7412 | 1.0000 |
| rq2.loco.closest_sign.calibrated | rq2_fidelity_paired_loco | 0.0000 | 3 | 0.0000 | 0.0000 | 0.0001 | 0.0002 |
| rq2.loco.closest_wilcoxon.calibrated | rq2_fidelity_paired_loco | 0.0000 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rq2.loco.closest_sign.no_repulsion | rq2_fidelity_paired_loco | 0.0000 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- 研究横断 family size（overall）: **21** 検定
- overall BH-FDR で生存（q<0.05）: **4** / Holm で生存: **3**

> **overall の読み方**: overall は RQ2 忠実度と RQ1b 計画安全という*異なる問い*を1 family に束ねた最保守の境界（cross-suite 過剰補正）。適切な評価単位は各 family 内補正（上表 `family_bh_q`）と下の RQ1b family 定義感度。overall は『最悪でもこの程度』の sanity 上限として読む。

## 付録: auxiliary（補助・overall から除外）

LOSO（LOCO と同じ忠実度の問いの再分割＝二重計上回避）、診断専用 pooled KS（対標本に独立2標本検定＝p は読まない、review F5）、RQ1b 集計 Fisher（M8 noise-grade）、DUT 汎化 KS（pseudo-replication caveat）。canonical 仮説ではないため研究横断 overall には含めず、開示と family 内補正のみ示す（誤読防止のため overall_* 列は本表から省く＝auxiliary プール内の overall 補正値は研究横断値ではない）。

| test_id | family | p_value | family_size | family_bh_q | family_holm_p |
|---|---|---|---|---|---|
| rq1b.rand.fisher_aggregate.avec.cv | rq1b_claim2_fisher_aggregate | 0.5000 | 12 | 0.6000 | 1.0000 |
| rq1b.rand.fisher_aggregate.avec.lstm | rq1b_claim2_fisher_aggregate | 0.0287 | 12 | 0.3444 | 0.3444 |
| rq1b.rand.fisher_aggregate.calib.cv | rq1b_claim2_fisher_aggregate | 0.5000 | 12 | 0.6000 | 1.0000 |
| rq1b.rand.fisher_aggregate.calib.lstm | rq1b_claim2_fisher_aggregate | 0.1822 | 12 | 0.4916 | 1.0000 |
| rq1b.rand.fisher_aggregate.calib_lo.cv | rq1b_claim2_fisher_aggregate | 0.2458 | 12 | 0.4916 | 1.0000 |
| rq1b.rand.fisher_aggregate.calib_lo.lstm | rq1b_claim2_fisher_aggregate | 0.1186 | 12 | 0.4744 | 1.0000 |
| rq1b.rand.fisher_aggregate.calib_hi.cv | rq1b_claim2_fisher_aggregate | 1.0000 | 12 | 1.0000 | 1.0000 |
| rq1b.rand.fisher_aggregate.calib_hi.lstm | rq1b_claim2_fisher_aggregate | 0.3060 | 12 | 0.5246 | 1.0000 |
| rq1b.rand.fisher_aggregate.calib_loso_vmax.cv | rq1b_claim2_fisher_aggregate | 1.0000 | 12 | 1.0000 | 1.0000 |
| rq1b.rand.fisher_aggregate.calib_loso_vmax.lstm | rq1b_claim2_fisher_aggregate | 0.5000 | 12 | 0.6000 | 1.0000 |
| rq1b.rand.fisher_aggregate.calib_loso_smin.cv | rq1b_claim2_fisher_aggregate | 0.1186 | 12 | 0.4744 | 1.0000 |
| rq1b.rand.fisher_aggregate.calib_loso_smin.lstm | rq1b_claim2_fisher_aggregate | 0.2458 | 12 | 0.4916 | 1.0000 |
| rq2.dut.multivehicle.closest_ks.calibrated | rq2_dut_fidelity_ks_multivehicle | 0.0133 | 3 | 0.0200 | 0.0266 |
| rq2.dut.multivehicle.closest_ks.avec_default | rq2_dut_fidelity_ks_multivehicle | 0.0238 | 3 | 0.0238 | 0.0266 |
| rq2.dut.multivehicle.closest_ks.no_repulsion | rq2_dut_fidelity_ks_multivehicle | 0.0019 | 3 | 0.0056 | 0.0056 |
| rq2.dut.single.closest_ks.calibrated | rq2_dut_fidelity_ks_single | 0.3517 | 3 | 0.3517 | 0.7034 |
| rq2.dut.single.closest_ks.avec_default | rq2_dut_fidelity_ks_single | 0.3517 | 3 | 0.3517 | 0.7034 |
| rq2.dut.single.closest_ks.no_repulsion | rq2_dut_fidelity_ks_single | 0.1259 | 3 | 0.3517 | 0.3776 |
| rq2.loco.closest_ks.calibrated | rq2_fidelity_ks_loco_diagnostic | 0.0071 | 3 | 0.0071 | 0.0214 |
| rq2.loco.closest_ks.avec_default | rq2_fidelity_ks_loco_diagnostic | 0.0071 | 3 | 0.0071 | 0.0214 |
| rq2.loco.closest_ks.no_repulsion | rq2_fidelity_ks_loco_diagnostic | 0.0071 | 3 | 0.0071 | 0.0214 |
| rq2.loso.closest_sign.calibrated | rq2_fidelity_paired_loso | 0.0000 | 3 | 0.0000 | 0.0000 |
| rq2.loso.closest_wilcoxon.calibrated | rq2_fidelity_paired_loso | 0.0000 | 3 | 0.0000 | 0.0000 |
| rq2.loso.closest_sign.no_repulsion | rq2_fidelity_paired_loso | 0.0000 | 3 | 0.0000 | 0.0000 |
| rq2.loso.closest_ks.calibrated | rq2_fidelity_ks_loso_diagnostic | 0.0071 | 3 | 0.0071 | 0.0214 |
| rq2.loso.closest_ks.avec_default | rq2_fidelity_ks_loso_diagnostic | 0.0071 | 3 | 0.0071 | 0.0214 |
| rq2.loso.closest_ks.no_repulsion | rq2_fidelity_ks_loso_diagnostic | 0.0071 | 3 | 0.0071 | 0.0214 |
| rq2cap.loco.closedloop.closest_sign.calibrated | rq2_cap_sensitivity_loco | 0.1686 | 9 | 0.1686 | 0.1686 |
| rq2cap.loco.closedloop.closest_sign.avec_default | rq2_cap_sensitivity_loco | 0.0755 | 9 | 0.0850 | 0.1510 |
| rq2cap.loco.closedloop.closest_sign.no_repulsion | rq2_cap_sensitivity_loco | 0.0290 | 9 | 0.0372 | 0.0869 |
| rq2cap.loco.median.closest_sign.calibrated | rq2_cap_sensitivity_loco | 0.0000 | 9 | 0.0000 | 0.0001 |
| rq2cap.loco.median.closest_sign.avec_default | rq2_cap_sensitivity_loco | 0.0000 | 9 | 0.0000 | 0.0001 |
| rq2cap.loco.median.closest_sign.no_repulsion | rq2_cap_sensitivity_loco | 0.0000 | 9 | 0.0000 | 0.0000 |
| rq2cap.loco.uncapped.closest_sign.calibrated | rq2_cap_sensitivity_loco | 0.0000 | 9 | 0.0000 | 0.0000 |
| rq2cap.loco.uncapped.closest_sign.avec_default | rq2_cap_sensitivity_loco | 0.0000 | 9 | 0.0000 | 0.0001 |
| rq2cap.loco.uncapped.closest_sign.no_repulsion | rq2_cap_sensitivity_loco | 0.0000 | 9 | 0.0000 | 0.0000 |
| rq2dm.loco.pure.closest_sign.calibrated | rq2_distmatch_loco | 0.0094 | 12 | 0.0094 | 0.0100 |
| rq2dm.loco.pure.closest_sign.avec_default | rq2_distmatch_loco | 0.0000 | 12 | 0.0000 | 0.0001 |
| rq2dm.loco.pure.closest_sign.no_repulsion | rq2_distmatch_loco | 0.0000 | 12 | 0.0000 | 0.0000 |
| rq2dm.loco.w0.5.closest_sign.calibrated | rq2_distmatch_loco | 0.0025 | 12 | 0.0027 | 0.0100 |
| rq2dm.loco.w0.5.closest_sign.avec_default | rq2_distmatch_loco | 0.0000 | 12 | 0.0000 | 0.0001 |
| rq2dm.loco.w0.5.closest_sign.no_repulsion | rq2_distmatch_loco | 0.0000 | 12 | 0.0000 | 0.0000 |
| rq2dm.loco.w1_id8.closest_sign.calibrated | rq2_distmatch_loco | 0.0025 | 12 | 0.0027 | 0.0100 |
| rq2dm.loco.w1_id8.closest_sign.avec_default | rq2_distmatch_loco | 0.0000 | 12 | 0.0000 | 0.0001 |
| rq2dm.loco.w1_id8.closest_sign.no_repulsion | rq2_distmatch_loco | 0.0000 | 12 | 0.0000 | 0.0000 |
| rq2dm.loco.w1.closest_sign.calibrated | rq2_distmatch_loco | 0.0025 | 12 | 0.0027 | 0.0100 |
| rq2dm.loco.w1.closest_sign.avec_default | rq2_distmatch_loco | 0.0000 | 12 | 0.0000 | 0.0001 |
| rq2dm.loco.w1.closest_sign.no_repulsion | rq2_distmatch_loco | 0.0000 | 12 | 0.0000 | 0.0000 |

## RQ1b claim-(2) family 定義感度

最有意 per-scenario Fisher（S2/avec）の補正後 p は family の取り方に敏感。3 つの view を併記する:

| view | m | min test | raw p | BH q | Holm p | BH 生存 | Holm 生存 |
|---|---|---|---|---|---|---|---|
| avec 条件付き | 3 | `rq1b.rand.fisher.avec.scenario_02` | 0.0078 | 0.0234 | 0.0234 | 生存 | 生存 |
| headline GT（avec+calib） | 6 | `rq1b.rand.fisher.avec.scenario_02` | 0.0078 | 0.0468 | 0.0468 | 生存 | 生存 |
| 全 GT×scenario 走査（弱パワー corner 含む） | 18 | `rq1b.rand.fisher.avec.scenario_02` | 0.0078 | 0.1404 | 0.1404 | 不成立 | 不成立 |

## 修論向け要約（データ駆動）

- **RQ2 忠実度（per-encounter 対応検定, review F5）**: 較正 sim vs 実の最接近距離は同一遭遇の対標本＝符号検定が正当な見出し。real>sim が 24/26 遭遇・raw p=1.05e-05。**忠実度 family 内（m=3, Wilcoxon 併記）で BH q=1.05e-05＝明確に有意**＝sim が実 standoff を再現しきれない＝~0.68m の忠実度ギャップは統計的に実在（限界を補強する向きの所見）。研究横断プール（m=21）でも q=7.34e-05。旧見出しの pooled KS（p=0.007）は対標本に独立2標本検定を当てた仕様ミスで、診断（auxiliary）に降格。
- **RQ1b claim-(2)（分布なし計画は危険）**: 最有意セル S2/avec は raw p=0.0078（pseudo-replication で反保守的＝真の p の下界）。avec 条件付き family（m=3）では BH q=0.0234＝生存だが、弱パワー corner を含む全 GT×scenario 走査（m=18）では BH q=0.1404＝不成立。**＝claim-(2) の per-scenario 信号は family 定義に敏感な境界事例で、確定的ではなく示唆に留まる**（既存 REPORT の『示唆・反応モデル依存・外的妥当性ではない』枠組みと整合）。
- **総括**: 強い結論は RQ1b claim-(1)（robust 利得は全 GT で頑健・`robust_gain_holds`、有意性検定を要さない決定的判定）と RQ2 の忠実度ギャップの実在。claim-(2) の CV 危険性は多重比較後は弱い示唆。多重比較補正は**既存の正直なフレーミングを覆さず追認する**。
