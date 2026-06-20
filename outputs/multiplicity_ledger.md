# RQ スイート横断 多重比較 ledger

family-wise / FDR 補正（alpha=0.05）。BH-FDR を primary、Holm-Bonferroni を conservative sensitivity として併記。NaN（空 arm の Fisher・空プールの KS）は仮説ではないので family size に数えない。

> **RQ1a は検定を持たない**（開ループ ADE/FDE/NLL は点推定・有意性検定なし＝意図的、review M1/M10）。実行していない検定は補正対象にならない＝「やらなかった検定で p-hack できない」という多重性衛生そのもの。

## 収集した sidecar

- `outputs/rq1b/headline_tests.json` — source=RQ1b-rand, tests=12
- `outputs/rq2_evaluation/headline_tests_loco.json` — source=RQ2-loco, tests=1
- `outputs/rq2_evaluation/headline_tests_loso.json` — source=RQ2-loso, tests=1

## 補正結果（canonical 研究横断 family）

| test_id | family | p_value | family_size | family_bh_q | family_holm_p | overall_bh_q | overall_holm_p |
|---|---|---|---|---|---|---|---|
| rq1b.rand.fisher.avec.scenario_01 | rq1b_claim2_fisher | 0.6000 | 12 | 0.7200 | 1.0000 | 0.7091 | 1.0000 |
| rq1b.rand.fisher.avec.scenario_02 | rq1b_claim2_fisher | 0.0078 | 12 | 0.0936 | 0.0936 | 0.0507 | 0.0936 |
| rq1b.rand.fisher.avec.scenario_03 | rq1b_claim2_fisher | 0.2116 | 12 | 0.6597 | 1.0000 | 0.5717 | 1.0000 |
| rq1b.rand.fisher.calib.scenario_01 | rq1b_claim2_fisher | 0.3576 | 12 | 0.7152 | 1.0000 | 0.6641 | 1.0000 |
| rq1b.rand.fisher.calib.scenario_02 | rq1b_claim2_fisher | 1.0000 | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| rq1b.rand.fisher.calib.scenario_03 | rq1b_claim2_fisher | 0.2199 | 12 | 0.6597 | 1.0000 | 0.5717 | 1.0000 |
| rq1b.rand.fisher.calib_hi.scenario_01 | rq1b_claim2_fisher | 0.3551 | 12 | 0.7152 | 1.0000 | 0.6641 | 1.0000 |
| rq1b.rand.fisher.calib_hi.scenario_02 | rq1b_claim2_fisher | 1.0000 | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| rq1b.rand.fisher.calib_hi.scenario_03 | rq1b_claim2_fisher | 0.4716 | 12 | 0.7200 | 1.0000 | 0.7091 | 1.0000 |
| rq1b.rand.fisher.calib_lo.scenario_01 | rq1b_claim2_fisher | 0.6000 | 12 | 0.7200 | 1.0000 | 0.7091 | 1.0000 |
| rq1b.rand.fisher.calib_lo.scenario_02 | rq1b_claim2_fisher | 0.6000 | 12 | 0.7200 | 1.0000 | 0.7091 | 1.0000 |
| rq1b.rand.fisher.calib_lo.scenario_03 | rq1b_claim2_fisher | 0.0673 | 12 | 0.4038 | 0.7403 | 0.2916 | 0.7403 |
| rq2.loco.closest_ks.calibrated | rq2_fidelity_ks_loco | 0.0071 | 1 | 0.0071 | 0.0071 | 0.0507 | 0.0927 |

- 研究横断 family size（overall）: **13** 検定
- overall BH-FDR で生存（q<0.05）: **0** / Holm で生存: **0**

> **overall の読み方**: overall は RQ2 忠実度と RQ1b 計画安全という*異なる問い*を1 family に束ねた最保守の境界（cross-suite 過剰補正）。適切な評価単位は各 family 内補正（上表 `family_bh_q`）と下の RQ1b family 定義感度。overall は『最悪でもこの程度』の sanity 上限として読む（実際 RQ2 忠実度は family 内 q=0.007 で明確に有意・overall でのみ境界化）。

## 付録: LOSO（補助・overall から除外）

LOSO は LOCO と同じ忠実度の問いの再分割。二重計上を避けるため overall には含めず、family 内補正のみ示す。

| test_id | family | p_value | family_size | family_bh_q | family_holm_p | overall_bh_q | overall_holm_p |
|---|---|---|---|---|---|---|---|
| rq2.loso.closest_ks.calibrated | rq2_fidelity_ks_loso | 0.0071 | 1 | 0.0071 | 0.0071 | 0.0071 | 0.0071 |

## RQ1b claim-(2) family 定義感度

最有意 per-scenario Fisher（S2/avec）の補正後 p は family の取り方に敏感。3 つの view を併記する:

| view | m | min test | raw p | BH q | Holm p | BH 生存 | Holm 生存 |
|---|---|---|---|---|---|---|---|
| avec 条件付き（3） | 3 | `rq1b.rand.fisher.avec.scenario_02` | 0.0078 | 0.0234 | 0.0234 | 生存 | 生存 |
| headline GT（avec+calib, 6） | 6 | `rq1b.rand.fisher.avec.scenario_02` | 0.0078 | 0.0468 | 0.0468 | 生存 | 生存 |
| 全 GT×scenario 走査（12, 弱パワー corner 含む） | 12 | `rq1b.rand.fisher.avec.scenario_02` | 0.0078 | 0.0936 | 0.0936 | 不成立 | 不成立 |

## 修論向け要約（データ駆動）

- **RQ2 忠実度（closest-approach KS）**: 較正 sim vs 実の pooled KS は raw p=0.007。**忠実度 family 内（m=1）では BH q=0.007＝明確に有意**＝sim が実 standoff を再現しきれない＝~0.68m の忠実度ギャップは統計的に実在（限界を補強する向きの所見）。RQ1b の無関係な計画検定まで一括する最保守の研究横断プール（m=13）でのみ q=0.051＝境界化（異質な問いを跨ぐ過剰補正のため参考値）。
- **RQ1b claim-(2)（分布なし計画は危険）**: 最有意セル S2/avec は raw p=0.0078（pseudo-replication で反保守的＝真の p の下界）。avec 条件付き family（m=3）では BH q=0.0234＝生存だが、弱パワー corner を含む全 12 セル走査（m=12）では BH q=0.0936＝不成立。**＝claim-(2) の per-scenario 信号は family 定義に敏感な境界事例で、確定的ではなく示唆に留まる**（既存 REPORT の『示唆・反応モデル依存・外的妥当性ではない』枠組みと整合）。
- **総括**: 強い結論は RQ1b claim-(1)（robust 利得は全 GT で頑健・`robust_gain_holds`、有意性検定を要さない決定的判定）と RQ2 の忠実度ギャップの実在。claim-(2) の CV 危険性は多重比較後は弱い示唆。多重比較補正は**既存の正直なフレーミングを覆さず追認する**。
