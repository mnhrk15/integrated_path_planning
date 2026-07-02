# 較正点3種の正準化注記（review 1.2-5）

このディレクトリの `cruise_sensitivity.csv` の `baseline_median` 行
(sigma, v0) = (1.2005, 1.6219) は、既定 cruise 推定器下の**全プール単一フィット**であり、
正準較正値ではない。リポジトリ内に併存する3点を混同しないこと:

| 点 (sigma, v0) | 実体 | 使用箇所 |
|---|---|---|
| (1.2005, 1.6219) | 全プール単一フィット（cruise 診断の baseline） | この CSV／DUT CLI 既定 1.20/1.62 の丸め元 |
| (1.156, 1.681) | radius=0.35 の LOCO fold 平均 | RQ1b GT `calib` アーム／committed DUT fidelity CSV の実行点 |
| (1.168, 1.712) | radius=0.30 の LOCO fold 平均＝**現行正準** | outputs/rq2_evaluation/summary_loco.txt |

3点の差は σ で最大 ~4%・v0 で最大 ~5.6% で、いずれも RQ1b の ±1SD 感度箱の内側
（M6 注記参照）＝committed な結論は選択に依存しないが、修論本文では正準点
(1.168, 1.712) を引用し、他2点は由来を明示して言及すること。

（examples/run_rq2_cruise_sensitivity.py --note-only で再生成）
