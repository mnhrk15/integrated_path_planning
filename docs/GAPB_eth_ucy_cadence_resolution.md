# gap-b 決着: ETH/UCY の歩行速度異常は「座標スケール」ではなく「per-scene 時間 cadence」

**結論（仮説A 確定）**: ETH/UCY ローダの歩行速度異常（eth が速すぎ・univ が遅すぎ）は
**座標スケールバグではなく、シーンごとの物理 cadence（1 annotation step が表す実時間）の差**である。
**ADE/FDE は dt 非依存**なので **RQ1a（開ループ予測）の再生成は不要**（再実行しても byte 一致を実証済み）。

直前の軸A レビュー（`docs/CODE_REVIEW_axisA_20260619.md`）の完全性 critic が、検査ツール
`examples/inspect_eth_ucy_data.py` で速度異常を発見し「座標スケールの可能性」とフラグしたが、本調査で
誤診と判明したため、検査ツールの枠組みを「annotation cadence」へ是正した。

## 1. 症状（`inspect_eth_ucy_data.py --dt 0.4` 一律、すなわち補正前）

dt=0.4s 一律での per-step 速度分布 [m/s]:

| scene / file | median | p90 | p95 | frac<0.1 | 空間スパン (x×y)[m] |
|---|---|---|---|---|---|
| eth / biwi_eth | **2.45** | **3.06** | 3.23 | 0.04 | 22.1×16.4 |
| hotel / biwi_hotel | 1.28 | 1.81 | 1.93 | 0.22 | 7.6×14.6 |
| univ / students001 | **0.52** | 1.14 | 1.31 | 0.19 | 15.9×14.2 |
| univ / students003 | **0.72** | 1.21 | 1.39 | 0.10 | 15.6×14.1 |
| zara1 / crowds_zara01 | 1.15 | 1.44 | 1.54 | 0.03 | 15.6×12.8 |
| zara2 / crowds_zara02 | 1.01 | 1.46 | 1.57 | 0.31 | 15.9×14.2 |

正常な歩行 median は ~1.0–1.3 m/s。eth が約2倍速く、univ の median が遅い。

## 2. 仮説A（cadence）vs 仮説B（座標スケール）の決着

### 決定的証拠①（データ）: UCY 内での空間スパン同一性
univ(students) のスパン 15.9×14.2 / 15.6×14.1 は zara1(15.6×12.8)・zara2(15.9×14.2) と一致。
**全て同一 UCY プラザ**。仮説B（univ 座標が ~2.5× 小スケール）ならスパンは zara の 1/2.5 に縮むはずだが、
一致している → **univ 座標は正しい metre**。速度異常の原因は時間軸しか残らない。

### 決定的証拠②（文献）: 座標は world-frame metres・eth は加速動画
- OpenTraj / Social-GAN / Trajectron++ はいずれも ETH/UCY を world-frame metres として扱い、
  **座標スケールバグの既知報告は無い**。UCY native 25fps・10 frame ごと（2.5fps=0.4s）が標準。
- **eth は元動画 `seq_eth.avi` が accelerated**（[Trajectron++ issue #67](https://github.com/StanfordASL/Trajectron-plus-plus/issues/67)）。
  これは空間ではなく時間のアーティファクト。複数論文が「eth は 0.4s を 6 frame 扱いにする」等の時間補正を実践。

### 決定的証拠③（コード）: ADE は dt 非依存
`src/core/metrics.py` の `_standard_ade_fde_details`（L88-93）の displacement は純粋な位置差(m)。
dt が効くのは `stride = _steps_for_interval(prediction_dt, dt)`（L50）のみ。RQ1a は
`run_openloop_prediction.py:127` で `calculate_aggregate_metrics(history, dt, dt, ...)`＝
**prediction_dt==dt → stride=1 固定**。dt の絶対値（0.4/0.8/0.2）に関わらず ADE/FDE は不変。
予測入力 `obs_traj` も生座標で dt を含まない。→ **ローダに per-scene dt を入れても予測も ADE も byte 不変**。

## 3. univ の取り扱い（当初案 0.2s → 0.4s 維持へ修正）

ハンドオフは「univ dt=0.2 で median が正常化」を根拠に univ=0.2s を提案したが、これは **median だけ見た錯覚**。
速度**分布**を見ると:
- univ の上位パーセンタイル（p90=1.14/1.21・p95=1.31/1.39）は **zara 並みの正常歩行速度**。
- 低 median は滞留/低速歩行者（frac<0.1≈18.5%）が引き下げているだけ。
- もし真に 0.2s（2×密サンプリング）なら歩行者の p90 も半減（~0.7）するはずだが、半減していない。
- univ=0.2s にすると実歩行者が ~2.6 m/s と過大化する **over-correction**。文献（students も 0.4s）にも反する。

→ **univ は文献どおり 0.4s で正常**。低 median は「実際に滞留の多い大学プラザ」という本物のシーン特性。
これに合わせ、検査ツールの plausibility 判定を **median ではなく moving-speed (p90)** に変更した
（滞留集団にロバスト）。eth のように分布**全体**がシフトしている場合のみ flag される。

## 4. per-scene cadence（`src/datasets/eth_ucy_loader.py: SCENE_DT`）

| scene | cadence dt[s] | 根拠 |
|---|---|---|
| eth | **0.8** | accelerated source video（Trajectron++ #67）。dt=0.8 で分布全体が正常歩行に戻る（median 1.23・p90 1.53）。 |
| hotel | 0.4 | 標準。 |
| univ | 0.4 | 標準 UCY cadence（OpenTraj）。低 median は滞留＝cadence artifact ではない。 |
| zara1 | 0.4 | 標準。 |
| zara2 | 0.4 | 標準。 |

`SGAN_PROTOCOL_DT = 0.4` は SGAN leave-one-out 評価プロトコルが全シーンに一律で割り当てる step で、
ADE/FDE はこれを使う（＝公表 SGAN 結果と比較可能・dt 非依存）。SCENE_DT は**歩行速度サニティチェック専用**。

## 5. RQ1a を再生成しない理由（実証）

ADE パイプライン（座標読込・`run_openloop_prediction.py` の `SGAN_DT=0.4`・`metrics.py`）は**一切変更していない**。
変更箇所（SCENE_DT/scene_dt の追加、検査ツール、docstring）は ADE 経路に非関与。
- `outputs/openloop_full_eval_zara_eth_ucy_seeds0-4.csv` は git 上で**不変**。
- spot-check: eth の `cv,seed0`(ADE 1.075458103993371) と `lstm,seed0`(ADE 0.6286930261597685) を
  再実行し、committed 値と**完全一致（byte 同値）**を確認。
- 回帰テスト `tests/test_metrics.py::test_ade_fde_independent_of_dt_value_when_prediction_dt_equals_dt`
  で dt=0.4 と dt=0.8 が同一 ADE/FDE を返すことを pin。

## 6. Limitation（H1 解釈への影響・文書化のみ）

per-scene cadence は ADE の数値は動かさないが、以下は **修論の limitation として明記すべき**:
- **学習モデルの cadence ミスマッチ confound**: leave-one-out で eth をテストするモデルは eth 以外
  （主に 0.4s 系）で学習されており、eth の物理 0.8s/step の移動を 0.4s 前提で予測する。
  これは lstm/sgan の eth 予測 displacement にバイアスを与え得る。**cv は観測 displacement の等速外挿で
  scale-free＝無影響**。これは我々のローダのバグではなく ETH/UCY + SGAN の既知 confound。
- **横断 ADE 絶対比較の caveat**: シーンごとに物理 horizon（seq_len × 物理 cadence）が異なるため、
  eth の ADE 絶対値は他シーンと異なる実時間スパンの誤差。H1（序列はシーン依存＝sim 人工物）の
  定性結論は不変だが、絶対値の横断比較にはこの caveat が付く。

## 7. 変更ファイル
- `src/datasets/eth_ucy_loader.py`: `SCENE_DT` / `SGAN_PROTOCOL_DT` / `scene_dt()` 追加、docstring 是正。
- `examples/inspect_eth_ucy_data.py`: `scene_scale_report`→`scene_cadence_report`（p90 ベース判定・
  per-scene dt 既定・低 median 注記）、Gap-1/SUMMARY 文言是正、`--dt` を診断用 override に。
- `tests/`: `test_eth_ucy_loader.py`（scene_dt）、`test_metrics.py`（dt 非依存 pin）、
  `test_inspect_eth_ucy.py`（p90 flag・loiter ケース）。
