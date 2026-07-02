# RQ2 計装監査 REPORT: 速度キャップ切り分け × 分布マッチング較正

生成: `examples/make_rq2_instrument_report.py`（全 verdict は純関数、prose は表から機械生成）。入力: `outputs/rq2_instrument_audit/` の cap / distmatch / surfaces 出力。

判定しきい値: gap 縮小 >= 25%・sign 優位崩壊 p > 0.05・F1 は ADE >= 2% かつ |gap| >= 10% の同時改善・識別性回復はバンド幅 <= 0.5x(ADE 基準)。

## 1. 速度キャップ方策の切り分け（review F2）

pooled 表（1.1/1.2/2.1）は in-sample（全26遭遇で fit→同一遭遇で評価）の記述値であり、p 値は掲載しない（in-sample の検定 p は確証的に読めず、ledger 登載対象は held-out LOCO の対応検定のみ）。判定は全て LOCO held-out（1.3/2.2）に基づく。

### 1.1 pooled 較正（全26遭遇・方策内3アーム対照）

| policy | cap_headroom | sigma | v0 | fit_loss | ade_calibrated | gap_calibrated | n_real_gt_sim_calibrated | n_pairs |
|---|---|---|---|---|---|---|---|---|
| median | 1.000 | 1.211 | 1.648 | 0.657 | 0.657 | 0.679 | 24 | 26 |
| closedloop | 1.000 | 0.616 | 35.865 | 1.707 | 1.707 | 0.351 | 17 | 26 |
| uncapped | inf | 1.005 | 0.890 | 0.715 | 0.715 | 0.763 | 25 | 26 |
| capfit | 1.000 | 1.211 | 1.648 | 0.657 | 0.657 | 0.679 | 24 | 26 |

### 1.2 capfit ヘッドルーム掃引（pooled）

| m | sigma | v0 | fit_loss | refined | ade | gap | n_pairs | n_real_gt_sim |
|---|---|---|---|---|---|---|---|---|
| 1.000 | 1.211 | 1.648 | 0.657 | True | 0.657 | 0.679 | 26 | 24 |
| 1.150 | 0.496 | 3.306 | 0.686 | True | 0.686 | 0.739 | 26 | 25 |
| 1.300 | 0.299 | 4.487 | 0.700 | True | 0.700 | 0.741 | 26 | 25 |
| 1.500 | 1.007 | 0.886 | 0.714 | True | 0.714 | 0.765 | 26 | 25 |
| 2.000 | 1.005 | 0.890 | 0.715 | True | 0.715 | 0.763 | 26 | 25 |

m*（pooled fit loss 最小）= **1**。m=1.0 は median 経路へのエイリアス（構成的にビット同一・回帰テストで固定）のため、capfit の LOCO 再走は median の複製になるだけであり実行しない（uncapped が分離キャップの held-out 証拠を担う）。

### 1.3 LOCO held-out（判定の根拠）

| policy | sigma_mean | v0_mean | test_ade | ade_avec | mean_gap | gap_avec | n_real_gt_sim | n_pairs | sign_p |
|---|---|---|---|---|---|---|---|---|---|
| closedloop | 0.4991 | 4298 | 1.599 | 1.765 | 0.3465 | 0.68 | 17 | 26 | 0.1686 |
| median | 1.168 | 1.712 | 0.6402 | 0.6392 | 0.6797 | 0.6826 | 24 | 26 | 1.05e-05 |
| uncapped | 0.9322 | 0.9394 | 0.694 | 0.7192 | 0.7645 | 0.6219 | 25 | 26 | 8.05e-07 |

注記: closedloop 行の v0_mean は fold ごとの縮退フィット（fold 範囲 [13.7, 3.54e+04]）の平均であり「較正値」ではない（fold 詳細は folds_cap_closedloop_loco.csv）。

### 1.4 verdict（standoff 過小再現の帰属）

**structural_limit** — no decoupled policy shrinks the gap >= 25% with broken direction dominance

- `closedloop`: gap +0.347 m（median +0.680 m、変化 -49.0%）・sign 17/26 (p=0.169)・gap縮小=yes・優位崩壊=no（**verdict 対象外**: confounded regime probe (speed confound)）
- `uncapped`: gap +0.764 m（median +0.680 m、変化 +12.5%）・sign 25/26 (p=8.05e-07)・gap縮小=no・優位崩壊=no

クロスチェック（fitter 交絡対策）: 判定は再フィット較正アームに基づくが、ADE fitter は v0 にほぼ不感（C2）なので「fitter が縮めに行かないだけ」の可能性は方策内 AVEC 対照（固定の強斥力・再フィットなし）で棄却する — uncapped レジームの AVEC 対照でも gap +0.622 m と正の standoff 過小再現が残存＝強斥力を固定しても gap は閉じない。

## 2. 分布マッチング較正（(A)-2）

### 2.1 pooled 重み掃引

| config | dist_metric | w_dist | interaction_distance | sigma | v0 | fit_loss | ade | emd_closest | gap | n_real_gt_sim | n_pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w0 | emd | 0.000 | nan | 1.211 | 1.648 | 0.657 | 0.657 | 0.687 | 0.679 | 24 | 26 |
| w0.25 | emd | 0.250 | nan | 1.366 | 2.101 | 0.821 | 0.662 | 0.638 | 0.626 | 21 | 26 |
| w0.5 | emd | 0.500 | nan | 1.479 | 2.468 | 0.976 | 0.673 | 0.604 | 0.578 | 21 | 26 |
| w1 | emd | 1.000 | nan | 0.649 | 21.282 | 1.219 | 0.776 | 0.443 | 0.418 | 20 | 26 |
| w2 | emd | 2.000 | nan | 0.446 | 104.355 | 1.626 | 0.828 | 0.399 | 0.371 | 20 | 26 |
| w4 | emd | 4.000 | nan | 0.338 | 529.040 | 2.378 | 0.866 | 0.378 | 0.349 | 20 | 26 |
| pure | emd | 1.000 | nan | 0.318 | 874.042 | 0.376 | 0.879 | 0.376 | 0.340 | 19 | 26 |
| w0_id8 | emd | 0.000 | 8.000 | 1.211 | 1.648 | 0.659 | 0.657 | 0.687 | 0.679 | 24 | 26 |
| w1_id8 | emd | 1.000 | 8.000 | 0.649 | 21.281 | 1.222 | 0.776 | 0.443 | 0.418 | 20 | 26 |

注: `fit_loss` は目的関数値そのもの（w 依存）で**行間比較不可**（pure の 0.376 は EMD 単独値であり「fit が良い」の意味ではない）。行間で比較可能な共通尺度は `ade` と `emd_closest`/`gap` のみ。

### 2.2 LOCO held-out

| config | cap_policy | dist_metric | sigma_mean | v0_mean | test_ade | ade_avec | mean_gap | gap_avec | n_real_gt_sim | n_pairs | sign_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pure | median | emd | 0.3175 | 949.2 | 0.8449 | 0.6392 | 0.3455 | 0.6826 | 20 | 26 | 0.009355 |
| w0.5 | median | emd | 1.41 | 2.713 | 0.6614 | 0.6392 | 0.6018 | 0.6826 | 21 | 26 | 0.002494 |
| w1_id8 | median | emd | 0.7613 | 21.7 | 0.7351 | 0.6392 | 0.509 | 0.6826 | 21 | 26 | 0.002494 |
| w1 | median | emd | 0.711 | 23.62 | 0.7426 | 0.6392 | 0.4969 | 0.6826 | 21 | 26 | 0.002494 |
| w0 | median | emd | 1.168 | 1.712 | 0.6402 | 0.6392 | 0.6797 | 0.6826 | 24 | 26 | 1.05e-05 |

**警告**: config ['w0'] は非デフォルトの --scenario で実行されており、verdict/F1 から除外している。

### 2.3 verdict（standoff 改善 × ADE 犠牲）

**undetermined** — w0 reference row missing


## 3. 識別性監査（σ軸・v0軸の 2% バンド幅）

| objective | policy | axis | band_lo | band_hi | band_width | band_contiguous | censored_lo | censored_hi | fitted | fitted_on_grid_edge |
|---|---|---|---|---|---|---|---|---|---|---|
| ade | closedloop | sigma | 0.900 | 2.100 | 1.200 | True | False | False | 0.616 | False |
| ade | closedloop | v0 | 6.500 | 8.000 | 1.500 | True | False | True | 35.865 | True |
| ade | median | sigma | 0.500 | 1.900 | 1.400 | True | False | False | 1.214 | False |
| ade | median | v0 | 0.800 | 2.400 | 1.600 | True | False | False | 1.637 | False |
| ade | uncapped | sigma | 0.300 | 1.500 | 1.200 | True | True | False | 0.759 | False |
| ade | uncapped | v0 | 0.400 | 2.000 | 1.600 | True | False | False | 1.147 | False |
| pure | closedloop | sigma | 0.500 | 0.900 | 0.400 | True | False | False | 0.224 | True |
| pure | closedloop | v0 | 8.000 | 8.000 | 0.000 | True | False | True | 22772.626 | True |
| pure | median | sigma | 0.900 | 1.100 | 0.200 | True | False | False | 0.308 | False |
| pure | median | v0 | 6.500 | 8.000 | 1.500 | True | False | True | 1108.213 | True |
| pure | uncapped | sigma | 0.900 | 0.900 | 0.000 | True | False | False | 0.592 | False |
| pure | uncapped | v0 | 8.000 | 8.000 | 0.000 | True | False | True | 24.875 | True |
| w1 | closedloop | sigma | 0.500 | 1.300 | 0.800 | True | False | False | 0.207 | True |
| w1 | closedloop | v0 | 6.500 | 8.000 | 1.500 | True | False | True | 33660.780 | True |
| w1 | median | sigma | 0.900 | 0.900 | 0.000 | True | False | False | 0.649 | False |
| w1 | median | v0 | 6.500 | 8.000 | 1.500 | True | False | True | 21.320 | True |
| w1 | uncapped | sigma | 0.700 | 0.900 | 0.200 | True | False | False | 0.425 | False |
| w1 | uncapped | v0 | 6.500 | 8.000 | 1.500 | True | False | True | 26.523 | True |

識別性の回復（バンド幅 <= 0.5x ADE 基準・自軸非打切り・連結バンド・fitted が自バンド内・他軸の fitted がグリッド内・ADE 基準行も同品質、の全条件）: **no**

判定規則の理由: profile_band は各軸を「他軸の fitted に最近傍のグリッドノード」で切る。本表の分布目的の fitted v0 は 21.3〜3.37e+04 とグリッド外へ発散するため（最上ノードへクランプ）、そのスライス上の鋭い σ バンドは**最適点から遠い条件付き断面**の性質であり、識別性の回復とは読めない（(他軸端) 注記）。幅 0 は「グリッド刻み未満」（(1ノード) 注記）、(非連結) はバンドの凸包が out-of-band の尾根を跨ぐ多峰プロファイル。なお 2% バンドは各目的関数自身の最小値に対する相対幅であり、目的関数間の絶対許容は同一でない（バンド幅比較はこの前提つきで読む）。

- `closedloop/sigma`: ADE 基準 1.2(基準劣化) → pure: 0.4(他軸端)(fit外), w1: 0.8(他軸端)(fit外)
- `closedloop/v0`: ADE 基準 1.5(打切り)(基準劣化) → pure: 0(打切り)(他軸端)(1ノード)(fit外), w1: 1.5(打切り)(他軸端)(fit外)
- `median/sigma`: ADE 基準 1.4 → pure: 0.2(他軸端)(fit外), w1: 0(他軸端)(1ノード)(fit外)
- `median/v0`: ADE 基準 1.6 → pure: 1.5(打切り)(fit外), w1: 1.5(打切り)(fit外)
- `uncapped/sigma`: ADE 基準 1.2(打切り)(基準劣化) → pure: 0(他軸端)(1ノード)(fit外), w1: 0.2(他軸端)(fit外)
- `uncapped/v0`: ADE 基準 1.6 → pure: 0(打切り)(1ノード)(fit外), w1: 1.5(打切り)(fit外)

## 4. 総合 verdict（review F1: 較正は手調整に勝てるか）

**f1_stands**

対象（実際に判定した LOCO 構成）: ['cap:median', 'cap:uncapped', 'dm:pure', 'dm:w0.5', 'dm:w1', 'dm:w1_id8']（方策内 AVEC 対照との比較）。closedloop は両アームとも歩速 ~30% 過大の壊れたレジーム内比較になるため F1 の証拠から除外（§1.4 の交絡注記と同一の理由）。非デフォルト provenance / 非 median レジームの構成も除外（§1.3/§2.2 の警告参照）。

監査した全 6 構成で、較正は AVEC 手調整 (0.7, 3.5) を held-out ADE と standoff の両方で同時に上回れなかった（F1 の否定的所見は維持・強化）。
片側のみ改善した構成: ['cap:uncapped', 'dm:pure', 'dm:w0.5', 'dm:w1_id8', 'dm:w1']

### 4.1 新較正点の RQ1b 掃引域チェック

| point | sigma | v0 | status |
|---|---|---|---|
| cap:closedloop | 0.499 | 4298.491 | outside |
| cap:median | 1.168 | 1.712 | inside_box |
| cap:uncapped | 0.932 | 0.939 | outside |
| dm:pure | 0.318 | 949.212 | outside |
| dm:w0.5 | 1.410 | 2.713 | outside |
| dm:w1_id8 | 0.761 | 21.697 | outside |
| dm:w1 | 0.711 | 23.619 | outside |
| dm:w0 | 1.168 | 1.712 | inside_box |

`outside` の点は committed RQ1b 掃引（±1SD 箱＋LOSO 包絡）がカバーしない。これらは診断用の器具設定であり正準較正点 (1.168, 1.712) を置換するものではないが、いずれかを研究上採用する場合は RQ1b の追加 arm が必要（研究判断・本レポートは再走しない）。

## 5. §3(B)（実データ接地閉ループ）への設計含意

キャップ方策を解放しても分布目的を足しても、SFM 斥力は実データの standoff 分布を held-out で再現できなかった。正準（median レジーム）では ADE でも手調整と識別不能（0.640 vs 0.639）のまま残り、片側のみの改善（§4 partial）は ADE×standoff の同時改善に届かなかった。これは (B) の replay 対照設計を直接支持する: 較正 SFM を「実歩行者の代理」として信頼する根拠は現状存在しないため、閉ループ評価の反応性軸には replay（記録実歩行者）アームが不可欠であり、SFM 系アーム（較正/手調整/斥力なし）は「反応モデル仮定の感度幅」を張る器具として位置づけるべきである。較正の限界そのものが測定妥当性研究の証拠（ベンチマーク結論の誤差棒）になる。

### 実行上の注記
- closedloop アームの歩速 ~30% 過大は F2 開示（calibration_harness module docstring）参照。閉ループ徹底整合には desired 速度も 1.3x する本アームの挙動が「閉ループが録画歩行と不整合」という所見そのもの。
- 本レポートに掲載する p 値は held-out LOCO の対応検定のみで、全て auxiliary sidecar（rq2cap.*/rq2dm.*）として multiplicity ledger に登載済み（canonical family には不算入）。pooled CSV に含まれる in-sample の sign_p 列は記述用の生データであり、本レポートには掲載せず確証的に読まないこと（fit と検定が同一26遭遇＝in-sample）。
