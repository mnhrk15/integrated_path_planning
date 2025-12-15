# Integrated Path Planning with Pedestrian Trajectory Prediction

統合経路計画システム - Social ForceモデルとSocial-GANによる歩行者軌道予測、Frenet座標系を用いた自動運転車の安全な経路計画を実現するシミュレーション環境

## 概要

このプロジェクトは、以下の3つのコンポーネントを統合します：

1. **Social Force Model**: 歩行者の動きをシミュレート（Ground Truth生成）
2. **Social-GAN**: 歩行者の未来軌道を予測
3. **Frenet Optimal Trajectory**: 予測された歩行者を回避する安全な経路を計画

## システムアーキテクチャ

```
[Social Force Simulator] → [Pedestrian Observer] → [Social-GAN Predictor]
                                                            ↓
                                                    [Predicted Trajectories]
                                                            ↓
[Ego Vehicle State] ← [Frenet Planner] ← [Coordinate Converter]
```

## インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd integrated_path_planning

# 仮想環境の作成と有効化
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt

# パッケージのインストール（開発モード）
pip install -e .
```

## 学習済みモデルのダウンロード（NEW! 🆕）

Social-GANの公式学習済みモデルを簡単にダウンロードできます：

### 方法1: Pythonスクリプト（推奨）

```bash
# 基本モデルのみダウンロード
python scripts/download_sgan_models.py

# プーリングモデルも含めてダウンロード
python scripts/download_sgan_models.py --pooling
```

### 方法2: Bashスクリプト

```bash
bash scripts/download_sgan_models.sh
```

ダウンロードされるモデル：
- `models/sgan-models/eth_8.pt` - ETHデータセット（予測長8ステップ）
- `models/sgan-models/hotel_8.pt` - HOTELデータセット
- `models/sgan-models/univ_8.pt` - UNIVデータセット
- `models/sgan-models/zara1_8.pt` - ZARA1データセット
- `models/sgan-models/zara2_8.pt` - ZARA2データセット
- `models/sgan-models/*_12.pt` - 各データセット（予測長12ステップ）

モデルサイズ: 各モデル約5-10MB

## 使用方法

### 基本的な使い方

```python
from src.simulation.integrated_simulator import IntegratedSimulator
from src.config import load_config

# 設定ファイルの読み込み
config = load_config('scenarios/scenario_01_crossing.yaml')

# シミュレータの初期化
simulator = IntegratedSimulator(config)

# シミュレーションの実行
results = simulator.run(n_steps=100)

# 結果の保存と可視化
simulator.save_results()
simulator.visualize()
```

### コマンドラインからの実行

#### 基本実行
```bash
python examples/run_simulation.py --scenario scenarios/scenario_01_crossing.yaml
```

#### アニメーション生成（NEW! 🆕）
```bash
# GIFアニメーション生成
python examples/run_simulation.py \
    --scenario scenarios/scenario_01_crossing.yaml \
    --animate \
    --animation-format gif \
    --fps 10

# MP4アニメーション生成（高品質）
python examples/run_simulation.py \
    --scenario scenarios/scenario_02_corridor.yaml \
    --animate \
    --animation-format mp4 \
    --fps 20
```

#### アニメーションデモ
```bash
# 両フォーマット（GIF + MP4）を生成するデモ
python examples/demo_animation.py
```

### Pythonコードでアニメーション作成

```python
from src.simulation.integrated_simulator import IntegratedSimulator
from src.visualization import create_simple_animation
from src.config import load_config

# シミュレーション実行
config = load_config('scenarios/scenario_01_crossing.yaml')
simulator = IntegratedSimulator(config)
results = simulator.run(n_steps=150)

# アニメーション作成
create_simple_animation(
    results=results,
    output_path='output/my_animation.gif',
    show=True,  # 表示する
    show_predictions=True,  # 予測軌道を表示
    show_metrics=True,  # メトリクスを表示
    fps=10
)
```

### 学習済みモデルを使用する場合

シナリオYAMLファイルでモデルパスを指定：

```yaml
# scenarios/my_scenario.yaml
sgan_model_path: "models/sgan-models/eth_8.pt"  # モデルを使用
# sgan_model_path: null  # モデルなし（定速度予測）
```

## プロジェクト構成

```
integrated_path_planning/
├── src/
│   ├── config/          # 設定管理
│   ├── core/            # 基本データ構造と座標変換
│   ├── pedestrian/      # Social Force統合と観測
│   ├── prediction/      # Social-GAN統合
│   ├── planning/        # Frenet経路計画
│   ├── simulation/      # 統合シミュレータ
│   └── visualization/   # 可視化
├── scenarios/           # シミュレーションシナリオ
├── models/              # 学習済みモデル
├── tests/               # ユニットテスト
└── examples/            # 使用例
```

## シナリオ

複数のシナリオが用意されています：

1. **scenario_01_crossing.yaml**: 歩行者との交差シナリオ
2. **scenario_02_corridor.yaml**: 狭い通路でのすれ違いシナリオ

## テスト

```bash
pytest tests/
```

## 評価指標

- **安全性**: 最小距離（歩行者との最短距離）
- **効率性**: 目標到達時間
- **快適性**: 最大加速度、最大ジャーク

## ライセンス

MIT License

## 参考文献

1. Helbing, D., & Molnár, P. (1995). Social force model for pedestrian dynamics.
2. Gupta, A., et al. (2018). Social GAN: Socially Acceptable Trajectories with GANs.
3. Werling, M., et al. (2010). Optimal trajectory generation for dynamic street scenarios in a Frenet Frame.
