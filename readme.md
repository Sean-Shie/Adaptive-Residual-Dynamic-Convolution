# ARDC-Conv: Adaptive Residual Dynamic Convolution
**用於淺層卷積神經網路之穩定且高效的卷積算子設計**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/) 
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Academic](https://img.shields.io/badge/Status-Research-blue.svg)](https://github.com/)

## 📖 研究摘要
傳統卷積神經網路（CNNs）往往透過增加深度來提升性能，但這在資源受限的臨床系統或邊緣運算設備上並非首選。本研究提出 **自適應殘差動態卷積 (ARDC-Conv)**，其核心理念在於「**動態校正而非完全動態替換**」。透過將動態卷積核作為輸入依賴的殘差項，結合靜態主幹卷積核，本方法在僅有三層的淺層網路架構下，能顯著提升特徵表徵能力與訓練穩定性。

<div align="center">
  <img src="./results/ARDC_Conv_Architecture.png" width="100%">
</div>

## ✨ 核心設計特性

### 1. 殘差動態配方 (Residual Dynamic Formulation)
不同於傳統動態卷積完全依賴路由權重（容易導致訓練不穩定或路由塌陷），ARDC-Conv 採用以下結構：
$$W_{eff}(x) = W_{base} + \sum_{k=1}^{K} \alpha_k(x) \cdot \lambda_k \cdot \Delta W_k$$
* **$W_{base}$**：穩定的靜態基礎卷積核，提供穩定的梯度傳遞路徑。
* **$\Delta W_k$**：動態殘差卷積核，負責針對輸入樣本進行微小的輸入依賴修正。
* **$\lambda_k$**：可學習的核縮放參數，用以調節各動態分支的響應強度 。

### 2. 全域-局部協作路由 (Global-Local Collaborative Routing)
為了更精準地產生路由係數 $\alpha_k$，我們設計了雙分支機制 ：
* **全域分支 (Global Branch)**：利用全域平均池化與 1x1 卷積建立全域語意上下文 。
* **局部分支 (Local Branch)**：使用空洞卷積（Dilated Conv）提取局部多尺度特徵 。
* **溫度調控 (Temperature Control)**：引入溫度參數 $\tau$ 來控制 Softmax 的分佈熵，防止路由權重過於集中或過於平滑 。

### 3. 協作注意力與輸出重新校正
* **Channel-Spatial Attention**：在卷積前先進行通道與空間注意力的雙重加權，有效抑制背景噪聲 。
* **Learnable Internal Residual**：當輸入輸出維度相同時，引入帶有 `tanh` 調控的可學習殘差縮放 $\gamma$，確保在淺層網路中梯度能有效回傳 。

---

## 📊 實驗結果與分析

本研究於 **CIFAR-10** 資料集進行驗證，模型採用標準三層淺層 CNN 架構 。

### 1. 整體效能對比
| 模型 (Shallow CNN) | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (Standard Conv2d) | 85.88% | 85.89% | 85.88% | 85.84% |
| **ARDC-Conv (Ours)** | **90.37%** | **90.36%** | **90.37%** | **90.33%** |

*結果顯示，僅透過卷積算子的重新設計，即可帶來 **+4.49%** 的絕對準確率提升 。*

### 2. 細粒度分類能力提升
實驗證明，ARDC-Conv 在辨識高度依賴局部紋理特徵的類別時表現尤為優異 ：
* **鳥類 (Bird)**: **+7.6%**
* **鹿 (Deer)**: **+7.2%**
* **貓 (Cat)**: **+6.2%**

這些類別通常需要捕捉細微的邊緣與毛髮結構，ARDC-Conv 的動態修正機制能比靜態卷積更有效地提取這些特徵 。

---

## 🛠️ 實作指引

### 環境需求
* Python 3.10+
* PyTorch 2.70+
* Torchvision

### 核心模塊實作範例
您可以在 `new_cnn_test.py` 中找到 `ARDC_Conv` 的具體實作。以下為其呼叫方式：

```python
from new_cnn_test import ARDC_Conv

# 替換標準的 nn.Conv2d
# in_channels=64, out_channels=128, kernel_size=3
self.conv = ARDCConv(in_channels=64, out_channels=128, kernel_size=3, padding=1, num_kernels=4)
```

### 訓練腳本
我們在訓練過程中額外啟用了 **Random Horizontal Flip**，**Random Crop**，**MixUp** 與 **CutMix** 資料增強技術，以進一步提升泛化能力 。
```bash
python new_cnn_test.py
```

---

## 📂 專案目錄結構
```text
├── train_and_evaluate.py       # 主訓練與測試的程式碼
├── models.py                   # ARDC-Conv 算子核心實作及 BaselineCNN (nn.Conv2d)
├── checkpoints/
│   ├── BaselineCNN_best.pth    # 三層標準 CNN 架構的訓練後權重
│   └── ARDCCNN_best.pth        # 使用 ARDC-Conv 的提議架構的訓練後權重
└── data/cifar-10-batches-py    # CIFAR10 資料集
```

---

## 📝 結論與應用潛力
ARDC-Conv 證明了卷積算子的數學重構在淺層網路中的關鍵性 。透過保留靜態路徑與引入受控動態殘差，本方法達到了效能與穩定性的平衡，非常適合應用於：
1.  **臨床醫療影像分析**：資源受限的診斷設備 。
2.  **Edge AI**：行動裝置上的即時推論任務 。
3.  **細粒度影像分類**：需要高度局部自適應能力的視覺任務 。

---

## 📚 參考文獻
* [1]	LeCun Y, Bottou L, Bengio Y et al: Gradient-Based Learning Applied to Document Recognition. In: Proceedings of the IEEE. vol. 86; 1998: 2278--2324.
* [2]	Krizhevsky A, Hinton G: Learning multiple layers of features from tiny images. In.; 2009.
* [3]	Brabandere BD, Jia X, Tuytelaars T et al: Dynamic filter networks. In: Proceedings of the 30th International Conference on Neural Information Processing Systems. Barcelona, Spain: Curran Associates Inc.; 2016: 667–675.
* [4]	Yang B, Bender G, Le QV et al: CondConv: conditionally parameterized convolutions for efficient inference. Red Hook, NY: Curran Associates Inc.; 2019.
* [5]	Chen Y, Dai X, Liu M et al: Dynamic Convolution: Attention Over Convolution Kernels. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020.
* [6]	Hou Q, Zhou D, Feng J: Coordinate Attention for Efficient Mobile Network Design. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 20-25 June 2021 2021; 2021: 13708-13717.
