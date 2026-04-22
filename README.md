## steel-fault-detection-u-net

My attempt at an old kaggle competition to get more perspective into encoding and multi-label detection! Firstly shoutout to these people and the resourses they created which taught me a lot:

1. https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
2. https://www.kaggle.com/code/amanooo/defect-detection-starter-u-net - best EDA on the dataset if you want to get started on learning about the problem
3. https://www.kaggle.com/code/liyufeng816/notebook-lyf
4. https://www.sciencedirect.com/science/article/pii/S259012302500060X
5. https://github.com/qubvel-org/segmentation_models.pytorch - another great resource I found later for quick solution for testing and deeper archetictures for segmentation tasks. (if i had a proper GPU the next step would be to compare my perfomrances ith these SOTA segmentors)

This project explores lightweight U-Net architectures for steel surface defect segmentation, with a focus on handling **class imbalance** and improving **generalization** under constrained compute (Apple M1). (the following is a ChatGPT refined readme because I have a tendency of getting verbose and writing long sentences - I have added notes to a clean summery where I feel depth is needed)

---

## ⚙️ Design Goals
- Keep the model **lightweight** enough to train on an M1 (~4–8 hours)
- Improve **generalization** with minimal but effective augmentation - the dataset is quite low quality in that sense that it has a lot og light variation and objects
- Address **severe class imbalance**, especially dominance of class 3
- Gain better **training visibility** via per-class metrics

---

## 🖼️ Data Processing & Augmentation
- Images are normalized to `[0, 1]`
- Augmentation is intentionally **lightweight** to avoid synthetic artifacts:
  - **Contrast stretching** (percentile-based normalization)
  - **Light intensity distortion** (random brightness scaling)

This approach aims to improve robustness without introducing unrealistic defects.

---

## 🏗️ Model Variants

### `unet4_16`
- Baseline architecture
- Loss: **Binary Cross-Entropy + Dice Loss**
- Serves as a reference for evaluating more advanced loss strategies

### `unet5_8`
- Modified, deeper U-Net with reduced filter sizes (compute-efficient)
- Focus: improving class balance and segmentation quality via loss design

---

## 📉 Loss Function Experiments

### ⚠️ Core Challenge
Class 3 is:
- **Highly overrepresented**
- Has **much larger masks**

This causes models to **optimize for class 3**, inflating global Dice score while neglecting smaller classes. - This is where I learnt a lot about Dice computation and I highly recommend you look into it too for the [perspective]([url](https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient)). TLDR - its a pixel-wise comparison. Over-represented classes and/or classes with bigger masks dominate and compound till your gradients for other classes disappear and your performance bottlenecks in the late stage. 

While static weighting apporach gets 85+ performance - I believe code should be adaptive so I explored some custom computations and see if I can hit that 0.90 last mile with a lightweight model.

---

### 🧪 Approaches Tested

#### 1. Mean Dice Loss (MDL)
- Computes Dice per class and averages - so each class takes a 0.25 space of the 1 in dice loss (thoeretically should work but didnt)
- ✅ Improves performance on smaller classes  
- ❌ Overcompensates → hurts overall trade-offs and stability

---

#### 2. Generalized Dice Loss (GDL) + Weighted BCE (**final approach**)

**Goal:** Balance gradient contribution across classes based on pixel coverage

- Class weights computed as **inverse of class pixel frequency (per batch)**
- Helps prevent dominance of large classes

##### Weighting Modes:

- **Root scaling (RT)** *(best performing)*  
  - Reduces harshness of inverse weighting  
  - Tunable via power parameter  
  - Best results around: `power = 0.3 – 0.4`

- **Log scaling**
  - Softer weighting alternative  
  - ❌ Behaved similarly to MDL → overemphasized small classes

---

### 🔧 BCE Component
- Weighted BCE with `α = 0.75`
- Biases learning toward **foreground (defect masks)** over background

---

## 📊 Metrics & Monitoring
- Global **Dice Coefficient**
- **Per-class Dice scores** for all 4 classes

This provides visibility into class-wise learning and helps diagnose imbalance issues during training.

---

## ✅ Key Takeaways
- Vanilla Dice Loss is **not robust to extreme class imbalance**
- Direct inverse weighting can be **too aggressive**
- **Root-scaled GDL + weighted BCE** provides the best balance so far
- Careful loss design is critical for improving performance on **underrepresented defect classes**

## Final notes and discussions 
The last model I want to try is a unet5_16 with (4&5 being 128 filters to reduce my compute because of hardware issues but a scaling encoder will be better ofc) during training i kept on seeing an insane trade off between classes even when correcting for class weights - which i believe is a matter of simply geting more weights to accodomadte all the lower level patterns while depth gets the bigger picture - training for which i havent done.

yet to put a run model submit on kaggle so will update this alst time then!

There were couple of attempts at training and I have maintainted training records of each archeticture for clarity and transparency, in case I can save someone a bit of research and training time - cheers!

