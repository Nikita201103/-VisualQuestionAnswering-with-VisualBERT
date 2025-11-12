# 🧠 Visual Question Answering (VQA) with VisualBERT

This project implements and evaluates a **Visual Question Answering (VQA)** system using the **VisualBERT** architecture. VQA is a challenging multimodal task that requires a model to understand both an image and a natural language question about that image, and then generate or classify the correct answer.  

The primary goal of this project was to explore the effectiveness of **VisualBERT** in this task, specifically by comparing a **frozen VisualBERT encoder** against a **fully fine-tuned VisualBERT** model.

---

## 📘 Dataset and Preprocessing

### **Dataset**
- **Source:** VQA 2.0 dataset (validation split of MS COCO)  
- **Annotations:** `v2_mscoco_val2014_annotations.json`  
- **Questions:** `v2_OpenEnded_mscoco_val2014_questions.json`  
- **Images:** `val2014/` directory  

### **Preprocessing Steps**

1. **Answer Aggregation:**  
   For each question-image pair, multiple human-provided answers are available. The **most common answer** among these is selected as the ground truth for training to simplify the classification task.

2. **Answer Encoding:**  
   All unique answers are collected, and a **LabelEncoder** converts these categorical strings into numerical IDs, creating a vocabulary of **14,088 unique answers**.

3. **Feature Extraction:**
   - **Visual Features:** Extracted using a **pre-trained ResNet50** (without final classification layer).  
     Each image is resized to **(224, 224)**, normalized using **ImageNet mean and standard deviation**, and encoded into a **2048-dimensional vector**.  
   - **Textual Features:** Processed using a **BERT tokenizer (bert-base-uncased)**.  
     Questions are tokenized into IDs, attention masks, and token type IDs with a **maximum sequence length of 128**.

4. **Caching:**  
   To improve efficiency, all precomputed image and text features (and encoded answers) are saved to disk as `vqa_val_features_pytorch.pkl`.

5. **Data Splitting:**  
   Processed data is split as follows:
   - **Train:** 150,467 samples  
   - **Validation:** 21,128 samples  
   - **Test:** 42,759 samples  
   All questions from the same image are kept in the same split to prevent **data leakage**.

---

## 🧩 Model Architectures and Training Strategy

Two distinct VisualBERT-based models were trained and evaluated:

---

### **Model 1: Frozen VisualBERT + Small Classification Head**

- **VisualBERT Encoder:**  
  Initialized from `bert-base-uncased` with a `visual_embedding_dim` of 2048.  
  All VisualBERT parameters are **frozen** — only the classifier is trainable.
  
- **Classification Head:**  
  A small `nn.Sequential` network consisting of:
  - Linear → ReLU → Dropout → Linear  
  (mapping from 768-dimensional VisualBERT output to `num_answers` classes)

- **Training Details:**
  - Optimizer: **Adam**
  - Learning Rate: `1e-4`
  - Epochs: 10
  - VisualBERT remains in `eval()` mode throughout training (not updated).

---

### **Model 2: Fine-tuned VisualBERT + Classification Head**

- **VisualBERT Encoder:**  
  Same base architecture as Model 1, but **all layers are trainable**.

- **Classification Head:**  
  A slightly **larger** network than Model 1, designed to handle more complex interactions.

- **Training Details:**
  - Optimizer: **Adam**
  - Learning Rate: `2e-5`
  - Epochs: 10
  - Linear LR scheduler with **warmup** to stabilize training.
  - Entire model set to `train()` mode to fine-tune all parameters.

---

## 📊 Results and Discussion

### **Evaluation Metrics**
Both models were evaluated on the test set using:
- **Categorical Accuracy**
- **Macro-Averaged Precision**
- **Macro-Averaged Recall**
- **Macro-Averaged F1-score**

---

### **Observed Performance (After 10 Epochs)**

#### 🔹 Model 1: Frozen VisualBERT + Head
| Metric | Value |
|--------|-------:|
| Test Loss | 3.9324 |
| Categorical Accuracy | 0.2810 |
| Macro Avg Precision | 0.0012 |
| Macro Avg Recall | 0.0015 |
| Macro Avg F1-score | 0.0010 |

#### 🔹 Model 2: Fine-tuned VisualBERT + Head
| Metric | Value |
|--------|-------:|
| Test Loss | 3.7551 |
| Categorical Accuracy | 0.2702 |
| Macro Avg Precision | 0.0004 |
| Macro Avg Recall | 0.0010 |
| Macro Avg F1-score | 0.0005 |

---

### **Insights**
- Surprisingly, **Model 1 (Frozen VisualBERT)** achieved slightly **better accuracy** and macro-averaged metrics than the fine-tuned version.
- **Possible Reasons:**
  - **Limited Training Epochs:**  
    Fine-tuning a large model like VisualBERT often requires **more epochs** and **careful hyperparameter tuning**. The model may still have been stabilizing after 10 epochs.
  - **Catastrophic Forgetting:**  
    Without strategies like discriminative learning rates or gradual unfreezing, fine-tuning can lead to an initial **performance drop** as the model “forgets” pre-trained knowledge.
  - **Classification Head Complexity:**  
    The larger head in Model 2 might not have benefited from limited training data and epochs.
  - **Class Imbalance:**  
    The VQA dataset is **highly imbalanced**, dominated by frequent answers (e.g., “yes”, “no”, “2”, “person”), resulting in very low macro-averaged metrics for rare answers.

---

## 🧠 Conclusion

This project demonstrates the complete setup for **Visual Question Answering (VQA)** using **VisualBERT**, including:
- Data preprocessing
- Feature extraction
- Model construction
- Training and evaluation

While the frozen model slightly outperformed the fine-tuned version in this short training run, these results highlight the importance of:
- Longer training durations  
- Better fine-tuning strategies (e.g., layer-wise learning rates)  
- Techniques to mitigate **class imbalance**  

These experiments serve as a **foundational benchmark**, illustrating the initial behavior of VisualBERT in the VQA domain and setting the stage for further improvements.

---

## ⚙️ Setup Instructions

### **Dependencies**
Install all required libraries:
```bash
pip install torch torchvision transformers scikit-learn numpy pandas matplotlib pillow tqdm
```

### **Dataset Preparation**
1. Download the **VQA 2.0 Validation Split**:
   - Images: `val2014/`
   - Questions: `v2_OpenEnded_mscoco_val2014_questions.json`
   - Annotations: `v2_mscoco_val2014_annotations.json`
2. Adjust dataset paths in the notebook accordingly.

### **Execution**
1. Clone this repository.  
2. Open and run the Jupyter Notebook (`vqa_visualbert.ipynb`).  
3. Preprocessing automatically generates cached features in `vqa_val_features_pytorch.pkl`.

---

---

## 🧰 Tech Stack
- **Model:** VisualBERT (Hugging Face Transformers)
- **Backbone:** BERT-base-uncased
- **Vision Encoder:** ResNet50 (Image feature extractor)
- **Frameworks:** PyTorch, Transformers, Scikit-learn
- **Dataset:** VQA 2.0 (MS COCO validation split)

---

---
