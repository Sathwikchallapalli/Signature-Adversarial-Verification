## 🚀 Run on Google Colab
<a href="https://colab.research.google.com/drive/1PkmFmONni69H041tQiuRz2KajyN8e1rC">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
</a>

# Characterizing and Evaluating Adversarial Examples in Handwritten Signature Verification

Handwritten signature verification models are increasingly used in digital documentation, banking, and biometric authentication. 
However, deep learning models are vulnerable to **adversarial attacks** — tiny intentionally crafted perturbations that can trick a trained classifier into labeling forged signatures as genuine.

This project investigates:  
✔️ Model accuracy against adversarial examples  
✔️ Impact of FGSM perturbations on model predictions  
✔️ How **adversarial training** improves robustness  

---

## 📌 Problem Statement
Traditional manual signature verification is slow, non-scalable, and prone to human error.  
Deep learning models such as CNNs solve this — but they:

- Misclassify adversarial signatures
- Are vulnerable to pixel-level perturbations
- Fail silently in real biometric systems

### This project shows:
- How adversarial noise fools a ResNet50 classifier
- How adversarial training increases resilience

---

## 🧠 Approach

### 1️⃣ Dataset — CEDAR Signature Dataset
- 55 writers
- 24 genuine + 24 forged per writer
- 2,640 scanned grayscale images
- Benchmark dataset for offline signature verification

> Dataset Source: https://cedar.buffalo.edu/signature/

---

### 2️⃣ Preprocessing Pipeline
To standardize input to ResNet50:

- Grayscale conversion
- Resize to **224×224**
- **OTSU thresholding** (binarization)
- Bitwise inversion
- Convert to 3-channel RGB
- Normalize with `preprocess_input`

---

### 3️⃣ Base Model — ResNet50 (Transfer Learning)
ResNet50 pretrained on ImageNet (frozen)
→ GlobalAveragePooling2D
→ Dropout(0.5)
→ Dense(1, sigmoid)

✔️ Binary classification (genuine vs forged)  
✔️ Dropout reduces overfitting  
✔️ Fine-tuning last 20 layers improves accuracy  

---

## ⚔️ Adversarial Attack — Fast Gradient Sign Method (FGSM)

FGSM creates adversarial sample:
x_adv = x + ε · sign(∇loss)
Where:
- `ε` = perturbation strength
- `sign(gradient)` = direction to increase loss

Produces **imperceptible noise** that changes model predictions.

---

## 🛡️ Adversarial Training
The model is retrained using mixed batches:

- Clean inputs
- FGSM perturbed inputs

> Result: Model learns **robust features** and resists attacks.

---

## 📊 Evaluation Metrics

- Accuracy
- Precision / Recall
- F1 Score
- ROC–AUC
- Misclassification patterns
- Visual inspection (original vs adversarial)

---

# 📂 Project Structure  
Signature-Adversarial-Verification/  
│
├── src/ # Source code  
│ └── signature_verification_adversarial.py  
│
├── docs/ # Documentation  
│ └── Project_Report.pdf  
│
├── assets/ # Images / plots / sample outputs    
│
├── requirements.txt  
├── LICENSE  
└── README.md  
---  

# ▶️ Running Locally

1️⃣ Install dependencies  
pip install -r requirements.txt

2️⃣ Run the projectpython
src/signature_verification_adversarial.py  
Note: This script was developed in Google Colab.  
Local paths may need modification depending on your environment.  

📄 Full Project Report
The complete documentation with diagrams and experimental results is provided here:  
docs/Project_Report.pdf

🧪 Future Improvements
Evaluate stronger adversarial attacks:  
PGD  
DeepFool  
CW  
Train Siamese or Triplet networks for signature embeddings  
Add visual explainability (Grad-CAM)  
Deploy as an API for real-time verification  

👥 Contributors
Challapalli Sathwik
Talasila Revanth
B Sanjeev Roy

📚 References
Goodfellow et al., Explaining and Harnessing Adversarial Examples
He et al., Deep Residual Learning for Image Recognition (ResNet)
Simonyan & Zisserman, Very Deep Convolutional Networks
CEDAR Signature Dataset — https://cedar.buffalo.edu/signature/
TensorFlow FGSM tutorial — https://www.tensorflow.org/tutorials/generative/adversarial_fgsm
