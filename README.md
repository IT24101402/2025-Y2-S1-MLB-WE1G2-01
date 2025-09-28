# 🍃 Tea Plantation Disease Detection

Tea Plantation Disease Detection is an agriculture-focused AI project aimed at supporting **precision farming** in Sri Lanka. Tea, being one of the country’s top exports, is vulnerable to several leaf diseases that impact yield and export quality. This project helps farmers by automatically identifying diseases in tea leaves using machine learning and image classification techniques.

---

## 📖 Project Overview

Leaf diseases like **Blister Blight**, **Red Rust**, and **Brown Blight** can drastically reduce tea crop productivity. This system enables early detection by allowing farmers to simply take a photo of a tea leaf, and get an instant classification of whether it's healthy or infected.

---

## 🌱 Real-World Benefits

- ✅ Reduces crop loss and improves farmer income.
- 🌾 Promotes sustainable farming via targeted pesticide usage.
- 🌍 Enhances tea quality to meet global export standards.

---

## ⚙️ Core Approach

- 📷 Image identification and classification via machine learning.
- 🤖 Automated disease detection from smartphone-captured images.
- 📊 Instant feedback for timely, informed decision-making by farmers.

---

## 🎯 Expected Outcome

A **reliable AI model** with **high accuracy**, capable of identifying multiple tea leaf diseases, and deployable in real-world tea plantation environments.

---

## 📊 Dataset

- **Source:** Kaggle Tea Leaf Disease Dataset  
- **Total Images:** 32,421  
- **Image Classes:**

| Class Name    | Number of Images |
|---------------|------------------|
| Algal Spot    | 5,497            |
| Brown Blight  | 4,908            |
| Gray Blight   | 5,537            |
| Healthy       | 5,492            |
| Helopeltis    | 5,482            |
| Red Spot      | 5,505            |
| **Total**     | **32,421**       |

- **Image Formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`

- **Folder Structure:**

---

## 🔧 Preprocessing Pipeline

The dataset undergoes the following preprocessing:

1. **Handling corrupted data** – Remove unreadable images.
2. **Image resizing** – Resize all images to 224×224.
3. **Image normalization** – Scale pixel values.
4. **Color conversion** – Convert all to RGB format.
5. **Class balancing** – Address imbalance via augmentation.
6. **Data split & augmentation** – Apply transformations and split into train/val/test sets.

---

## 👥 Group Members & Responsibilities

| IT Number      | Name                     | Preprocessing Technique         |
|----------------|--------------------------|---------------------------------|
| IT24101352     | Kumarawansha O.A.        | Data split & augmentation       |
| IT24100855     | Ranaweera R.K.D.D.N.     | Color conversion                |
| IT24100858     | Wijesinghe D.T.D.        | Image normalization             |
| IT24101402     | Kumarasiri H.T.D.        | Handling corrupted data         |
| IT24100811     | Wijedasa H.A.W.R.        | Image resizing                  |
| IT24100705     | Gunarathna A.A.S.R.      | Class balancing                 |

---

## 🏃 How to Run the Code

Follow these steps to run the project locally:

### 1️⃣ Clone the Repository
```bash
git clone git@github.com:IT24101402/2025-Y2-S1-MLB-WE1G2-01.git
cd tea-disease-detection

 
