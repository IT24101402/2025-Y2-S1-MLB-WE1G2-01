📖 Project Overview
Tea Plantation Disease Detection is an agriculture-focused project designed to support precision farming. Tea is one of Sri Lanka’s top exports, but leaf diseases such as Blister Blight, Red Rust, and Brown Blight can cause severe yield losses. Detecting these diseases early is crucial to prevent large-scale infestations and maintain the quality of export products.
This project leverages image-based machine learning models to automatically classify tea leaf diseases. Farmers can capture photos of tea leaves using a smartphone, and the system provides instant classification results, indicating whether the leaf is healthy or infected.
🌱 Real-World Benefits
• Reduces crop loss and improves farmer income.
• Promotes sustainable farming by ensuring pesticides are only used when necessary.
• Enhances tea quality to meet international market standards.

⚙️ Core Approach
• Image identification and classification using machine learning.
• Automated disease detection from captured tea leaf images.
• Provides outputs that assist farmers in making timely decisions.

🎯 Expected Outcome
A reliable AI model with high accuracy capable of identifying multiple tea leaf diseases, enabling practical, real-world deployment in tea plantations.

📊 Dataset
This project uses a Tea Leaf Disease Dataset sourced from Kaggle, containing 32,421 images across 6 classes. The dataset covers multiple disease types as well as healthy leaves, enabling robust model training and evaluation.

📂 Classes & Distribution
Class Name        Number of Images
Algal Spot              5,497
Brown Blight            4,908
Gray Blight             5,537
Healthy                 5,492
Helopeltis              5,482
Red Spot                5,505

Total: 32,421 images

🖼️ File Details
• Formats: .jpg, .jpeg, .png, .bmp, .webp, .tif, .tiff
• Source: Kaggle Tea Leaf Disease Dataset
• Structure:
• dataset/
• ├── algal_spot/
• ├── brown_blight/
• ├── gray_blight/
• ├── healthy/
• ├── helopeltis/
• └── red_spot/

🔧 Preprocessing Pipeline
The dataset underwent the following preprocessing steps before model training:
1. Handling corrupted data – Removed unreadable or invalid images.
2. Image resizing – Standardized all images to a fixed input size (e.g., 224×224) for model compatibility.
3. Image normalization – Scaled pixel values to [0,1] or [-1,1] for faster convergence.
4. Color conversion – Converted all images to RGB format for consistency.
5. Class balancing – Addressed class imbalance by using augmentation or resampling.
6. Data split & augmentation – Split dataset into training, validation, and testing sets, with augmentations (rotation, flipping, brightness adjustment, etc.) applied to improve generalization.

Group Members & Roles

IT number                    Name                  Preprocessing Technique

IT24101352               Kumarawansha O.A.          Data split & augmentation 
IT24100855               Ranaweera R.K.D.D.N.       Color conversion 
IT24100858               Wijesinghe D.T.D.          Image normalization 
IT24101402               Kumarasiri H.T.D.          Handling corrupted data 
IT24100811               Wijedasa H.A.W.R.          Image resizing 
IT24100705               Gunarathna A.A.S.R.        Class balancing 
 

 🏃 How to Run the Code

Follow these steps to set up and run the project locally:

1️⃣ Clone the Repository

git clone https://github.com/your-username/tea-disease-detection.git

cd tea-disease-detection

2️⃣ Open in PyCharm

• Open PyCharm (or any IDE that supports Jupyter Notebooks).
• Load the cloned project folder.
• Make sure the Python interpreter is set up with Jupyter support.
• Install related libraries
3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Explore Individual Preprocessing Notebooks

Each preprocessing technique is implemented in a separate Jupyter Notebook:

• 01_handle_corrupted_data.ipynb
• 02_image_resizing.ipynb
• 03_image_normalisation.ipynb
• 04_colour_conversion.ipynb
• 05_class_balancing.ipynb
• 06_data_split_and_augmentation.ipynb
👉 You can open and run these notebooks one by one to understand how each preprocessing step works.

5️⃣ Run the Complete Pipeline

The final group pipeline notebook combines all preprocessing techniques in the correct order:

• Open group_pipeline.ipynb in PyCharm.
• Run all cells to execute the full preprocessing workflow on the dataset.
 
