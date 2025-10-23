import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import pathlib

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ---------------------------
# Paths - Update this BASE_DIR to your dataset path
# ---------------------------
BASE_DIR = r"C:\Users\Danidu Wijesinghe\Desktop\IT24100858\Final_Preprocessed_Dataset-20251022T235629Z-1-001\Final_Preprocessed_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
OUTPUT_DIR = 'training_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Parameters
# ---------------------------
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
NUM_FOLDS = 3
EPOCHS = 10

# ---------------------------
# Utility Function: Build DataFrame from Directory
# ---------------------------
def build_dataframe_from_dir(root_dir):
    rows = []
    root = pathlib.Path(root_dir)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    for cls in classes:
        cls_path = root / cls
        for img in cls_path.rglob('*'):
            if img.is_file():
                rows.append({'filepath': str(img.resolve()), 'label': cls})
    df = pd.DataFrame(rows)
    return df

# ---------------------------
# Prepare DataFrames
# ---------------------------
train_df = build_dataframe_from_dir(TRAIN_DIR)
test_df = build_dataframe_from_dir(TEST_DIR)

labels = sorted(train_df['label'].unique())
label_to_idx = {lab: idx for idx, lab in enumerate(labels)}
train_df['label_idx'] = train_df['label'].map(label_to_idx)
test_df['label_idx'] = test_df['label'].map(label_to_idx)
NUM_CLASSES = len(labels)

print(f"Classes found: {labels}")
print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

# ---------------------------
# Model Builders
# ---------------------------
def build_transfer_model(backbone_name='resnet50', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                         base_trainable_layers=20, dropout_rate=0.4, dense_units=256):

    if backbone_name == 'mobilenetv2':
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'resnet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported backbone")

    # Freeze all layers first
    for layer in base.layers:
        layer.trainable = False
    # Unfreeze last n layers
    for layer in base.layers[-base_trainable_layers:]:
        layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    return model

def build_custom_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), dropout_rate=0.4):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# ---------------------------
# Create ImageDataGenerators
# ---------------------------
def create_generators(train_df_subset, valid_df_subset, aug_params, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(rescale=1./255, **aug_params)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_dataframe(
        train_df_subset, x_col='filepath', y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical', batch_size=batch_size, shuffle=True, seed=SEED
    )
    val_gen = val_datagen.flow_from_dataframe(
        valid_df_subset, x_col='filepath', y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical', batch_size=batch_size, shuffle=False
    )
    return train_gen, val_gen

# ---------------------------
# Training Function
# ---------------------------
def train_and_evaluate(variety):
    print(f"\nTraining variety: {variety['name']}")

    # Model creation
    if variety['family'] in ['resnet50', 'mobilenetv2']:
        model = build_transfer_model(
            backbone_name=variety['family'],
            base_trainable_layers=variety.get('base_trainable_layers', 20),
            dropout_rate=variety.get('dropout_rate', 0.4),
            dense_units=variety.get('dense_units', 256)
        )
    else:
        model = build_custom_cnn(dropout_rate=variety.get('dropout_rate', 0.4))

    # Compile model
    opt = optimizers.Adam(learning_rate=variety.get('lr', 1e-3))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    X = train_df['filepath'].values
    y = train_df['label_idx'].values
    fold = 1
    acc_scores = []

    for train_idx, val_idx in skf.split(X, y):
        print(f"\n--- Fold {fold}/{NUM_FOLDS} ---")
        train_subset = train_df.iloc[train_idx]
        val_subset = train_df.iloc[val_idx]

        train_gen, val_gen = create_generators(train_subset, val_subset, variety.get('augmentation', {}))

        # Callbacks
        ckpt_path = os.path.join(OUTPUT_DIR, f"{variety['name']}_fold{fold}.keras")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3),
            ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss')
        ]

        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, verbose=2)
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        acc_scores.append(val_acc)
        fold += 1

    avg_acc = np.mean(acc_scores)
    print(f"\nAverage Accuracy for {variety['name']}: {avg_acc:.4f}")

# ---------------------------
# Varieties List
# ---------------------------
VARIETIES = [
    {
        'name': 'ResNet50_var1',
        'family': 'resnet50',
        'base_trainable_layers': 10,
        'dropout_rate': 0.4,
        'dense_units': 256,
        'lr': 5e-5,
        'augmentation': {'rotation_range': 20, 'zoom_range': 0.1, 'horizontal_flip': True}
    },
    {
        'name': 'MobileNetV2_var1',
        'family': 'mobilenetv2',
        'base_trainable_layers': 20,
        'dropout_rate': 0.4,
        'dense_units': 256,
        'lr': 5e-5,
        'augmentation': {'rotation_range': 15, 'zoom_range': 0.15, 'horizontal_flip': True}
    },
    {
        'name': 'CustomCNN_var1',
        'family': 'custom',
        'dropout_rate': 0.4,
        'lr': 1e-3,
        'augmentation': {'rotation_range': 10, 'horizontal_flip': True}
    }
]

# ---------------------------
# Run Training for All Varieties
# ---------------------------
for variety in VARIETIES:
    train_and_evaluate(variety)

print("\nAll training complete. Models saved in 'training_outputs' folder.")


