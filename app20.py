import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from PIL import Image
import tempfile
import os
import shutil
import zipfile

# --- App Configuration ---
st.set_page_config(page_title="Image Classifier", layout="wide")

st.title("🧠 Deep Learning on Bulk Images")
st.write("Upload training and validation image ZIPs. Train and test a CNN image classifier.")

# --- Utility Functions ---
def load_dataset(zip_file, img_size, batch_size):
    # Extract ZIP to a persistent temp folder
    temp_dir = os.path.join("temp_data", os.path.splitext(zip_file.name)[0])
    os.makedirs(temp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        temp_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )
    return dataset


def normalize_ds(ds):
    norm_layer = layers.Rescaling(1./255)
    return ds.map(lambda x, y: (norm_layer(x), y))

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- File Upload ---
col1, col2 = st.columns(2)

with col1:
    train_zip = st.file_uploader("📂 Upload Training Images (ZIP)", type=["zip"])
with col2:
    val_zip = st.file_uploader("📂 Upload Validation Images (ZIP)", type=["zip"])

# --- Training Parameters ---
with st.sidebar:
    st.header("⚙️ Model Settings")
    img_size = st.selectbox("Image Size", [(64, 64), (128, 128)], index=1)
    batch_size = st.slider("Batch Size", 8, 64, 32, step=8)
    epochs = st.slider("Epochs", 1, 20, 5)

# --- Training ---
model = None
if train_zip and val_zip and st.button("🚀 Train Model"):
    with st.spinner("Loading data..."):
        train_ds = load_dataset(train_zip, img_size, batch_size)
        val_ds = load_dataset(val_zip, img_size, batch_size)
        class_names = train_ds.class_names
        train_ds = normalize_ds(train_ds)
        val_ds = normalize_ds(val_ds)

    st.success(f"Loaded {len(class_names)} classes: {class_names}")
    
    with st.spinner("Training model..."):
        model = build_model(input_shape=img_size + (3,), num_classes=len(class_names))
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[es])
        st.success("✅ Training complete!")

    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        model.save(tmp.name)
        st.download_button("📥 Download Trained Model", data=open(tmp.name, "rb"), file_name="image_model.h5")

# --- Prediction ---
st.markdown("---")
st.header("🔍 Test with Uploaded Image")
test_img = st.file_uploader("Upload an Image for Prediction", type=["jpg", "jpeg", "png"])

if test_img and model:
    image = Image.open(test_img).convert("RGB").resize(img_size)
    st.image(image, caption="Uploaded Image", width=200)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, 0)

    pred = model.predict(img_array)[0]
    pred_idx = np.argmax(pred)
    pred_class = class_names[pred_idx]
    confidence = pred[pred_idx]

    st.markdown(f"### 🧠 Predicted: `{pred_class}` ({confidence*100:.2f}% confidence)")
elif test_img and not model:
    st.warning("Please train the model before testing.")
