import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image

ART = Path(__file__).parent / "app_artifacts"
IMG_SIZE = (224, 224)

# Batas skor: di bawah angka ini, foto dianggap bukan daun singkong.
# Ubah angkanya setelah diuji dengan foto nyata.
BATAS_SKOR = 0.60

CLASS_NAMES = json.loads((ART / "class_names.json").read_text(encoding="utf-8"))
NUM_CLASSES = len(CLASS_NAMES)
model = keras.models.load_model(ART / "best_model.keras")


def get_gradcam_heatmap(model, batch):
    batch = tf.convert_to_tensor(batch, dtype=tf.float32)
    if any(layer.name == "efficientnetb0" for layer in model.layers):
        base = model.get_layer("efficientnetb0")
        probe = keras.Model(base.inputs, [base.get_layer("top_conv").output, base.output])
        with tf.GradientTape() as tape:
            x = model.get_layer("augmenter")(batch, training=False)
            fmap, features = probe(x, training=False)
            pooled = model.get_layer("pool")(features)
            pooled = model.get_layer("classifier_dropout")(pooled, training=False)
            logits = model.get_layer("classifier_logits")(pooled)
            class_idx = tf.argmax(logits[0])
            score = tf.gather(logits[0], class_idx)
        grads = tape.gradient(score, fmap)
    else:
        probe = keras.Model(model.inputs, [
            model.get_layer("conv_3").output,
            model.get_layer("classifier_logits").output])
        with tf.GradientTape() as tape:
            fmap, logits = probe(batch, training=False)
            class_idx = tf.argmax(logits[0])
            score = tf.gather(logits[0], class_idx)
        grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2))[0]
    heat = tf.reduce_sum(fmap[0] * weights, axis=-1)
    heat = tf.nn.relu(heat)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy(), int(class_idx.numpy())


def overlay_gradcam(image_224, heatmap, alpha=0.40):
    original = np.clip(image_224, 0, 255).astype("uint8")
    resized = tf.image.resize(heatmap[..., None], IMG_SIZE).numpy()[..., 0]
    colors = (plt.get_cmap("jet")(resized)[..., :3] * 255).astype("uint8")
    return ((1 - alpha) * original.astype(float) + alpha * colors.astype(float)).astype("uint8")


def classify_leaf(pil_image):
    if pil_image is None:
        return {}, None, "Unggah foto daun terlebih dahulu."
    arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    arr = tf.image.resize_with_pad(arr, *IMG_SIZE).numpy()
    probs = model.predict(arr[None, ...], verbose=0)[0]
    top = float(probs.max())

    # Pagar: skor terlalu rendah berarti foto ditolak
    if top < BATAS_SKOR:
        message = (f"Bukan termasuk kategori daun singkong! "
                   f"(skor tertinggi {top:.1%}, di bawah batas {BATAS_SKOR:.0%}). "
                   "Jika ini memang daun singkong, coba unggah foto yang lebih jelas.")
        return {}, None, message

    heat, predicted = get_gradcam_heatmap(model, arr[None, ...])
    overlay = overlay_gradcam(arr, heat)
    message = (f"Prediksi: {CLASS_NAMES[predicted]}; skor softmax: {top:.1%}. "
               "Ini bukan diagnosis definitif.")
    scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    return scores, Image.fromarray(overlay), message


with gr.Blocks(title="SDG 2 - Klasifikasi Daun Singkong") as demo:
    gr.Markdown("# Klasifikasi Gangguan Kesehatan Daun Singkong")
    gr.Markdown("Proyek AI SDG 2. Hasil model bukan diagnosis pertanian definitif.")
    with gr.Row():
        image_in = gr.Image(type="pil", label="Foto daun", sources=["upload"])
        heat_out = gr.Image(type="pil", label="Grad-CAM")
    prob_out = gr.Label(label="Probabilitas kelas", num_top_classes=5)
    note_out = gr.Textbox(label="Interpretasi", interactive=False)
    gr.Button("Klasifikasikan").click(
        classify_leaf, inputs=image_in, outputs=[prob_out, heat_out, note_out])

if __name__ == "__main__":
    demo.launch()
