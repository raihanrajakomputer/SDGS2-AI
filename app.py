import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import gradio as gr

ART = Path(__file__).parent / "app_artifacts"
CLASS_NAMES = json.loads((ART / "class_names.json").read_text())
META = json.loads((ART / "metadata.json").read_text())
IMG_SIZE = tuple(META["img_size"])
model = tf.keras.models.load_model(ART / "best_model.keras")

BATAS = 0.50   # skor tertinggi di bawah batas ini dianggap bukan daun singkong


def prediksi(pil_image):
    if pil_image is None:
        return {}, "Unggah foto daun terlebih dahulu."
    arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    arr = tf.image.resize_with_pad(arr, *IMG_SIZE).numpy()
    probs = model.predict(arr[None, ...], verbose=0)[0]
    top = float(probs.max())
    if top < BATAS:
        return {}, f"Bukan termasuk kategori daun singkong! (skor tertinggi {top:.1%}, di bawah batas {BATAS:.0%})"
    skor = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    kelas = CLASS_NAMES[int(probs.argmax())]
    return skor, f"Prediksi: {kelas} (skor {top:.1%}). Ini bukan diagnosis definitif."


with gr.Blocks(title="Klasifikasi Daun Singkong") as demo:
    gr.Markdown("# Klasifikasi Penyakit Daun Singkong (CNN)\nUnggah foto daun singkong, lalu klik Klasifikasikan.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Foto daun", sources=["upload", "webcam"])
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Skor tiap kelas")
            out_msg = gr.Textbox(label="Hasil")
    gr.Button("Klasifikasikan").click(prediksi, inp, [out_label, out_msg])

if __name__ == "__main__":
    demo.launch()
