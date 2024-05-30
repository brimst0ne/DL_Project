# Ackownledgement: https://huggingface.co/spaces/kadirnar/Yolov10/blob/main/app.py
# Thanks to @kadirnar

import os
import gradio as gr
import cv2
import sys
import numpy as np
from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import mean_squared_error as mse
from ultralytics import YOLOv10

def compare_pics():
    image1 = cv2.imread("./testimage/image0.jpg")
    image2 = cv2.imread("./testimage/output_image.jpg")
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    m = mse(gray1, gray2)
    s = ssim(gray1, gray2)
    return m, s

def mse(image1, image2):
    diff = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    diff /= float(image1.shape[0] * image1.shape[1])
    return diff

def copy_to_folder(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_path = os.path.join('./testimage', 'output_image.jpg')
    cv2.imwrite(output_path, image_bgr)
    return f"Image saved to {output_path}"

def yolov10_inference(image, model_path, image_size, conf_threshold):
    model = YOLOv10(model_path)
    
    model.predict(source=image, imgsz=image_size, conf=conf_threshold, save=True)
    
    return model.predictor.plotted_img[:, :, ::-1]

def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image")
                
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov10n.pt",
                        "yolov10s.pt",
                        "yolov10m.pt",
                        "yolov10b.pt",
                        "yolov10l.pt",
                        "yolov10x.pt",
                    ],
                    value="yolov10s.pt",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.25,
                )
                yolov10_infer = gr.Button(value="Detect Objects")
                copybtn = gr.Button(value="Copy image to testing folder")
                checkbtn = gr.Button(value="Test generated image")
                mse_output = gr.Textbox(label="MSE")
                ssim_output = gr.Textbox(label="SSIM")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image")

        copybtn.click(
            fn=copy_to_folder,
            inputs=[output_image],
            outputs=[gr.Textbox(label="Copy Status")]
        )

        checkbtn.click(
            fn=compare_pics,
            outputs=[mse_output, ssim_output]
        )

        yolov10_infer.click(
            fn=yolov10_inference,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/proger.jpg",
                    "yolov10s.pt",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/guitarist.jpg",
                    "yolov10s.pt",
                    640,
                    0.25,
                ],
            ],
            fn=yolov10_inference,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples=False,
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(debug=True)