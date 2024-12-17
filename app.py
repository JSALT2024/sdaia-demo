import cv2
import gradio as gr
from DatabaseHandler import DatabaseHandler

# Initialize handler globally
print("Initializing Gradio, loading models...")
db_path = "sign_db.npz" # features database
checkpoints_pose = "data/pose"
checkpoint_dino = "data/dino/hand/teacher_checkpoint.pth"
handler = DatabaseHandler(checkpoints_pose, checkpoint_dino, db_path) # backend
handler.load_models()
print("Models loaded.")

def process_image(input_image):
    print("Processing image...")
    prediction = handler.predict(input_image)
    print("Prediction done.")
    return f"<div style='font-size: 300px; text-align: center;'>{prediction}</div>"

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,  # The function to be called
    inputs=[
        gr.Image(type="numpy", sources=["upload", "webcam"], label="Upload an image or take a picture"),
    ],
    outputs=gr.HTML(label="Predicted Sign"), 
    title="Sign Language Recognition - Alphanumerals in SAUDI SL",  # Title of the interface
    description="This demo is a proof of concept for the recognition system of the SAUDI SL. It uses MediaPipe and DINO in the backend, trained on the SAUDI SL dataset from Mohammad Alghannami and Maram Aljuaid."
)

example_images = gr.Markdown(
    """
    ## Example input
    Here are three example input images, you can drag and drop them into the image input window:
    """
)

example_image1 = gr.Image(value='sdaia-demo/img/numeral1.jpg', label="Example Image of 1", width=420, height=280)
example_image2 = gr.Image(value='sdaia-demo/img/numeral5.jpg', label="Example Image of 5", width=420, height=280)
example_image3 = gr.Image(value='sdaia-demo/img/numeral8.jpg', label="Example Image of 8", width=420, height=280)

# Combine the interface and example images
app = gr.Blocks()

with app:
    iface.render()
    example_images.render()
    with gr.Row():
        example_image1.render()
        example_image2.render()
        example_image3.render()

# Launch the interface
if __name__ == "__main__":
    app.launch()