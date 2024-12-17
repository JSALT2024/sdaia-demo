import cv2
import gradio as gr

def process_image(input_image):
    checkpoints_pose = "checkpoints/pose"
    checkpoint_mae = "checkpoints/mae/16-07_21-52-12/checkpoint-440.pth"
    checkpoint_dino = "checkpoints/dino/hand/teacher_checkpoint.pth"
    image_dir = input_image
    db_path = "sign_db.npz"
    db_files_path = "Numerals/Numerals_SaudiSL"
    k = 3
    gen_db = 1
    
    from DatabaseHandler import DatabaseHandler
    handler = DatabaseHandler(checkpoints_pose, checkpoint_mae, checkpoint_dino, db_path, k)
    annotation = handler.predict(db_files_path, db_path, image_dir, gen_db)
    return annotation

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,  # The function to be called
    inputs=[
        gr.Image(type="numpy", sources=["upload", "webcam"], label="Upload an image or take a picture"),
    ],
    outputs=gr.Textbox(label="Predicted Sign"),
    title="Sign Language Recognition - Alphanumerals in SAUDI SL",  # Title of the interface
    description="This demo is a proof of concept for the recognition system of the SAUDI SL. It uses MediaPipe and DINO in the backend, trained on the SAUDI SL dataset from Mohammad Alghannami and Maram Aljuaid."
)

example_images = gr.Markdown(
    """
    ## Example input
    Here are three example input images, you can drag and drop them into the image input window:
    """
)

example_image1 = gr.Image(value='sdaia-demo/img/A.jpg', label="Example Image of A", width=420, height=280)
example_image2 = gr.Image(value='sdaia-demo/img/1.jpg', label="Example Image of 1", width=420, height=280)
example_image3 = gr.Image(value='sdaia-demo/img/2.jpg', label="Example Image of 2", width=420, height=280)

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