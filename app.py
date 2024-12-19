import cv2
import gradio as gr
from DatabaseHandler import DatabaseHandler

# Initialize handler globally
print("Initializing Gradio, loading models...")
db_path = "sign_db.npz" # features database
checkpoints_pose = "data/pose"
checkpoint_dino = "data/dino/hand/teacher_checkpoint.pth"
data_db_path = "data/Numerals/Numerals_database_saudi"
handler = DatabaseHandler(checkpoints_pose, checkpoint_dino, db_path, data_db_path) # backend
handler.load_models()
print("Models loaded.")

def process_image(input_image):
    if input_image == None:
        source = None
    elif "webcam" in input_image:
        source = "webcam"
    else:
        source = "upload"
    
    print("Processing image...")
    if source == None:
        return f"<div style='font-size: 50px; text-align: center;'>Please pass your image.</div>", None
    else:
        prediction, detected_hand = handler.predict(input_image, source)
        print("Prediction done.")
        return f"<div style='font-size: 100px; text-align: center;'>{prediction}</div>", detected_hand

iface = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(
        type="filepath", 
        sources=["upload", "webcam"], 
        label="Upload an image or take a picture",
        elem_id="input-image"
    )],
    outputs=[
        gr.HTML(label="Predicted Sign"),
        gr.Image(label="Detected Hand", type="numpy", width=200, height=200),
    ]
)

demo = gr.Blocks(css="""
    .tutorial-image {
        width: 7% !important;  /* Ensures images are small enough to fit in one line */
        display: inline-block;
        margin: 0;
        padding: 0;
        text-align: center;
    }
    .tutorial-row {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        justify-content: flex-start; /* Align items to the start */
        width: 100%;
        overflow-x: auto; /* Allow horizontal scrolling if needed */
        padding: 0;
        gap: 0; /* Remove spacing between items */
    }
    .tutorial-row .label {
        font-size: 12px !important;
        text-align: center;
        margin-top: 2px;
    }
    #input-image img {
        max-height: 300px; /* Limit the height of uploaded images */
        width: auto;      /* Keep aspect ratio */
    }
""")


with demo:
    try:
        # Title and description
        gr.Markdown("<h1 style='text-align: center; font-size: 2.5em'>Sign Language Recognition - Alphanumerals in SAUDI SL</h1>")
        gr.Markdown(
            "This demo is a proof of concept for the recognition system of the SAUDI SL. "
            "It uses MediaPipe and DINO in the backend, trained on the SAUDI SL dataset "
            "from Mohammad Alghannami and Maram Aljuaid.\n\n"
            "You can try signing numerals 1-9 with your right hand in Saudi Sign language as shown below:"
        )

        with gr.Row(elem_classes="tutorial-row"):
            for i in range(1, 10):
                gr.Image(
                    value=f'img/tutorial/{i}.JPG',
                    label=str(i),
                    show_label=True,
                    container=True,
                    elem_id=f"tutorial-img-{i}",
                    elem_classes="tutorial-image"
                )

        iface.render()

        gr.Markdown(
            """
            ## Example input
            Here are three example input images, you can drag and drop them into the image input window:
            """
        )
        
        with gr.Row():
            gr.Image(value='img/numeral1.jpg', label="Example Image of 1", width=420, height=280)
            gr.Image(value='img/numeral2.jpg', label="Example Image of 2", width=420, height=280)
            gr.Image(value='img/numeral3.jpg', label="Example Image of 3", width=420, height=280)
                
    except Exception as e:
        print(f"Error during interface creation: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        demo.launch()
    except Exception as e:
        print(f"Error during launch: {str(e)}")
        raise