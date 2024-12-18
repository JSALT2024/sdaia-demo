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

def process_image(input_image, source):
    print("Processing image...")
    if source == None:
        return f"<div style='font-size: 100px; text-align: center;'>Error</div>", None, None
    else:
        prediction, detected_hand, best_match = handler.predict(input_image, source)
        #if detected_hand is not None:
        #    height, width = detected_hand.shape[:2]
        #    detected_hand = cv2.resize(detected_hand, (width//2, height//2))
        #if best_match is not None:
        #    height, width = best_match.shape[:2]
        #    best_match = cv2.resize(best_match, (width//2, height//2))
        print("Prediction done.")
        return f"<div style='font-size: 100px; text-align: center;'>{prediction}</div>", detected_hand, best_match

def process_input(input_image):
    if input_image == None:
        source = None
    elif "webcam" in input_image:
        source = "webcam"
    else:
        source = "upload"
    
    return process_image(input_image, source)

# Create the Gradio interface
iface = gr.Interface(
    fn=process_input,  # The function to be called
    inputs=[
        gr.Image(type="filepath", sources=["upload", "webcam"], label="Upload an image or take a picture"),
    ],
    outputs=[
        gr.HTML(label="Predicted Sign"),
        gr.Image(label="Detected Hand", type="numpy", width=200, height=200), # Different sized input gives different sized patches
        gr.Image(label="Best Match", type="numpy", width=200, height=200),
    ],
    title="Sign Language Recognition - Alphanumerals in SAUDI SL",  # Title of the interface
    description="This demo is a proof of concept for the recognition system of the SAUDI SL. It uses MediaPipe and DINO in the backend, trained on the SAUDI SL dataset from Mohammad Alghannami and Maram Aljuaid.\n\n" + 
    "You can try signing numerals 1-9 with your right hand in Saudi Sign language."
)

demo = gr.Blocks()

with demo:
    iface.render()
    
    gr.Markdown(
        """
        ## Tutorial: Saudi Sign Language Numerals 1-9
        Below are reference images showing how to sign each numeral:
        """
    )
    
    with gr.Row():
        for i in range(1, 10):
            gr.Image(
                value=f'img/tutorial/{i}.JPG',
                label=f"Numeral {i}",
                show_label=True,
                width=20,
                height=100
            )
    
    gr.Markdown(
        """
        ## Example input
        Here are three example input images, you can drag and drop them into the image input window:
        """
    )
    
    with gr.Row():
        example_image1 = gr.Image(value='img/numeral1.jpg', label="Example Image of 1", width=420, height=280)
        example_image2 = gr.Image(value='img/numeral5.jpg', label="Example Image of 5", width=420, height=280)
        example_image3 = gr.Image(value='img/numeral8.jpg', label="Example Image of 8", width=420, height=280)

if __name__ == "__main__":
    demo.launch()