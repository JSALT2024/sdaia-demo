from assets.predict_pose import predict_pose, create_mediapipe_models
from assets.pose_utils import crop_pad_image
from assets.crop_frames import crop_frame
from assets.predict_mae import *
from assets.predict_dino import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import cv2

# Define the BackendRunner class to handle various backend operations
class BackendRunner:
    def __init__(self, checkpoint_pose, checkpoint_mae, checkpoint_dino):
        # Initialize with checkpoints for pose, MAE, and DINO models
        self.checkpoint_pose = checkpoint_pose
        self.checkpoint_mae = checkpoint_mae
        self.checkpoint_dino = checkpoint_dino
        self.models_pose = None
        self.model_mae = None
        self.model_dino_hand = None
        # Set the device to GPU if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_video_frames(self, video_path):
        # Load video frames from the specified path
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Check if the video file can be opened
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        # Read frames from the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        # Release the video capture object
        cap.release()

        return np.array(frames)

    def load_models(self):
        self.models_pose = create_mediapipe_models(self.checkpoint_pose)
        arch = 'vit_base_patch16'
        self.model_mae = create_mae_model(arch, self.checkpoint_mae)
        self.model_dino_hand = create_dino_model(self.checkpoint_dino)

    def pose_video(self, video_dir):
        # Predict pose for a video
        video = self.load_video_frames(video_dir)
        prediction = predict_pose(video, self.models_pose, 4)
        return prediction

    def pose_img(self, image_dir):
        # Predict pose for a single image
        image = cv2.imread(image_dir)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = [rgb_image]
        prediction = predict_pose(image, self.models_pose, 4)
        return prediction

    def mae(self, images):
        # Generate MAE embeddings for a list of images
        self.model_mae = self.model_mae.to(self.device)

        mae_embeddings = []
        for image in images:
            mae_embedding = mae_predict(image, self.model_mae, transform_mae, self.device)
            mae_embeddings.append(mae_embedding)
        return mae_embeddings

    def dino(self, pose_output, lhand=0):
        # Generate DINO embeddings for pose output
        self.model_dino_hand.to(self.device)

        rhand_embeddings = []
        for i in range(len(pose_output["images"])):
            right_features = dino_predict(pose_output["cropped_right_hand"][i], self.model_dino_hand, transform_dino, self.device)
            rhand_embeddings.append(right_features)

        if lhand:
            lhand_embeddings = []
            for i in range(len(pose_output["images"])):
                left_features = dino_predict(pose_output["cropped_left_hand"][i], self.model_dino_hand, transform_dino, self.device)
                lhand_embeddings.append(left_features)
            features = np.hstack((lhand_embeddings, rhand_embeddings))
            return np.squeeze(features)
        else:
            return np.squeeze(rhand_embeddings)

    def similarity(self, emb_1, emb_2):
        # Calculate cosine similarity between two embeddings
        sim = cosine_similarity(emb_1.reshape(1, -1), emb_2.reshape(1, -1))[0][0]
        return sim

if __name__ == "__main__":
    # Define paths to checkpoints and image input
    checkpoints_pose = "checkpoints/pose"
    checkpoint_mae = "checkpoints/mae/16-07_21-52-12/checkpoint-440.pth"
    checkpoint_dino = "checkpoints/dino/hand/teacher_checkpoint.pth"
    image_dir = "PoseEstimation/data/clips/img_example.jpg"

    # Create an instance of BackendRunner
    runner = BackendRunner(checkpoints_pose, checkpoint_mae, checkpoint_dino)
    # Load models
    runner.load_models()
    # Perform pose prediction on an image and crop
    pose_output = runner.pose_img(image_dir)
    # Generate MAE embeddings from cropped images
    mae_embeddings = runner.mae(pose_output["cropped_images"])
    # Generate DINO embeddings from pose output
    dino_embeddings = runner.dino(pose_output, 0)
    
    # Calculate similarity between embeddings
    sim = runner.similarity(dino_embeddings, dino_embeddings)