from assets.predict_pose import predict_pose, create_mediapipe_models
from assets.pose_utils import crop_pad_image
from assets.crop_frames import crop_frame
from assets.predict_mae import *
from assets.predict_dino import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import cv2

class BackendRunner:
    def __init__(self, checkpoint_pose, checkpoint_mae, checkpoint_dino):
        self.checkpoint_pose = checkpoint_pose
        self.checkpoint_mae = checkpoint_mae
        self.checkpoint_dino = checkpoint_dino
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        cap.release()

        return np.array(frames)

    def pose_video(self, video_dir):
        models = create_mediapipe_models(self.checkpoint_pose)
        video = self.load_video_frames(video_dir)
        prediction = predict_pose(video, models, 4)
        return prediction

    def pose_img(self, image_dir):
        models = create_mediapipe_models(self.checkpoint_pose)
        image = cv2.imread(image_dir)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video = [rgb_image]
        prediction = predict_pose(video, models, 4)
        return prediction

    def mae(self, images):
        arch = 'vit_base_patch16'
        model = create_mae_model(arch, self.checkpoint_mae)
        model = model.to(self.device)

        mae_embeddings = []
        for image in images:
            mae_embedding = mae_predict(image, model, transform_mae, self.device)
            mae_embeddings.append(mae_embedding)
        return mae_embeddings

    def dino(self, pose_output, lhand=0):
        hand_model = create_dino_model(self.checkpoint_dino)
        hand_model.to(self.device)

        rhand_embeddings = []
        for i in range(len(pose_output["images"])):
            right_features = dino_predict(pose_output["cropped_right_hand"][i], hand_model, transform_dino, self.device)
            rhand_embeddings.append(right_features)

        if lhand:
            lhand_embeddings = []
            for i in range(len(pose_output["images"])):
                left_features = dino_predict(pose_output["cropped_left_hand"][i], hand_model, transform_dino, self.device)
                lhand_embeddings.append(left_features)
            features = np.hstack((lhand_embeddings, rhand_embeddings))
            return np.squeeze(features)
        else:
            return np.squeeze(rhand_embeddings)

    def similarity(self, emb_1, emb_2):
        sim = cosine_similarity(emb_1.reshape(1, -1), emb_2.reshape(1, -1))[0][0]
        return sim

if __name__ == "__main__":
    checkpoints_pose = "checkpoints/pose"
    checkpoint_mae = "checkpoints/mae/16-07_21-52-12/checkpoint-440.pth"
    checkpoint_dino = "checkpoints/dino/hand/teacher_checkpoint.pth"
    image_dir = "PoseEstimation/data/clips/img_example.jpg"

    runner = BackendRunner(checkpoints_pose, checkpoint_mae, checkpoint_dino)
    pose_output = runner.pose_img(image_dir)
    mae_embeddings = runner.mae(pose_output["cropped_images"])
    dino_embeddings = runner.dino(pose_output, 0)

    sim = runner.similarity(dino_embeddings, dino_embeddings)