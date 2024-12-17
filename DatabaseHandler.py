from BackendRunner import BackendRunner
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from itertools import groupby
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
import os
import re

class DatabaseHandler(BackendRunner):
    def __init__(self, checkpoint_pose, checkpoint_dino, db_path, k=11):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_pose = checkpoint_pose
        self.checkpoint_dino = checkpoint_dino
        self.models_pose = None
        self.model_dino_hand = None
        self.k = k
        self.db_path = db_path
    
    def gen_database(self, folderpath, db_path):
        feature_database = {}
        for filename in os.listdir(folderpath):
            file_path = os.path.join(folderpath, filename)
            if os.path.isfile(file_path) and filename.lower().endswith('.jpg'):
                file_features = self.get_features(file_path)
                feature_database[filename] = file_features
        
        np.savez(db_path, feature_database)
    
    def load_db(self, db_path):
        data = np.load(db_path, allow_pickle=True)
        feature_database = {key: data[key].item() for key in data}
        return feature_database["arr_0"]
    
    def get_features(self, filepath):
        pose_output = self.pose_img(filepath)
        dino_embeddings = self.dino(pose_output, filepath, 0, save_patches=0)
        return dino_embeddings

    def knn_search(self, database, query_vector):
        db_values = database.values()
        db_keys = database.keys()
        distances = []
        for idx, db_vector in enumerate(db_values):
            sim = self.cosine_similarity(db_vector, query_vector)
            distances.append((sim, list(db_keys)[idx]))
        
        # Sort by distance and get the k smallest
        distances.sort(reverse=True)
        nearest_distances, indices = zip(*distances[:self.k])
        return np.array(nearest_distances), np.array(indices)

    def cosine_similarity(self, emb_1, emb_2):
        # Calculate cosine similarity between two embeddings
        sim = cosine_similarity(np.squeeze(emb_1).reshape(1, -1), np.squeeze(emb_2).reshape(1, -1))[0][0]
        return sim

    def predict(self, image_dir, db_files_path = None, gen_db=False):
        print("Predicting...")
        if gen_db:
            self.gen_database(db_files_path, self.db_path)
        # Load the database
        print("Loading database...")
        db = self.load_db(self.db_path)
        print("Database loaded.")
        # Extract features from the query image
        print("Extracting features from the query image...")
        query = self.get_features(image_dir)
        print("Features extracted.")
        # Perform k-nearest neighbors search
        print("Performing k-nearest neighbors search...")
        distances, annotations = self.knn_search(db, query)
        print("k-nearest neighbors search done.")
        
        # Decide on the most common result
        numbers = [int(re.search(r'numeral(\d+)\.jpg', filename).group(1))
                for filename in annotations if re.search(r'numeral(\d+)\.jpg', filename)]
        numbers.sort()
        total_count = len(numbers)
        number_percentage = {
            num: (len(list(group)) / total_count) * 100
            for num, group in groupby(numbers)
        }
        number_percentage = {key: f"{value:.2f}%" for key, value in number_percentage.items()}
        
        # Write the output to a text file
        with open("output.txt", "a") as file:
            file.write(f"Image: {image_dir}\n")
            file.write(str(number_percentage))
            file.write("\n")
        
        return number_percentage

if __name__ == "__main__":
    checkpoints_pose = "checkpoints/pose"
    checkpoint_dino = "checkpoints/dino/hand/teacher_checkpoint.pth"
    image_dir = "Numerals/Numerals_SaudiSL/numeral2.jpg"
    db_path = "patches/sign_db.npz"
    db_files_path = "Numerals/Numerals_new"
    gen_db = 0
    
    directory_path = "Numerals/Numerals_SaudiSL"
    handler = DatabaseHandler(checkpoints_pose, checkpoint_dino, db_path)
    handler.load_models()
    
    for k in range(1, 12):
        with open("output.txt", "a") as file:
            file.write(f"------K: {k}\n")
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith('.jpg'):
                number = handler.predict(db_files_path, db_path, file_path, gen_db, k)