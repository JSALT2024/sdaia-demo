from BackendRunner import BackendRunner
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import torch
import json
import re
import os

class DatabaseHandler(BackendRunner):
    def __init__(self, checkpoint_pose, checkpoint_dino, db_path, db_data_path, k=11):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_pose = checkpoint_pose
        self.checkpoint_dino = checkpoint_dino
        self.models_pose = None
        self.model_dino_hand = None
        self.k = k
        self.db_path = db_path
        self.db_data_path = db_data_path
    
    def gen_database(self, folderpath, db_path):
        feature_database = {}
        for filename in os.listdir(folderpath):
            file_path = os.path.join(folderpath, filename)
            if os.path.isfile(file_path) and filename.lower().endswith('.jpg'):
                file_features, _ = self.get_features(file_path, source = None)
                feature_database[filename] = file_features
        #with open('output.txt', 'a') as file:
        #    file.write(str(feature_database))
                
        np.savez(db_path, feature_database)
    
    def load_db(self, db_path):
        data = np.load(db_path, allow_pickle=True)
        feature_database = {key: data[key].item() for key in data}
        return feature_database["arr_0"]
    
    def get_features(self, filepath, source):
        pose_output = self.pose_img(filepath, source)
        dino_embeddings = self.dino(pose_output, filepath, 0, save_patches=1)
        #print(dino_embeddings)
        predicted_hand = pose_output["cropped_right_hand"][0]
        return dino_embeddings, predicted_hand

    def knn_search(self, database, query_vector):
        db_values = database.values()
        db_keys = database.keys()
        similarities = []
        for idx, db_vector in enumerate(db_values):
            sim = self.cosine_similarity(db_vector, query_vector)
            similarities.append((sim, list(db_keys)[idx]))
        
        # Sort by similarity and get the k largest
        similarities.sort(reverse=True, key=lambda x: x[0])
        best_similarities, indices = zip(*similarities[:self.k])
        return dict(zip(indices, best_similarities))

    def cosine_similarity(self, emb_1, emb_2):
        # Calculate cosine similarity between two embeddings
        sim = cosine_similarity(np.squeeze(emb_1).reshape(1, -1), np.squeeze(emb_2).reshape(1, -1))[0][0]
        return sim

    def compute_results(self, image_dir, similarities_dict):
        # create a dict with keys == annotations, values == percentage
        total_keys = len(similarities_dict)
        numbers = re.findall(r'(?<=numeral)\d+', ' '.join(similarities_dict.keys()))
        numbers = {num: (count / len(numbers)) * 100 for num, count in Counter(numbers).items()}
        
        #If there are multiple best results with the same percentage, decide according to the cosine similarity value and reorder the dict
        max_percentage = max(numbers.values())
        identical_percentages = [k for k, v in numbers.items() if v == max_percentage]
        best_key = max((num for num in identical_percentages if any(num in k for k in similarities_dict)), key=lambda n: max((v for k, v in similarities_dict.items() if n in k), default=0), default=None)
        numbers = {best_key: numbers.pop(best_key), **numbers} if best_key else numbers
        best_key_name = max((k for k in similarities_dict if f"numeral{best_key}.jpg" in k), key=similarities_dict.get, default=None)
        
        return numbers, best_key_name

    def retrieve_best_match_patch(self, filename, source):
        folderpath = self.db_data_path
        filepath = os.path.join(folderpath, filename)
        pose_output = self.pose_img(filepath, source, retrieve=True)
        predicted_hand = pose_output["cropped_right_hand"][0]
        return predicted_hand

    def predict(self, image_dir, source, db_files_path = None):
        # Load the database
        db = self.load_db(self.db_path)
        # Extract features from the query image
        query, predicted_hand = self.get_features(image_dir, source)
        # Perform k-nearest neighbors search
        similarities_dict = self.knn_search(db, query)
        if any(value > 0.9 for value in similarities_dict.values()):
            best_key_name = max(similarities_dict, key=similarities_dict.get)
            best_key = int(re.findall(r'(?<=numeral)\d+', best_key_name)[0])
            best_match = self.retrieve_best_match_patch(best_key_name, source)
        else:
            _, best_key_name = self.compute_results(image_dir, similarities_dict)
            if best_key_name == None:
                print("Best key name is None.")
            best_key = int(re.findall(r'(?<=numeral)\d+', best_key_name)[0])
            best_match = self.retrieve_best_match_patch(best_key_name, source)
        
        print(best_key_name)
        return best_key, predicted_hand, best_match

if __name__ == "__main__":
    checkpoints_pose = "data/pose"
    checkpoint_dino = "data/dino/hand/teacher_checkpoint.pth"
    image_dir = "img/numeral1.jpg"
    data_db_path = "SignLLaVA-app/Numerals/Numerals_new"
    db_path = "sign_db.npz"
    k=15
    
    handler = DatabaseHandler(checkpoints_pose, checkpoint_dino, db_path, data_db_path, k)
    handler.load_models()
    prediction = handler.predict(image_dir)
    print(prediction)