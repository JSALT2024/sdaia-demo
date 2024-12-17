from BackendRunner import BackendRunner
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import os

class DatabaseHandler(BackendRunner):
    def __init__(self, checkpoint_pose, checkpoint_mae, checkpoint_dino, db_path, k=3):
        # Initialize the device to use GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Store checkpoint paths for different models
        self.checkpoint_pose = checkpoint_pose
        self.checkpoint_mae = checkpoint_mae
        self.checkpoint_dino = checkpoint_dino
        # Initialize model variables
        self.models_pose = None
        self.model_mae = None
        self.model_dino_hand = None
        # Set the number of nearest neighbors to find
        self.k = k
        # Path to the database file
        self.db_path = db_path
    
    def gen_database(self, folderpath, db_path):
        # Generate a feature database from images in the specified folder
        feature_database = {}
        for filename in os.listdir(folderpath):
            file_path = os.path.join(folderpath, filename)
            # Check if the file is a JPEG image
            if os.path.isfile(file_path) and filename.lower().endswith('.jpg'):
                # Extract features from the image
                file_features = self.get_features(file_path)
                # Store the features in the database
                feature_database[filename] = file_features
        
        # Save the feature database to a file
        np.savez(db_path, feature_database)
    
    def load_db(self, db_path):
        # Load the feature database from a file
        data = np.load(db_path, allow_pickle=True)
        # Convert the loaded data to a dictionary
        feature_database = {key: data[key].item() for key in data}
        return feature_database["arr_0"]
    
    def get_features(self, filepath):
        # Extract features from an image file
        pose_output = self.pose_img(filepath)
        # Get DINO embeddings for the image
        dino_embeddings = self.dino(pose_output, filepath, 0, save_patches=0)
        return dino_embeddings

    def knn_search(self, database, query_vector, k=3):
        # Perform k-nearest neighbors search in the database
        db_values = database.values()
        db_keys = database.keys()
        distances = []
        for idx, db_vector in enumerate(db_values):
            # Calculate cosine similarity between the query and database vectors
            sim = self.cosine_similarity(db_vector, query_vector)
            distances.append((sim, list(db_keys)[idx]))
        
        # Sort by similarity and get the top k results
        distances.sort(reverse=True)
        nearest_distances, indices = zip(*distances[:k])
        return np.array(nearest_distances), np.array(indices)

    def cosine_similarity(self, emb_1, emb_2):
        # Calculate cosine similarity between two embeddings
        sim = cosine_similarity(np.squeeze(emb_1).reshape(1, -1), np.squeeze(emb_2).reshape(1, -1))[0][0]
        return sim

    def predict(self, db_files_path, db_path, image_dir, gen_db, k):
        self.load_models()
        if gen_db:
            self.gen_database(db_files_path, db_path)
        # Load the database
        db = self.load_db(db_path)
        # Extract features from the query image
        query = self.get_features(image_dir)
        # Perform k-nearest neighbors search
        distances, annotations = self.knn_search(db, query, k)
        
        # Decide on the most common result
        numbers = [int(re.search(r'numeral(\d+)\.jpg', filename).group(1)) 
                for filename in annotations if re.search(r'numeral(\d+)\.jpg', filename)]
        numbers.sort()
        most_common_number = max((len(list(group)), num) for num, group in groupby(numbers))[1]
        
        print(annotations)
        print("Closest annotation: {}".format(most_common_number))
        return most_common_number

if __name__ == "__main__":
    # Define paths to model checkpoints and image directories
    checkpoints_pose = "checkpoints/pose"
    checkpoint_mae = "checkpoints/mae/16-07_21-52-12/checkpoint-440.pth"
    checkpoint_dino = "checkpoints/dino/hand/teacher_checkpoint.pth"
    image_dir = "Numerals/Numerals_SaudiSL/numeral4.jpg"
    db_path = "patches/sign_db.npz"
    db_files_path = "Numerals/Numerals_SaudiSL"
    k = 9
    gen_db = 1

    # Execute
    handler = DatabaseHandler(checkpoints_pose, checkpoint_mae, checkpoint_dino, db_path)
    annotations = handler.predict(db_files_path, db_path, image_dir, gen_db)