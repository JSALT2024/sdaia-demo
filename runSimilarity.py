import os
import numpy as np
import matplotlib.pyplot as plt
from BackendRunner import BackendRunner

def eval(runner, files_custom, files_saudi, dino = 0, mae = 0):
    
    if dino:
        similarity_matrix_dino = np.zeros((len(files_custom), len(files_saudi)))
        for i, file_custom in enumerate(files_custom):
            # Process image from Numerals_custom
            pose_output_custom = runner.pose_img(file_custom)
            dino_embeddings_custom = runner.dino(pose_output_custom, file_custom, 0, save_patches=0)
    
            for j, file_saudi in enumerate(files_saudi):
                # Process image from Numerals_SaudiSL
                pose_output_saudi = runner.pose_img(file_saudi)
                dino_embeddings_saudi = runner.dino(pose_output_saudi, file_saudi, 0, save_patches=0)
                
                similarity_matrix_dino[i, j] = runner.similarity(dino_embeddings_custom, dino_embeddings_saudi)
    
    if mae:
        similarity_matrix_mae = np.zeros((len(files_custom), len(files_saudi)))
        for i, file_custom in enumerate(files_custom):
            # Process image from Numerals_custom
            pose_output_custom = runner.pose_img(file_custom)
            mae_embeddings_custom = runner.mae(pose_output_custom)
            
            for j, file_saudi in enumerate(files_saudi):
                # Process image from Numerals_SaudiSL
                pose_output_saudi = runner.pose_img(file_saudi)
                mae_embeddings_saudi = runner.mae(pose_output_saudi)

                similarity_matrix_mae[i, j] = runner.similarity(mae_embeddings_custom, mae_embeddings_saudi)
                
    if dino and mae:
        return [similarity_matrix_dino, similarity_matrix_mae]
    elif dino:
        return [similarity_matrix_dino]
    elif mae:
        return [similarity_matrix_mae]
    else:
        print("You should choose at leats one model features for similarity evaluation.")

def execute_visualization(similarity_matrices, files_custom, files_saudi, dino, mae):
    
    if len(similarity_matrices) == 2:
        arch = "dino"
        visualize(similarity_matrices[0], files_custom, files_saudi, arch)
        arch = "mae"
        visualize(similarity_matrices[1], files_custom, files_saudi, arch)
    else:
        arch = "dino" if dino else "mae" if mae else "unknown"
        visualize(similarity_matrices[0], files_custom, files_saudi, arch)

def visualize(similarity_matrix, files_custom, files_saudi, arch):
    # Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Cosine Similarity")
    plt.xlabel("Numerals_SaudiSL")
    plt.ylabel("Numerals_custom")
    plt.title("Similarity Matrix")
    plt.xticks(ticks=np.arange(len(files_saudi)), labels=[os.path.basename(f) for f in files_saudi], rotation=90)
    plt.yticks(ticks=np.arange(len(files_custom)), labels=[os.path.basename(f) for f in files_custom])
    plt.tight_layout()
    plt.savefig("patches/SimilarityMatrixFiltered-{}.png".format(arch), format='png', dpi=600)

if __name__ == "__main__":
    # Checkpoints
    checkpoints_pose = "checkpoints/pose"
    checkpoint_mae = "checkpoints/mae/16-07_21-52-12/checkpoint-440.pth"
    checkpoint_dino = "checkpoints/dino/hand/teacher_checkpoint.pth"
    runner = BackendRunner(checkpoints_pose, checkpoint_mae, checkpoint_dino)
    runner.load_models()
    # Directories
    dir_custom = "Numerals/Numerals_custom"
    dir_saudi = "Numerals/Numerals_SaudiSL"
    files_custom = [os.path.join(dir_custom, f) for f in os.listdir(dir_custom) if f.endswith(".jpg")]
    files_saudi = [os.path.join(dir_saudi, f) for f in os.listdir(dir_saudi) if f.endswith(".jpg")]
    # Dino x MAE features for evaluation
    run_dino = 1
    run_mae = 0
    similarity_matrices = eval(runner, files_custom, files_saudi, run_dino, run_mae)
    execute_visualization(similarity_matrices, files_custom, files_saudi, run_dino, run_mae)