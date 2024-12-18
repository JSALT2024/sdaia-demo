from DatabaseHandler import DatabaseHandler

checkpoints_pose = "data/pose"
checkpoint_dino = "data/dino/hand/teacher_checkpoint.pth"
image_dir = "img/numeral1.jpg"
data_db_path = "data/Numerals/Numerals_database_saudi"
db_path = "sign_db.npz"
k=9

#Create database from mirrored images to DINO in RGB
creator = DatabaseHandler(checkpoints_pose, checkpoint_dino, db_path, data_db_path, k)
creator.load_models()
creator.gen_database(data_db_path, db_path)