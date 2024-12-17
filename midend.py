def get_prediction(input_image):
    
    checkpoints_pose = "data/pose"
    checkpoint_mae = "data/mae/16-07_21-52-12/checkpoint-440.pth"
    checkpoint_dino = "data/dino/hand/teacher_checkpoint.pth"
    image_dir = input_image
    db_path = "data/sign_db.npz"
    
    from DatabaseHandler import DatabaseHandler
    handler = DatabaseHandler(checkpoints_pose, checkpoint_mae, checkpoint_dino, db_path, k)
    
    annotation = handler.predict(image_dir, db_path)
    
    return annotation