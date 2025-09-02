import os
import yaml
import mlflow
import mlflow.tensorflow
from datasets import load_dataset
import tensorflow as tf

from src.model import build_novel_unet
from src.trainer import RadioTrainer
from src.dataset import UAVChannelDataset
from src.evaluator import Evaluator
from src.constants import GLOBAL_MINS, GLOBAL_MAXS


def main():
    # -------------------------
    # Load config
    # -------------------------
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # MLflow setup
    mlflow.set_experiment(config["experiment_name"])
    run_name = config.get("run_name", "default-run")

    # Training config
    epochs = config["epochs"]
    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    save_path = config["save_weights_path"]                # default save path
    pretrained_path = config["pretrained_weights_path"]    # pre-trained model weights

    # Model config
    input_shape = tuple(config["input_shape"])

    # -------------------------
    # Prepare datasets
    # -------------------------       
    
    train_dataset = UAVChannelDataset(global_mins = GLOBAL_MINS, global_maxs = GLOBAL_MAXS, training=True).get_dataset()
    print("Training dataset size: ", len(train_dataset))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.shuffle(buffer_size=13500).prefetch(tf.data.AUTOTUNE)

    val_dataset = UAVChannelDataset(global_mins = GLOBAL_MINS, global_maxs = GLOBAL_MAXS, training=False).get_dataset()
    print("Test dataset size: ", len(val_dataset))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.shuffle(buffer_size=270).prefetch(tf.data.AUTOTUNE)

    # -------------------------
    # Build model
    # -------------------------
    print("Building UNet model...")
    model = build_novel_unet(input_shape=input_shape)

    # -------------------------
    # Trainer
    # -------------------------
    trainer = RadioTrainer(model, lr=lr)

    # -------------------------
    # Train and Evaluate under MLflow
    # -------------------------
    with mlflow.start_run(run_name=run_name):
        mlflow.tensorflow.autolog()

        print("Starting training...")
        trainer.train(train_dataset, val_dataset, epochs=epochs, save_path=save_path)

        print(f"Training completed. Best model saved at: {save_path}")
        
        model.load_weights(pretrained_path) # loading pre-trained weights. change to save_path to load custom trained weights
        evaluator = Evaluator(model=model, global_mins=GLOBAL_MINS, global_maxs=GLOBAL_MAXS)
        print("Starting Evaluating the trained model...")
        evaluator.evaluate()




if __name__ == "__main__":
    main()
