import os
import yaml
import mlflow
import mlflow.tensorflow
from datasets import load_dataset
import tensorflow as tf

from src.model import build_novel_unet
from src.trainer import RadioTrainer
from src.dataset import UAVChannelDataset


def main():
    # -------------------------
    # Load config
    # -------------------------
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # MLflow setup
    mlflow.set_experiment(config["experiment_name"])
    run_name = config.get("run_name", "default-run")

    # Dataset config
    hf_repo = config["hf_repo"]
    cache_dir = config["local_cache_dir"]
    train_split = config["train_split"]
    test_split = config["test_split"]

    # Globals
    GLOBAL_MINS = config["mins"]
    GLOBAL_MAXS = config["maxs"]

    # Training config
    epochs = config["epochs"]
    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    save_path = config["save_weights_path"]

    # Model config
    input_shape = tuple(config["input_shape"])

    # -------------------------
    # Prepare directories
    # -------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # -------------------------
    # Load dataset
    # -------------------------
        
    
    train_dataset = UAVChannelDataset(global_mins = GLOBAL_MINS, global_maxs = GLOBAL_MAXS, training=True).get_dataset()
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset)).prefetch(tf.data.AUTOTUNE)
    print("Training dataset size: ", len(train_dataset))

    val_dataset = UAVChannelDataset(global_mins = GLOBAL_MINS, global_maxs = GLOBAL_MAXS, training=False).get_dataset()
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.shuffle(buffer_size=len(val_dataset)).prefetch(tf.data.AUTOTUNE)
    print("Test dataset size: ", len(val_dataset))

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
    # Train under MLflow
    # -------------------------
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "input_shape": input_shape,
        })

        print("Starting training...")
        trainer.train(train_dataset, val_dataset, epochs=epochs, save_path=save_path)

        print(f"Training completed. Best model saved at: {save_path}")


if __name__ == "__main__":
    main()
