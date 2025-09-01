import mlflow
import mlflow.tensorflow
import time
import tensorflow as tf

class RadioTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.train_rmse = tf.keras.metrics.Mean(name='train_rmse')
        self.val_rmse = tf.keras.metrics.Mean(name='val_rmse')
        self.train_nmse = tf.keras.metrics.Mean(name='train_nmse')
        self.val_nmse = tf.keras.metrics.Mean(name='val_nmse')
        self.train_mae = tf.keras.metrics.Mean(name='train_mae')
        self.val_mae = tf.keras.metrics.Mean(name='val_mae')

    def rmse_loss(self, y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    def rmse_metric(self, y_true, y_pred):
        y_true = y_true * (GLOBAL_MAX_PATH_LOSS - GLOBAL_MIN_PATH_LOSS) + GLOBAL_MIN_PATH_LOSS
        y_pred = y_pred * (GLOBAL_MAX_PATH_LOSS - GLOBAL_MIN_PATH_LOSS) + GLOBAL_MIN_PATH_LOSS
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    def nmse_metric(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        power = tf.reduce_mean(tf.square(y_true))
        return mse / power

    def mae_metric(self, y_true, y_pred):
        y_true = y_true * (GLOBAL_MAX_PATH_LOSS - GLOBAL_MIN_PATH_LOSS) + GLOBAL_MIN_PATH_LOSS
        y_pred = y_pred * (GLOBAL_MAX_PATH_LOSS - GLOBAL_MIN_PATH_LOSS) + GLOBAL_MIN_PATH_LOSS
        return tf.reduce_mean(tf.abs(y_pred - y_true))

    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            output = self.model(inputs, training=True)

            loss = self.rmse_loss(targets, output)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Calculate metrics
        rmse = self.rmse_metric(targets, output)
        nmse = self.nmse_metric(targets, output)
        mae = self.mae_metric(targets, output)

        # Update metrics
        self.train_loss.update_state(loss)
        self.train_rmse.update_state(rmse)
        self.train_nmse.update_state(nmse)
        self.train_mae.update_state(mae)

        return loss

    @tf.function
    def val_step(self, inputs, targets):
        output = self.model(inputs, training=False)

        # Select appropriate output based on training phase
        loss = self.rmse_loss(targets, output)
        rmse = self.rmse_metric(targets, output)
        nmse = self.nmse_metric(targets, output)
        mae = self.mae_metric(targets, output)

        # Update metrics
        self.val_loss.update_state(loss)
        self.val_rmse.update_state(rmse)
        self.val_nmse.update_state(nmse)
        self.val_mae.update_state(mae)

        return loss

    def reset_metrics(self):
        self.train_loss.reset_state()
        self.val_loss.reset_state()
        self.train_rmse.reset_state()
        self.val_rmse.reset_state()
        self.train_nmse.reset_state()
        self.val_nmse.reset_state()
        self.train_mae.reset_state()
        self.val_mae.reset_state()

    def train(self, train_dataset, val_dataset, epochs, save_path):
            best_val_loss = float('inf')
            with mlflow.start_run():
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", self.optimizer.lr.numpy())

                for epoch in range(epochs):
                    start_time = time.time()
                    self.reset_metrics()
    
                    print(f"Starting Epoch {epoch+1}/{epochs}")
                    # Training loop - Iterate directly over the dataset
                    batch_count = 0
                    for inputs, targets in train_dataset:
                        self.train_step(inputs, targets)
                        batch_count += 1    
    
                    # Validation loop - Iterate directly over the dataset
                    val_batch_count = 0
                    for inputs, targets in val_dataset:
                        self.val_step(inputs, targets)
                        val_batch_count += 1

                    # log metrics
                    mlflow.log_metric("train_loss", float(self.train_loss.result()), step=epoch)
                    mlflow.log_metric("val_loss", float(self.val_loss.result()), step=epoch)
                    mlflow.log_metric("val_rmse", float(self.val_rmse.result()), step=epoch)
    
    
                    # Print epoch summary
                    epoch_time = time.time() - start_time
                    print(f"\nEpoch {epoch+1}/{epochs} - {epoch_time:.1f}s")
                    print(f"Train: Loss: {self.train_loss.result():.6f} | "
                          f"RMSE: {self.train_rmse.result():.6f} | "
                          f"NMSE: {self.train_nmse.result():.6f} | "
                          f"MAE: {self.train_mae.result():.6f}")
                    print(f"Val:   Loss: {self.val_loss.result():.6f} | "
                          f"RMSE: {self.val_rmse.result():.6f} | "
                          f"NMSE: {self.val_nmse.result():.6f} | "
                          f"MAE: {self.val_mae.result():.6f}")
    
                    # Save best model
                    if self.val_loss.result() < best_val_loss:
                        best_val_loss = self.val_loss.result()
                        # Assuming your model is a Keras Model instance
                        self.model.save_weights(save_path)
                        print(f"Saved best model with val loss: {best_val_loss:.6f}")
                # Reload best weights before logging final model
                self.model.load_weights(save_path)
                mlflow.tensorflow.log_model(self.model, artifact_path="model")
