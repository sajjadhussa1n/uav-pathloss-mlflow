import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
random.seed(42)
np.random.seed(42)

class UAVChannelDataset:
    def __init__(self, global_mins, global_maxs, directory = None, feature_cols=['Distance_3d', 'LOS_mask', 'Is_building'], target_col='Path_loss', training=True):
        self.training = training
        # Default directory is ./dataset in project root
        if directory is None:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            directory = os.path.join(root_dir, "dataset")
        self.directory = directory
        self.train_directory = os.path.join(directory, 'train')
        self.train_files = [file for file in os.listdir(self.train_directory)]
        self.test_directory = os.path.join(directory, 'test')
        self.test_files = [file for file in os.listdir(self.test_directory)]
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.global_mins = global_mins
        self.global_maxs = global_maxs

    def _create_3_channel_input(self):
        files = self.train_files if self.training else self.test_files
        the_directory = self.train_directory if self.training else self.test_directory
        input_channels_data = []
        output_channel_data = []

        for file in files:
            width, height = 384, 256

            total_images = 100
            operations = [np.flipud, np.fliplr]
            file_path = os.path.join(the_directory, file)
            data = pd.read_csv(file_path)

            # Create 3-channel input and target array
            array = np.stack([
                data[self.feature_cols[0]].values.reshape(width, height).astype(np.float32),
                data[self.feature_cols[1]].values.reshape(width, height).astype(np.float32),
                data[self.feature_cols[2]].values.reshape(width, height).astype(np.float32)
            ], axis=-1)

            target = data[self.target_col].values.reshape(width, height).astype(np.float32)
            target_norm = (target - self.global_mins[-1]) / (self.global_maxs[-1] - self.global_mins[-1])
            
            # Normalize each channel independently
            for i in range(3):
                channel = array[:, :, i]
                if i == 0:
                    channel = 20.0*np.log10(channel)    # Calculate 20*log10(Distance_3d)
                array[:, :, i] = (channel - self.global_mins[i]) / (self.global_maxs[i] - self.global_mins[i])

            used_combinations = set()

            systematic_patterns = [
                # Pattern 1: 3x2 grid (6 chunks)
                {'i_strides': (128, 1), 'j_strides': (128, 1)},
                # Pattern 2: 3x1 grid (3 chunks)
                {'i_strides': (128, 1), 'j_strides': (256, 2)},
                # Pattern 3: 2x2 grid (4 chunks)
                {'i_strides': (128, 2), 'j_strides': (128, 1)},
                # Pattern 4: 2x1 grid (2 chunks)
                {'i_strides': (128, 2), 'j_strides': (256, 2)},
                # Pattern 5: 1x2 grid (2 chunks)
                {'i_strides': (384, 3), 'j_strides': (128, 1)},
                # Pattern 6: 1x1 grid (1 chunk)
                {'i_strides': (384, 3), 'j_strides': (256, 2)}
            ]

            # Generate systematic chunks
            for pattern in systematic_patterns:
                i_step, i_stride = pattern['i_strides']
                j_step, j_stride = pattern['j_strides']

                # Calculate valid starting positions
                max_i = 384 - (128 * i_stride)
                max_j = 256 - (128 * j_stride)

                for i in range(0, max_i + 1, i_step):
                    for j in range(0, max_j + 1, j_step):
                        # Calculate window bounds
                        i_end = i + (128 * i_stride)
                        j_end = j + (128 * j_stride)

                        input_slice = array[i:i_end:i_stride, j:j_end:j_stride, :]
                        output_slice = target_norm[i:i_end:i_stride, j:j_end:j_stride]

                        input_channels_data.append(input_slice)
                        output_channel_data.append(output_slice)

            counter = 18

            if self.training:
                while counter < total_images:
                    v_stride = random.choice([1, 2, 3])  # Vertical
                    h_stride = random.choice([1, 2])      # Horizontal

                    # Calculate valid starting positions
                    max_i = 384 - 128*v_stride
                    max_j = 256 - 128*h_stride

                    if max_i < 0 or max_j < 0:
                        continue

                    i_start = random.randint(0, max_i)
                    j_start = random.randint(0, max_j)

                    combination = (i_start, v_stride, j_start, h_stride)
                    if combination in used_combinations:
                        continue
                    used_combinations.add(combination)

                    # Create slices with strides
                    input_slice = array[i_start:i_start+128*v_stride:v_stride,
                                        j_start:j_start+128*h_stride:h_stride, :]

                    output_slice = target_norm[i_start:i_start+128*v_stride:v_stride,
                                        j_start:j_start+128*h_stride:h_stride]

                    if input_slice.shape[:2] == (128, 128):
                        input_channels_data.append(input_slice)
                        output_channel_data.append(output_slice)
                        for i in range(2):
                          operation = operations[i]
                          input_channels_data.append(operation(input_slice)[:128, :128, :])
                          output_channel_data.append(operation(output_slice)[:128, :128])
                        counter = counter + 1

        input_channels_data = np.array(input_channels_data, dtype=np.float32)
        output_channel_data = np.expand_dims(np.array(output_channel_data, dtype=np.float32), axis=-1)
        print("Total number of 128x128 samples in dataset: ", input_channels_data.shape[0])
        return input_channels_data, output_channel_data

    def get_dataset(self):
        input_data, mask = self._create_3_channel_input()
        dataset = tf.data.Dataset.from_tensor_slices((input_data, mask))
        return dataset
