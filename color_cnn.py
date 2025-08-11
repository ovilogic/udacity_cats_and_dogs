import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
from tensorflow.keras import layers, models

all_cats = os.listdir('./data/cats')
all_dogs = os.listdir('./data/dogs')

class ColorCNN:
    def __init__(self, files_list):
        self.files_list = files_list
         
    def prepare_data(self, data_path, folder="train", subfolder="train1", num_samples=1000):
        if not os.path.isdir(f'./{folder}/{subfolder}'):
            os.makedirs(f'./{folder}/{subfolder}')
            samples = random.sample(range(len(self.files_list)), num_samples)
            for one_sample in samples:
                sample = self.files_list[one_sample]
                if not sample.endswith('.jpg'):
                    print(f"Skipping {sample} as it is not a .jpg file.")
                    continue
                file = os.path.join(data_path, sample)
                if not os.path.isfile(file):
                    print(f"File {file} does not exist.")
                    continue
                os.system(f'cp "{file}" "./{folder}/{subfolder}/{sample}"')

    def preprocess_data(self, images_path):
        # Load the images as tensors — specifically tf.Tensor objects — not plain bytes anymore.
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            images_path, # Loads your .jpg files
            image_size=(150, 150), # Resizes them to (150, 150)
            batch_size=32, # not about how the images are preprocessed (resized, normalized, etc.),
                # it’s about how the dataset is delivered to the model during training.
                # During training, the model processes batch #1, updates weights, processes batch #2, updates weights, etc.
            shuffle=True,
            seed=42 # If you set seed=42 (or any fixed number), that random shuffling will always happen in the exact same way each time you run the script.
                # If you don’t set a seed, the shuffle order changes every time you run the code — which is fine for training, 
                # but it makes experiments harder to reproduce exactly. This is useful for debugging or comparing training results — you remove one source of randomness.
        )
        # Normalize the images to [0, 1] range
        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
        def augment(image, label):
            image = data_augmentation(image)
            return image, label
        # Apply the normalization function to the dataset
        train_dataset = train_dataset.map(normalize)
        # Apply the data augmentation function to the dataset
        train_dataset = train_dataset.map(augment)
        train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors()) # Ignore errors in the dataset, such as corrupted images
        return train_dataset
    
    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),  # Dropout layer to reduce overfitting
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(2, activation='softmax')  # For binary classification
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def train_model(self, train_dataset, validation_dataset, epochs=10):
        model = self.build_model()
        model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)
        return model
        
cats = ColorCNN(all_cats)
cats.prepare_data('./data/cats', folder="train", subfolder="cats", num_samples=1000)
cats.prepare_data('./data/cats', folder="validation", subfolder="cats", num_samples=500)

dogs = ColorCNN(all_dogs)
dogs.prepare_data('./data/dogs', folder="train", subfolder="dogs", num_samples=1000)
dogs.prepare_data('./data/dogs', folder="validation", subfolder="dogs", num_samples=500)


preprocessed_training_set = dogs.preprocess_data('./train')
preprocessed_validation_set = dogs.preprocess_data('./validation')
# for i in preprocessed.take(1):
#     images, labels = i
#       # This will print the first batch of images and labels
#     # print(tf.shape(images), tf.shape(labels), sep='\n')
#     print(images[0], labels[0], sep='\n')

# # Build the model
# model = cats.build_model()
# # Print the model summary
# model.summary()

trained_model = dogs.train_model(preprocessed_training_set, preprocessed_validation_set, epochs=30)
 