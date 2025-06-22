import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Paths to the alphabet and word gesture folders
alphabet_dir = 'data/alphabet/'
word_dir = 'data/filtered_data/'

# Dictionary to hold the text label and corresponding GIF paths
data = []

# Process alphabet gestures
for gif_file in os.listdir(alphabet_dir):
    if gif_file.endswith('.gif'):
        label = os.path.splitext(gif_file)[0]
        gif_path = os.path.join(alphabet_dir, gif_file)
        data.append((label, gif_path))

# Process word gestures
for file in os.listdir(word_dir):
    if file.endswith('.webp'):
        label = os.path.splitext(file)[0]
        file_path = os.path.join(word_dir, file)
        data.append((label, file_path))

# Separate text labels and GIF paths
texts, gif_paths = zip(*data)

# Tokenize text input
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=5)  # Adjust maxlen based on your text length

def load_image(path):
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')  # Ensure image is in RGB mode
            img = img.resize((64, 64))  # Resize to (64, 64) to ensure consistent size
            return np.array(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return np.zeros((64, 64, 3))  # Return a blank image if an error occurs

# Preprocess the GIF/WEBP data
y = []
labels = []
for gif_path in gif_paths:
    try:
        gif_data = load_image(gif_path)
        if gif_data.shape == (64, 64, 3):
            y.append(gif_data)
            labels.append(os.path.splitext(os.path.basename(gif_path))[0])
        else:
            print(f"Unexpected shape: {gif_data.shape} for {gif_path}")
    except Exception as e:
        print(f"Error processing image {gif_path}: {e}")

# Encode labels
label_encoder = LabelEncoder()
y_labels = label_encoder.fit_transform(labels)
y_labels = np.array(y_labels)

# Ensure y is a NumPy array with consistent shapes
y = np.array(y)
print(f"Shape of y: {y.shape}")  # Verify the shape of the NumPy array

# Split dataset into training and testing
X_train, X_test, y_train, y_test, y_labels_train, y_labels_test = train_test_split(X, y, y_labels, test_size=0.2)

# Define and compile the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=X.shape[1]))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(1024, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer for classification

model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

model.summary()

# Train the model
model.fit(X_train, y_labels_train, epochs=10, batch_size=32, validation_data=(X_test, y_labels_test))


# Save the model
model.save('audio_to_gesture_model.h5')
