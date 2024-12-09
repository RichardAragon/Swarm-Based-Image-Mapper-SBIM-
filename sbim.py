import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from skimage.transform import resize
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from tqdm import tqdm

# Disable GPU by default (optional)
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class SwarmAgent:
    """Agent representing a point in the Gaussian Probability Space."""
    def __init__(self, position, velocity, image_shape):
        self.position = position
        self.velocity = velocity
        self.m = np.zeros_like(position)
        self.v = np.zeros_like(position)
        self.image_shape = image_shape

    def add_noise(self, noise_level):
        """Adds Gaussian noise to the agent's position."""
        self.position += np.random.randn(*self.image_shape) * np.sqrt(noise_level)


class SBIM:
    """Swarm-Based Image Mapper for denoising and mapping an image space."""
    def __init__(self, num_agents, image_shape, target_image_path):
        self.num_agents = num_agents
        self.image_shape = image_shape
        self.resized_shape = (128, 128, 3)  # Reduced for MobileNetV2
        self.agents = [SwarmAgent(self.random_position(), self.random_velocity(), image_shape)
                       for _ in range(num_agents)]
        self.target_image = self.load_image(target_image_path)
        self.generated_image = np.random.randn(*image_shape)
        self.mobilenet = self.load_mobilenet_model()
        self.noise_schedule = np.linspace(0.1, 0.002, 1000)  # Linear noise reduction

    @staticmethod
    def random_position():
        return np.random.randn(128, 128, 3)  # Gaussian noise initialization

    @staticmethod
    def random_velocity():
        return np.random.randn(128, 128, 3) * 0.01

    @staticmethod
    def load_image(img_path):
        img = Image.open(img_path).resize((128, 128))
        img_array = np.array(img) / 127.5 - 1  # Normalize to [-1, 1]
        return img_array

    def load_mobilenet_model(self):
        mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.resized_shape)
        return Model(inputs=mobilenet.input, outputs=mobilenet.get_layer('block_13_expand_relu').output)

    def resize_image(self, image):
        return resize(image, self.resized_shape, anti_aliasing=True)

    def perceptual_loss(self, agent_positions):
        target_features = self.extract_features(self.target_image)
        losses = []
        for agent_pos in agent_positions:
            agent_features = self.extract_features(agent_pos)
            loss = np.mean((target_features - agent_features) ** 2)
            losses.append(loss)
        return np.array(losses)

    def extract_features(self, image):
        resized_image = self.resize_image((image + 1) / 2)  # Normalize to [0, 1]
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized_image[np.newaxis, ...] * 255)
        return self.mobilenet.predict(preprocessed)

    def update_agents(self, timestep):
        noise_level = self.noise_schedule[min(timestep, len(self.noise_schedule) - 1)]
        for agent in self.agents:
            predicted_noise = agent.position - self.target_image
            denoised = (agent.position - noise_level * predicted_noise) / (1 - noise_level)
            agent.position = np.clip(denoised + np.random.randn(*self.image_shape) * np.sqrt(noise_level), -1, 1)

    def generate_image(self):
        self.generated_image = np.mean([agent.position for agent in self.agents], axis=0)
        image_pil = Image.fromarray(((self.generated_image + 1) / 2 * 255).astype(np.uint8))
        image_pil = image_pil.filter(ImageFilter.SHARPEN)
        return np.array(image_pil) / 255.0

    def train(self, epochs, log_interval=10):
        logging.basicConfig(filename='training.log', level=logging.INFO)
        for epoch in tqdm(range(epochs), desc="Training"):
            self.update_agents(epoch)
            self.generated_image = self.generate_image()

            # Compute loss
            mse = np.mean(((self.generated_image * 2 - 1) - self.target_image) ** 2)
            logging.info(f"Epoch {epoch}, MSE: {mse}")

            if epoch % log_interval == 0:
                print(f"Epoch {epoch}, MSE: {mse}")
                self.display_image(self.generated_image, title=f"Epoch {epoch}")

    @staticmethod
    def display_image(image, title=""):
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def save_model(self, filename):
        np.save(filename, {'agents': [agent.position for agent in self.agents]})

    def load_model(self, filename):
        saved_data = np.load(filename, allow_pickle=True).item()
        for agent, position in zip(self.agents, saved_data['agents']):
            agent.position = position


# Run SBIM
if __name__ == "__main__":
    # Hyperparameters
    IMAGE_PATH = "target_image.jpg"  # Path to target image
    NUM_AGENTS = 500
    IMAGE_SHAPE = (128, 128, 3)
    EPOCHS = 20

    # Initialize and Train
    sbim = SBIM(num_agents=NUM_AGENTS, image_shape=IMAGE_SHAPE, target_image_path=IMAGE_PATH)
    sbim.train(epochs=EPOCHS)
    sbim.display_image(sbim.generated_image, title="Final Generated Image")
    sbim.save_model("sbim_model.npy")
