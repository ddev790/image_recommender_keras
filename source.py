import os
import glob
import numpy as np

from PIL import Image
from tqdm import tqdm

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

from flask import Flask, request, render_template

app = Flask(__name__)


# disable TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = 'static/uploaded/' + file.filename
        img.save(uploaded_img_path)

        query = fe.extract(img)
        dist = np.linalg.norm(features - query, axis=1)  # Do search
        ids = np.argsort(dist)[:5]                       # Get top 5 matches
        dist = [dist[id] for id in ids]
        retrieved_img_paths = [img_paths[id] for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=zip(dist, retrieved_img_paths))
    else:
        return render_template('index.html')


if __name__=="__main__":
    fe = FeatureExtractor()
    features, img_paths = ([] for i in range(2))
    for img_path in tqdm(sorted(glob.glob('static/images/*.jpg'))):            
        feature = fe.extract(Image.open(img_path))    # get features from PIL image
        features.append(feature)
        img_paths.append(img_path)
    os.makedirs('static/uploaded/')
    app.run()
