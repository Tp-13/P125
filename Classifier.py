import numpy as np
import pandas as pd
import PIL.ImageOps
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


x = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9, train_size=3500, test_size=500)
x_train_scaled = x_train/255
x_test_scaled = x_test/255
clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(x_train_scaled, y_train)

def get_pred(img):
    img_pil = Image.open(img)
    img_bw = img_pil.convert("L")
    img_bw_resized = img_bw.resize((28, 28))
    pixel_factor = 20
    min_pxl = np.percentile(img_bw_resized, pixel_factor)
    img_bw_resized_inverted_scaled = np.clip(img_bw_resized - min_pxl, 0, 255)
    max_pxl = np.max(img_bw_resized)
    img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pxl
    test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1, 660)
    test_pred = clf.predict(test_sample)
    return test_pred[0]