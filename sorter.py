from sklearn.manifold import TSNE
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
from PIL import Image, ImageFile
from datetime import datetime
import pickle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm
from matplotlib import cm
import sys
from pprint import pprint
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True

thumbnail_size = (256, 256)
rootdir = "/home/woj/Pictures/Google Fotos/"
allowed_extensions = [".jpg", ".JPG", ".jpeg", "JPEG", ".png", ".PNG"]


def get_image_files():
    print("Get Image Files")
    global image_files, allowed_extensions
    for extension in allowed_extensions:
        image_files.extend(glob_extension(extension))

    with open("image_files.dat", "wb") as f:
        pickle.dump(image_files, f)
    return image_files


def glob_extension(extension):
    files = []
    for filename in glob.glob(rootdir + f"**/*{extension}", recursive=True):
        files.append(filename)
    return files


def get_meta_data():
    global image_files, allowed_extensions, metadata
    print("Get Metadata")
    for file in image_files[:]:
        size = os.stat(file).st_size

        extension = Path(file).suffix
        extension_param = allowed_extensions.index(extension)

        image = Image.open(file)
        width, height = image.size
        exif = image.getexif()
        if exif and 36867 in exif:
            date = exif[36867]
            print("*" + date)
            dateobj = datetime.strptime(date, "%Y:%m:%d %H:%M:%S")
            timestamp = datetime.timestamp(dateobj)
        elif exif and 306 in exif:
            date = exif[306]

            try:
                dateobj = datetime.strptime(date, "%Y:%m:%d %H:%M:%S")
            except:
                timestamp = 0
                print(f"Cant read: <{date}> as %Y:%m:%d %H:%M:%S")
            else:
                timestamp = datetime.timestamp(dateobj)
        else:
            timestamp = 0

        metadata.append([extension_param, size, width, height, timestamp])

    with open("metadata.dat", "wb") as f:
        pickle.dump(metadata, f)


def create_thumbnails():
    global image_files, thumbnail_size, thumbnails
    print("Creating thumbnails")

    # create thumbnail dir
    path = f'thumbnails/{"x".join([str(s) for s in thumbnail_size])}'
    Path(path).mkdir(parents=True, exist_ok=True)

    # clean dir
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

    # make thumbnails
    for i, file in enumerate(image_files):
        if i % 500 == 0:
            print(f"{i} of {len(image_files)}")

        # if i < 17000: continue
        filename = f"thumb_{i}"
        full_name = f"{path}/{filename}.jpg"
        thumbnails.append(full_name)

        image = Image.open(file)
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        image.thumbnail(thumbnail_size)
        image.save(full_name)

    with open("thumbnails.dat", "wb") as f:
        pickle.dump(thumbnails, f)


def getImage(path, zoom=1):
    image = Image.open(path)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    image.putalpha(128)
    return OffsetImage(image, zoom=zoom)


# get_image_files()
# get_meta_data()
# create_thumbnails()
class Plotter:
    def __init__(self):
        self.plotted = []
        self.image_files = []
        self.metadata = []
        self.tensors = []
        self.thumbnails = []
        self.thumbnail_zoom = 0.005
        self.thumbnail_radius = 128
        self.load_data()
        data = [meta + tensor[:20] for meta, tensor in zip(self.metadata, self.tensors)]
        self.np_data = np.array(data)
        self.start_tsne()
        self.scatter_plot()

    def load_data(self):
        if Path("image_files.dat").exists():
            with open("image_files.dat", "rb") as f:
                self.image_files = pickle.load(f)

        if Path("metadata.dat").exists():
            with open("metadata.dat", "rb") as f:
                self.metadata = pickle.load(f)

        if Path("tensors.dat").exists():
            with open("tensors.dat", "rb") as f:
                self.tensors = pickle.load(f)

        if Path("thumbnails_128.dat").exists():
            with open("thumbnails_128.dat", "rb") as f:
                self.thumbnails = pickle.load(f)

        print(f"Image_Files {len(self.image_files)}")
        print(f"metadata {len(self.metadata)}")
        print(f"tensors {len(self.tensors)}")
        print(f"thumbnails {len(self.thumbnails)}")

    def start_tsne(self):
        tsne = TSNE(n_iter=10000, verbose=3, perplexity=100, random_state=123)
        X = self.np_data[:]
        z = tsne.fit_transform(X)
        self.x = z[:, 0]
        self.y = z[:, 1]

    def time_colormap(self):
        def cutoff(x):
            x = x - 1.357e9
            return (abs(x) + x) / 2

        colormap = np.array([cutoff(d) for d in self.np_data[:, 4]])
        colormap /= max(colormap)
        return colormap

    def scatter_plot(self):
        fig, self.ax = plt.subplots()
        plt.rcParams["figure.dpi"] = 100
        xmin = min(self.x)
        xmax = max(self.x)
        ymin = min(self.y)
        ymax = max(self.y)
        delta_x = xmax - xmin
        delta_y = ymax - ymin
        zoom = 0.1
        self.ax.set_xlim(xmin - delta_x * zoom, xmax + delta_x * zoom)
        self.ax.set_ylim(ymin - delta_y * zoom, ymax + delta_y * zoom)

        self.ax.scatter(
            self.x, self.y, marker="+", s=20, c=self.time_colormap()[:], cmap=cm.jet
        )
        self.plot_thumbnails()
        plt.show()

    def plot_thumbnails(self):
        print("add thumbnails to plot")

        t = self.thumbnails[:]
        for i, (x, y, path) in tqdm(enumerate(zip(self.x, self.y, t)), total=len(t)):
            if i % 5 == 0:
                continue

            zoom = self.thumbnail_zoom
            if self.plot_collision(x, y, zoom):
                zoom *= 0.5
                if self.plot_collision(x, y, zoom):
                    zoom *= 0.5
                    if self.plot_collision(x, y, zoom):
                        continue
            
            self.plot_thumbnail(x,y,zoom,path)
            # nearest_neighbour = self.nearest_neighbour(x,y)
            # if nearest_neighbour < 128 * self.thumbnail_zoom:
            #     density_zoom = nearest_neighbour / (128 * self.thumbnail_zoom)

    def plot_thumbnail(self, x, y, zoom,path):
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        w = 0.5 * image.width * zoom
        h = 0.5 * image.height * zoom
        extent = (x - w, x + w, y - h, y + h)
        self.ax.imshow(image, extent=extent)
        self.plotted.append((x, y, zoom))

    def plot_collision(self, x, y, zoom):
        for x2, y2, zoom2 in self.plotted:
            dx = x2 - x
            dy = y2 - y
            if dx ** 2 + dy ** 2 < (self.thumbnail_radius * 0.5 * (zoom + zoom2)) ** 2:
                return True
        return False

    def nearest_neighbour(self, x, y):
        min = self.thumbnail_radius * self.thumbnail_zoom
        for x2, y2 in zip(self.x, self.y):
            dx = x2 - x
            dy = y2 - y
            d2 = math.sqrt(dx ** 2 + dy ** 2)
            if d2 < self.thumbnail_radius * self.thumbnail_zoom:
                continue
            if d2 < min:
                min = d2
        return min


def main():
    
    plotter = Plotter()


if __name__ == "__main__":
    main()
