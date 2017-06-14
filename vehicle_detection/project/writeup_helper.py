import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from project.lesson_functions import *

dist_pickle = pickle.load(open("test_svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
color_space = dist_pickle["color_space"]
hog_channel = dist_pickle["hog_channel"]
spatial_feat = dist_pickle["spatial_feat"]
hist_feat = dist_pickle["hist_feat"]
hog_feat = dist_pickle["hog_feat"]

car_image = mpimg.imread('/Users/lapolonio/Downloads/train_data_set/vehicles_smallset/cars1/1.jpeg')
not_car_image = mpimg.imread('/Users/lapolonio/Downloads/train_data_set/non-vehicles_smallset/notcars1/extra83.jpeg')
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Car')
plt.subplot(122)
plt.imshow(not_car_image)
plt.title('Not Car')
fig.tight_layout()
plt.savefig('../output_images/car_not_car.png')

fig_images = []
fig_titles = []

ctrans_tosearch = convert_color(car_image, conv='RGB2YCrCb')
fig_images.append(car_image)
# axarr[ind,channel+4].imshow(hog_image,cmap='gray')
fig_titles.append("car")

for channel in range(3):
    features, hog_image = get_hog_features(ctrans_tosearch[:, :, channel], orient, pix_per_cell,
                                           cell_per_block, vis=True, feature_vec=True)
    fig_images.append(hog_image)
    fig_titles.append("Car HOG ch {0}".format(channel))

ctrans_tosearch = convert_color(not_car_image, conv='RGB2YCrCb')
fig_images.append(not_car_image)
# axarr[ind,channel+4].imshow(hog_image,cmap='gray')
fig_titles.append("not car")

for channel in range(3):
    features, hog_image = get_hog_features(ctrans_tosearch[:, :, channel], orient, pix_per_cell,
                                           cell_per_block, vis=True, feature_vec=True)
    fig_images.append(hog_image)
    fig_titles.append("Not Car HOG ch {0}".format(channel))

fig = plt.figure(figsize=(10, 7), dpi=300)
visualize(fig, 2, 4, fig_images, fig_titles)
plt.savefig('../output_images/hog_example.jpg')