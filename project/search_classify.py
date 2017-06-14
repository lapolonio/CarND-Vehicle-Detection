import glob
import pickle

import time

from project.lesson_functions import *
from scipy.ndimage.measurements import label

dist_pickle = pickle.load(open("../test_output/svc_pickle.p", "rb"))
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


# image = mpimg.imread('../test_images/bbox-example-image.jpg')
images = glob.glob('../test_images/test*')
fig_images = []
fig_titles = []
for img_path in images:
    image = mpimg.imread(img_path)
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255

    t = time.time()
    x_start_stop = [None, None]
    scaled_box_params = [
        dict(scale=3.0, y_start=[300, 700]),
        dict(scale=2.5, y_start=[300, 650]),
        dict(scale=2.0, y_start=[350, 600]),
        dict(scale=1.5, y_start=[350, 550]),
        dict(scale=1.0, y_start=[400, 450]),
    ]

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    box_list = []

    for box_params in scaled_box_params:
        boxes, window_img = find_cars(image, box_params['y_start'][0], box_params['y_start'][1], box_params['scale'],
                                      svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                                      color_space)
        box_list = boxes + box_list

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    real_heat = apply_threshold(heat, 2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(real_heat)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    window_img = draw_boxes(draw_image, box_list, color=(0, 0, 255), thick=6)

    fig_images.append(window_img)
    fig_titles.append('All Boxes Car')

    fig_images.append(draw_img)
    fig_titles.append('Bounding Box Car')

    fig_images.append(heatmap)
    fig_titles.append('Heat Map')

    fig_images.append(labels[0])
    fig_titles.append('Label')

    print(time.time() - t, 'seconds to process on image searching', len(scaled_box_params), 'scales')

fig = plt.figure(figsize=(12, 18), dpi=300)
visualize(fig, 6, 4, fig_images, fig_titles)
plt.savefig('../test_output/search_classify.jpg')


