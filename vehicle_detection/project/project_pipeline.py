from moviepy.editor import VideoFileClip
import pickle
from project.lesson_functions import *
from scipy.ndimage.measurements import label


dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
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

from collections import deque
queue = deque(maxlen=5)


# Video Processing
def image_processor(image):
    global queue

    scaled_box_params = [
        dict(scale=2.0, y_start=[400, 656]),
        dict(scale=1.5, y_start=[400, 656]),
        dict(scale=1.25, y_start=[400, 656]),
    ]

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    box_list = []

    for box_params in scaled_box_params:
        boxes, window_img = find_cars(image, box_params['y_start'][0], box_params['y_start'][1], box_params['scale'],
                                         svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        box_list = boxes + box_list

    queue.append(box_list)

    box_history = []

    for x in queue:
        box_history = box_history + x

    # Add heat to each box in box list
    heat = add_heat(heat, box_history)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 15)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img

project_output_name = '../output_images/output_project_video.mp4'
clip2 = VideoFileClip('../project_video.mp4')
# project_clip = clip2.subclip(44, 48).fl_image(image_processor)
project_clip = clip2.fl_image(image_processor)
project_clip.write_videofile(project_output_name, audio=False)
