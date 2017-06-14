
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from project.lesson_functions import *
import pickle


def get_training_images():
    global cars, notcars
    ext = ['.png', '.jpeg', '.jpg']
    # Read in cars and notcars
    images = glob.glob('/Users/lapolonio/Downloads/train_data_set/vehicles_smallset/**/*', recursive=True)
    cars = []
    for image in images:
        if image.endswith(tuple(ext)):
            cars.append(image)
    images = glob.glob('/Users/lapolonio/Downloads/train_data_set/non-vehicles_smallset/**/*', recursive=True)
    notcars = []
    for image in images:
        if image.endswith(tuple(ext)):
            notcars.append(image)
    print("number of cars:", len(cars))
    print("number of notcars:", len(notcars))
    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 2000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    return cars, notcars


cars, notcars = get_training_images()


### TODO: Tweak these parameters and see how the results change.
# color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# orient = 8  # HOG orientations
# pix_per_cell = 8  # HOG pixels per cell
# cell_per_block = 2  # HOG cells per block
# hog_channel = 0  # Can be 0, 1, 2, or "ALL"
# spatial_size = (16, 16)  # Spatial binning dimensions
# hist_bins = 16  # Number of histogram bins
# spatial_feat = True  # Spatial features on or off
# hist_feat = True  # Histogram features on or off
# hog_feat = True  # HOG features on or off

color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 64  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

# color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# orient = 9  # HOG orientations
# pix_per_cell = 8  # HOG pixels per cell
# cell_per_block = 2  # HOG cells per block
# hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
# spatial_size = (16, 16)  # Spatial binning dimensions
# hist_bins = 32  # Number of histogram bins
# spatial_feat = True  # Spatial features on or off
# hist_feat = True  # Histogram features on or off
# hog_feat = True  # HOG features on or off

t_feat = time.time()

car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

t2 = time.time()
print(round(t2 - t_feat, 2), 'Seconds to extract features...')

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(scaled_X[0]))
# Use a linear SVC
svc = LinearSVC()
# Split up data into randomized training and test sets
skf = StratifiedKFold(random_state=0, n_splits=10)
scores = []
for train, test in skf.split(X, y):

    # Check the training time for the SVC
    t = time.time()
    svc.fit(scaled_X[train], y[train])
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    score = round(svc.score(scaled_X[test], y[test]), 4)
    print('Test Accuracy of SVC = ', score)
    scores.append(score)
print('Overall Test Accuracy of SVC = ', np.array(scores).mean())
    # svc.fit(X_train, y_train)
# Check the score of the SVC
# Check the prediction time for a single sample
t = time.time()

dist_pickle = dict()
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
dist_pickle["color_space"] = color_space
dist_pickle["hog_channel"] = hog_channel
dist_pickle["spatial_feat"] = spatial_feat
dist_pickle["hist_feat"] = hist_feat
dist_pickle["hog_feat"] = hog_feat
pickle.dump(dist_pickle, open("../../test_output/svc_pickle.p", "wb"))