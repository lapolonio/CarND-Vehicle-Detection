
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from project.lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
import pickle

# Read in cars and notcars
images = glob.glob('/Users/lapolonio/Downloads/train_data_set/vehicles/**/*.png', recursive=True)
cars = []
for image in images:
    cars.append(image)

images = glob.glob('/Users/lapolonio/Downloads/train_data_set/non-vehicles/**/*.png', recursive=True)
notcars = []
for image in images:
    notcars.append(image)

print("number of cars:", len(cars))
print("number of notcars:", len(notcars))

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 10000
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]


### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

# color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# orient = 10  # HOG orientations
# pix_per_cell = 8  # HOG pixels per cell
# cell_per_block = 2  # HOG cells per block
# hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
# spatial_size = (16, 16)  # Spatial binning dimensions
# hist_bins = 64  # Number of histogram bins
# spatial_feat = True  # Spatial features on or off
# hist_feat = True  # Histogram features on or off
# hog_feat = True  # HOG features on or off

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

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
skf = StratifiedKFold(random_state=0, n_splits=10)
scores = []
precisions = []
for train, test in skf.split(X, y):

    t = time.time()
    svc.fit(scaled_X[train], y[train])
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    score = round(svc.score(scaled_X[test], y[test]), 4)
    print('Test Accuracy of SVC = ', score)
    precision = precision_score(y[test], svc.predict(scaled_X[test]), average='micro')
    print('Precision of SVC = ', precision)
    scores.append(score)
    precisions.append(precision)

# Check the score of the SVC
print('Overall Test Accuracy of SVC = ', np.array(scores).mean())
print('Overall Test Precision of SVC = ', np.array(precisions).mean())

svc.fit(scaled_X, y)
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
pickle.dump(dist_pickle, open("../test_output/svc_pickle.p", "wb"))