import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay

# Import PCA feature data from MAT file
data = scipy.io.loadmat('Real_Human_UCLA_Data/UCLA_GR.mat')
label_data = data['cluster_class']
label_data = np.transpose(label_data)
label_data = label_data[0,:].astype('int')

for label in np.unique(label_data):
    print('{} spikes in group {}'.format(np.count_nonzero(label_data == label), label))
print()

plt.figure()
plt.hist(label_data)
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')

# Import spike data from MAT file
data = scipy.io.loadmat('Real_Human_UCLA_Data/UCLASpikes.mat')
spike_data = data['spikes']
spike_data = np.transpose(spike_data)
spike_time = np.linspace(0,
                         spike_data.shape[0],
                         num = spike_data.shape[0])

# Plot spikes
plt.figure(figsize = (20, 10))
colors = ['blue', 'green', 'red', 'black']
for count, label in enumerate(label_data):
    plt.subplot(2, 2, label + 1)
    plt.plot(spike_time,
             spike_data[:, count],
             color = colors[label_data[count]])
    plt.xlabel('Sample')
    plt.ylabel('Amplitude (uV)')
    plt.title('Spike Group {}'.format(label))
plt.show()

# Import PCA feature data from MAT file
data = scipy.io.loadmat('Real_Human_UCLA_Data/UCLA_PCA_Features.mat')
pca_data = data['X']
pca_data = np.transpose(pca_data)

# Plot first two PCA features
plt.figure()
for count, color in enumerate(colors):
    plt.scatter(pca_data[0,np.argwhere(label_data==count)],
                pca_data[1,np.argwhere(label_data==count)],
                c = color,
                marker = '.',
                alpha = 0.5)
plt.legend(['Spike 1', 'Spike 2', 'Spike 3', 'Spike 4'])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA Features')
plt.show()

# Create a list of integers from 1-10
integers = list(range(1, 11))

# Create an empty list to store the pairs of integers
pairs = []

# Loop through each integer in the list
for i in range(len(integers)):
    # Loop through each integer after the current integer
    for j in range(i+1, len(integers)):
        # Add the pair of integers to the list
        pairs.append((integers[i], integers[j]))


for pair in pairs:
    plt.figure()
    plt.scatter(pca_data[pair[0]-1,:],
                pca_data[pair[1]-1,:])
    plt.xlabel('PC {}'.format(pair[0]-1))
    plt.ylabel('PC {}'.format(pair[1]-1))
    plt.title('PCA Features')
    plt.show()

# Import PCA feature data from MAT file
data = scipy.io.loadmat('Real_Human_UCLA_Data/UCLA_Wavelet_Features.mat')
wavelet_data = data['X']
wavelet_data = np.transpose(wavelet_data)

# Create plotting function for labelled data
def plot_labelled_data(data, labels, xlabel, ylabel, title):
    plt.figure()
    for count, color in enumerate(colors):
        plt.scatter(data[0,np.argwhere(labels==count)],
                    data[1,np.argwhere(labels==count)],
                    c = color,
                    marker = '.',
                    alpha = 0.5)
    plt.legend(['Spike 1', 'Spike 2', 'Spike 3', 'Spike 4'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Plot first two Wavelet features
plot_labelled_data(
    wavelet_data,
    label_data,
    'Wavelet 1', 
    'Wavelet 2', 
    'Wavelet Features')

# "Spike 4" (value = 3 in labels) is likely noise. 
# Always on the fringe of other clusters.

# Fit k-means to data
kmeans = KMeans(n_clusters = len(colors), n_init = 10)

kmeans.fit(np.transpose(pca_data))

labels_kmeans = kmeans.predict(np.transpose(pca_data))



# Plot first two PCA features
plot_labelled_data(
    pca_data,
    labels_kmeans, 
    'PC 1',
    'PC 2', 
    'PCA Features - KMeans')

# Plot first two Wavelet features
plot_labelled_data(
    wavelet_data, 
    labels_kmeans, 
    'Wavelet 1',
    'Wavelet 2', 
    'Wavelet Features - KMeans')


# Try some clustering
# Fit GMM to data
gmm = GaussianMixture(n_components = 4, 
                      covariance_type = 'full')

# Fit the GMM model to the pca_data

gmm.fit(np.transpose(pca_data))

labels_gmm = gmm.predict(np.transpose(pca_data))

# Plot first two PCA features
plot_labelled_data(
    pca_data,
    labels_gmm, 
    'PC 1',
    'PC 2', 
    'PCA Features - KMeans')

# Plot first two Wavelet features
plot_labelled_data(
    wavelet_data, 
    labels_gmm, 
    'Wavelet 1',
    'Wavelet 2', 
    'Wavelet Features - KMeans')


# Try some classification
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    pca_data.T,
    label_data, 
    test_size = 0.2,
    random_state = 42)

classifier_names = [
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
    "Support Vector Machine",
    "Quadratic Discriminant Analysis",
    "K Nearest Neighbors"]

classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    SVC(),
    QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier()]

high_score = 0
for name, clf in zip(classifier_names, classifiers):
    clf = make_pipeline(
        StandardScaler(), 
        clf)
    clf.fit(
        X_train, 
        y_train)
    score = clf.score(
        X_test,
        y_test)
    if score > high_score:
        high_score = score
        high_clf = name
    print(name + " Accuracy: ", "{:.3f}".format(score))
    plt.figure()
    ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test, 
        labels = ["PC 1", "PC 2"],
        cmap = plt.cm.Blues)
    plt.title(name)
   
    

print(high_clf + " had the highest accuracy at", "{:.3f}".format(high_score))



