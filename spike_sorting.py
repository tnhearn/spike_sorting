import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import pandas as pd
from sklearn.cluster import KMeans, OPTICS

# Import PCA feature data from MAT file
data = scipy.io.loadmat('UCLA_GR.mat')
label_data = data['cluster_class']
label_data = np.transpose(label_data)
label_data = label_data[:,0].astype('int')

for label in np.unique(label_data):
    print('{} spikes in group {}'.format(np.count_nonzero(label_data == label), label))

plt.figure()
plt.hist(label_data)
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')

# Import spike data from MAT file
data = scipy.io.loadmat('UCLASpikes.mat')
spike_data = data['spikes']
spike_data = np.transpose(spike_data)
spike_time = np.linspace(0,
                         spike_data.shape[0],
                         num = spike_data.shape[0])

# Plot spikes
plt.figure(figsize=(20,10))
colors = ['blue', 'green', 'red', 'black']
for count, label in enumerate(label_data):
    plt.subplot(2, 2, label+1)
    plt.plot(spike_time,
             spike_data[:,count],
             color = colors[label_data[count]])
    plt.xlabel('Sample')
    plt.ylabel('Amplitude (uV)')
    plt.title('Spike Group {}'.format(label))
plt.show()

# Import PCA feature data from MAT file
data = scipy.io.loadmat('UCLA_PCA_Features.mat')
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

# Print the list of pairs of integers
print(pairs)

for pair in pairs:
    plt.figure()
    plt.scatter(pca_data[pair[0]-1,:],
                pca_data[pair[1]-1,:])
    plt.xlabel('PC {}'.format(pair[0]-1))
    plt.ylabel('PC {}'.format(pair[1]-1))
    plt.title('PCA Features')
    plt.show()

# Import PCA feature data from MAT file
data = scipy.io.loadmat('UCLA_Wavelet_Features.mat')
wavelet_data = data['X']
wavelet_data = np.transpose(wavelet_data)

# Plot first two Wavelet features
plt.figure()
for count, color in enumerate(colors):
    plt.scatter(wavelet_data[0,np.argwhere(label_data==count)],
                wavelet_data[1,np.argwhere(label_data==count)],
                c = color,
                marker = '.',
                alpha = 0.5)
plt.legend(['Spike 1', 'Spike 2', 'Spike 3', 'Spike 4'])
plt.xlabel('Wavelet 1')
plt.ylabel('Wavelet 2')
plt.title('Wavelet Features')
plt.show()

# "Spike 4" (value = 3 in labels) is likely noise. 
# Always on the fringe of other clusters.

# Fit k-means to data
kmeans = KMeans(n_clusters = len(colors),
                verbose = 1)

kmeans.fit(np.transpose(pca_data))

labels_kmeans = kmeans.predict(np.transpose(pca_data))

# Plot first two PCA features
plt.figure()
for count, color in enumerate(colors):
    plt.scatter(pca_data[0,np.argwhere(labels_kmeans==count)],
                pca_data[1,np.argwhere(labels_kmeans==count)],
                c = color,
                marker = '.',
                alpha = 0.5)
plt.legend(['Spike 1', 'Spike 2', 'Spike 3', 'Spike 4'])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA Features - KMeans')
plt.show()


# Fit OPTICS to data
optics = OPTICS(min_samples=10,
                min_cluster_size = 100,
                xi = 0.005)

optics.fit(np.transpose(pca_data))

labels_optics = optics.labels_

# Plot first two PCA features
plt.figure()
for count, color in enumerate(colors):
    plt.scatter(pca_data[0,np.argwhere(labels_optics==count)],
                pca_data[1,np.argwhere(labels_optics==count)],
                c = color,
                marker = '.',
                alpha = 0.5)
plt.legend(['Spike 1', 'Spike 2', 'Spike 3', 'Spike 4'])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA Features - OPTICS')
plt.show()
