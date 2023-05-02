import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy

# Import PCA feature data from MAT file
data = scipy.io.loadmat('UCLA_GR.mat')
label_data = data['cluster_class']
label_data = np.transpose(label_data)
label_data = label_data[:,0].astype('int')

plt.figure()
plt.plot(label_data[:,1])
plt.xlabel('Sample')
plt.ylabel('Amplitude (uV)')
plt.title('Spikes')
plt.show()

# Import spike data from MAT file
data = scipy.io.loadmat('UCLASpikes.mat')
spike_data = data['spikes']
spike_data = np.transpose(spike_data)
spike_time = np.linspace(0,
                         spike_data.shape[0],
                         num = spike_data.shape[0])

# Plot spikes
plt.figure()
colors = ['blue', 'green', 'red', 'black']
for count, label in enumerate(label_data):
    if label == 0:
        plt.subplot(2, 2, 1)
        plt.plot(spike_time,
                 spike_data[:,count],
                 color = colors[label_data[count]])
        plt.xlabel('Sample')
        plt.ylabel('Amplitude (uV)')
        plt.title('Spike Group {}'.format(label))
    elif label == 1:
        plt.subplot(2, 2, 2)
        plt.plot(spike_time,
                 spike_data[:,count],
                 color = colors[label_data[count]])
        plt.xlabel('Sample')
        plt.ylabel('Amplitude (uV)')
        plt.title('Spike Group {}'.format(label))
    elif label == 2:
        plt.subplot(2, 2, 3)
        plt.plot(spike_time,
                 spike_data[:,count],
                 color = colors[label_data[count]])
        plt.xlabel('Sample')
        plt.ylabel('Amplitude (uV)')
        plt.title('Spike Group {}'.format(label))
    elif label == 3:
        plt.subplot(2, 2, 4)
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
for plot_num in range(pca_data.shape[1]):
    plt.scatter(pca_data[0,plot_num],
                pca_data[1,plot_num],
                c = colors[label_data[plot_num]])
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
plt.scatter(wavelet_data[0,:],
            wavelet_data[1,:])
plt.xlabel('Wavelet 1')
plt.ylabel('Wavelet 2')
plt.title('Wavelet Features')
plt.show()