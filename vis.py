import os, sys
import h5py
import matplotlib.pyplot as plt

folder = sys.argv[1]

# read signal from hdf5 file
for fname in [f for f in os.listdir(folder) if 'output' in f]:
    label_fname = fname.replace('output', 'label')
    fname = os.path.join(folder, fname)
    label_fname = os.path.join(folder, label_fname)

    with h5py.File(fname, 'r') as f:
        predictions = f['respiration'][:]
    with h5py.File(label_fname, 'r') as f:
        labels = f['respiration'][:]
    
    # plot predictions and labels
    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='labels')
    plt.legend()
    plt.title(fname)
    plt.show()
    plt.close()