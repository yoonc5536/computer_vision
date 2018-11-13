"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math

import ps6

# I/O directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "output"

YALE_FACES_DIR = os.path.join(INPUT_DIR, 'Yalefaces')
FACES94_DIR = os.path.join(INPUT_DIR, 'faces94')
POS_DIR = os.path.join(INPUT_DIR, "pos")
NEG_DIR = os.path.join(INPUT_DIR, "neg")
NEG2_DIR = os.path.join(INPUT_DIR, "neg2")


def load_images_from_dir(data_dir, size=(24, 24), ext=".png"):
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imagesFiles.sort()
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    imgs = [cv2.resize(x, size) for x in imgs]

    return imgs

# Utility function
def plot_eigen_faces(eig_vecs, fig_name="", visualize=False):
    r = np.ceil(np.sqrt(len(eig_vecs)))
    c = int(np.ceil(len(eig_vecs)/r))
    r = int(r)
    fig = plt.figure()

    for i,v in enumerate(eig_vecs):
        sp = fig.add_subplot(r,c,i+1)


        plt.imshow(v.reshape(32,32).real, cmap='gray')
        sp.set_title('eigenface_%i'%i)
        sp.axis('off')

    fig.subplots_adjust(hspace=.5)

    if visualize:
        plt.show()

    if not fig_name == "":
        plt.savefig("output/{}".format(fig_name))


# Functions you need to complete
def visualize_mean_face(x_mean, size, new_dims):
    """Rearrange the data in the mean face to a 2D array

    - Organize the contents in the mean face vector to a 2D array.
    - Normalize this image.
    - Resize it to match the new dimensions parameter

    Args:
        x_mean (numpy.array): Mean face values.
        size (tuple): x_mean 2D dimensions
        new_dims (tuple): Output array dimensions

    Returns:
        numpy.array: Mean face uint8 2D array.
    """
    mean = x_mean.reshape(size).astype(np.uint8)
    return cv2.resize(mean, new_dims, interpolation = cv2.INTER_CUBIC)

def part_1a_1b():

    orig_size = (192, 231)
    small_size = (32, 32)
    X, y = ps6.load_images(YALE_FACES_DIR, small_size)

    # Get the mean face
    x_mean = ps6.get_mean_face(X)

    x_mean_image = visualize_mean_face(x_mean, small_size, orig_size)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "ps6-1-a-1.png"), x_mean_image)

    # PCA dimension reduction
    k = 10
    eig_vecs, eig_vals = ps6.pca(X, k)
    plot_eigen_faces(eig_vecs.T, "ps6-1-b-1.png")


def part_1c():
    runk = 1
    runp = 1
    if runk:
        p = 0.5  # Select a split percentage value
        ks = []
        accuracy = []
        for k in range(1,30):
            size = (32, 32)
            X, y = ps6.load_images(YALE_FACES_DIR, size)
            Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)

            # training
            mu = ps6.get_mean_face(Xtrain)
            eig_vecs, eig_vals = ps6.pca(Xtrain, k)
            Xtrain_proj = np.dot(Xtrain - mu, eig_vecs)

            # testing
            mu = ps6.get_mean_face(Xtest)
            Xtest_proj = np.dot(Xtest - mu, eig_vecs)

            good = 0
            bad = 0

            for i, obs in enumerate(Xtest_proj):

                dist = [np.linalg.norm(obs - x) for x in Xtrain_proj]

                idx = np.argmin(dist)
                y_pred = ytrain[idx]

                if y_pred == ytest[i]:
                    good += 1
                else:
                    bad += 1

            print 'Good predictions = ', good, 'Bad predictions = ', bad
            ks.append(k)
            accuracy.append(100 * float(good)/(good + bad))
            print '{0:.2f}% accuracy'.format(100 * float(good)/(good + bad))

        plt.figure()
        plt.ylabel('Accuracy')
        plt.xlabel('# of PCs')
        plt.title('Accuracy vs Number of PCs ')
        plt.plot(ks,accuracy)
        plt.grid()
        plt.draw()
        plt.savefig('./pca_plot.png')
        #plt.show()

    if runp:
        k = 10
        accuracy = []
        ps = []
        for p in np.arange(0.2, 1.0, 0.1):
            size = (32, 32)
            X, y = ps6.load_images(YALE_FACES_DIR, size)
            Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)

            # training
            mu = ps6.get_mean_face(Xtrain)
            eig_vecs, eig_vals = ps6.pca(Xtrain, k)
            Xtrain_proj = np.dot(Xtrain - mu, eig_vecs)

            # testing
            mu = ps6.get_mean_face(Xtest)
            Xtest_proj = np.dot(Xtest - mu, eig_vecs)

            good = 0
            bad = 0

            for i, obs in enumerate(Xtest_proj):

                dist = [np.linalg.norm(obs - x) for x in Xtrain_proj]

                idx = np.argmin(dist)
                y_pred = ytrain[idx]

                if y_pred == ytest[i]:
                    good += 1
                else:
                    bad += 1

            print 'Good predictions = ', good, 'Bad predictions = ', bad
            ps.append(p)
            accuracy.append(100 * float(good)/(good + bad))
            print '{0:.2f}% accuracy'.format(100 * float(good)/(good + bad))

        plt.figure()
        plt.ylabel('Accuracy')
        plt.xlabel('Percentage of data split')
        plt.title('Accuracy vs data split percentage ')
        plt.plot(ps,accuracy)
        plt.grid()
        plt.draw()
        plt.savefig('./split_P_plot.png')
        #plt.show()
def part_2a():
    y0 = 1
    y1 = 2

    X, y = ps6.load_images(FACES94_DIR)

    # Select only the y0 and y1 classes
    idx = y == y0
    idx |= y == y1

    X = X[idx,:]
    y = y[idx]

    # Label them 1 and -1
    y0_ids = y == y0
    y1_ids = y == y1
    y[y0_ids] = 1
    y[y1_ids] = -1

    p = 0.5
    Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)

    # Picking random numbers
    rand_y = np.random.choice([-1, 1], (len(ytrain)))
    # TODO: find which of these labels match ytrain and report its accuracy
    good = 0
    for i in range(0,len(ytrain)):
        if(rand_y[i] == ytrain[i]):
            good += 1

    rand_accuracy = float(good)/len(ytrain) * 100
    print '(Random) Training accuracy: {0:.2f}%'.format(rand_accuracy)

    # Using Weak Classifier
    uniform_weights = np.ones((Xtrain.shape[0],)) / Xtrain.shape[0]
    wk_clf = ps6.WeakClassifier(Xtrain, ytrain, uniform_weights)
    wk_clf.train()
    wk_results = [wk_clf.predict(x) for x in Xtrain]
    # TODO: find which of these labels match ytrain and report its accuracy
    good = 0
    for i in range(0,len(ytrain)):
        if(wk_results[i] == ytrain[i]):
            good += 1
    wk_accuracy = float(good)/len(ytrain) * 100
    print '(Weak) Training accuracy {0:.2f}%'.format(wk_accuracy)

    num_iter = 5

    boost = ps6.Boosting(Xtrain, ytrain, num_iter)
    boost.train()
    good, bad = boost.evaluate()
    boost_accuracy = 100 * float(good) / (good + bad)
    print '(Boosting) Training accuracy {0:.2f}%'.format(boost_accuracy)

    # Picking random numbers
    rand_y = np.random.choice([-1, 1], (len(ytest)))
    # TODO: find which of these labels match ytest and report its accuracy
    good = 0
    for i in range(0,len(ytest)):
        if(rand_y[i] == ytest[i]):
            good += 1
    rand_accuracyt = float(good)/len(ytest) * 100
    print '(Random) Testing accuracy: {0:.2f}%'.format(rand_accuracyt)

    # Using Weak Classifier
    wk_results = [wk_clf.predict(x) for x in Xtest]
    # TODO: find which of these labels match ytest and report its accuracy
    good = 0
    for i in range(0,len(ytest)):
        if(wk_results[i] == ytest[i]):
            good += 1
    wk_accuracyt = float(good)/len(ytest) * 100
    print '(Weak) Testing accuracy {0:.2f}%'.format(wk_accuracyt)

    y_pred = boost.predict(Xtest)
    # TODO: find which of these labels match ytest and report its accuracy
    good = 0
    for i in range(0,len(ytest)):
        if(y_pred[i] == ytest[i]):
            good += 1
    boost_accuracyt = float(good)/len(ytest) * 100
 
    print '(Boosting) Testing accuracy {0:.2f}%'.format(boost_accuracyt)
    return rand_accuracy,wk_accuracy,boost_accuracy,rand_accuracyt,wk_accuracyt,boost_accuracyt

def part_3a():
    """Complete the remaining parts of this section as instructed in the
    instructions document."""

    feature1 = ps6.HaarFeature((2, 1), (25, 30), (50, 100))
    feature1.preview((200, 200), filename="ps6-3-a-1.png")

    feature2 = ps6.HaarFeature((1, 2), (10, 25), (50, 150))
    feature2.preview((200, 200), filename="ps6-3-a-2.png")

    feature3 = ps6.HaarFeature((3, 1), (50, 50), (100, 50))
    feature3.preview((200, 200), filename="ps6-3-a-3.png")

    feature4 = ps6.HaarFeature((1, 3), (50, 125), (100, 50))
    feature4.preview((200, 200), filename="ps6-3-a-4.png")

    feature5 = ps6.HaarFeature((2, 2), (50, 25), (100, 150))
    feature5.preview((200, 200), filename="ps6-3-a-5.png")

def part_4_a_b():

    pos = load_images_from_dir(POS_DIR)
    neg = load_images_from_dir(NEG_DIR)
    train_pos = pos[:35]
    train_neg = neg[:]
    images = train_pos + train_neg
    labels = np.array(len(train_pos) * [1] + len(train_neg) * [-1])

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(train_pos, train_neg, integral_images)
    VJ.createHaarFeatures()

    VJ.train(4)

    VJ.haarFeatures[VJ.classifiers[0].feature].preview(filename="ps6-4-b-1")
    VJ.haarFeatures[VJ.classifiers[1].feature].preview(filename="ps6-4-b-2")
    VJ.haarFeatures[VJ.classifiers[2].feature].preview(filename="ps6-4-b-3")
    VJ.haarFeatures[VJ.classifiers[3].feature].preview(filename="ps6-4-b-4")
    predictions = VJ.predict(images)

    good = 0
    for i in range(0, len(predictions)):
        if(predictions[i] == labels[i]):
            good += 1

    vj_accuracy = float(good)/len(predictions) * 100
    print "Prediction accuracy on training: {0:.2f}%".format(vj_accuracy)

    neg = load_images_from_dir(NEG2_DIR)

    test_pos = pos[35:]
    test_neg = neg[:35]
    test_images = test_pos + test_neg
    real_labels = np.array(len(test_pos) * [1] + len(test_neg) * [-1])
    predictions = VJ.predict(test_images)

    good = 0
    for i in range(0, len(predictions)):
        if(predictions[i] == real_labels[i]):
            good += 1

    vj_accuracy = float(good)/len(predictions) * 100
    print "Prediction accuracy on testing: {0:.2f}%".format(vj_accuracy)


def part_4_c():
    pos = load_images_from_dir(POS_DIR)[:20]
    neg = load_images_from_dir(NEG_DIR)

    images = pos + neg

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(pos, neg, integral_images)
    VJ.createHaarFeatures()

    VJ.train(4)

    image = cv2.imread(os.path.join(INPUT_DIR, "man.jpeg"), -1)
    imagerz = cv2.resize(image, (120, 60))
    VJ.faceDetection(imagerz, filename="ps4-4-c-1")


if __name__ == "__main__":
    print("Running 1a_1b...\n")
    part_1a_1b()
    print("Running 1c...\n")
    part_1c()
    print("Running 2a...\n")
    accuracies = [0,0,0,0,0,0]
    name = ['(Random) Training accuracy','(Weak) Training accuracy','(Boosting) Training accuracy','(Random) Testing accuracy:',
            '(Weak) Testing accuracy','(Boosting) Testing accuracy']

    for i in range(1,6):
        print('Iteration:',i)
        ret = part_2a()
        for j in range(0,len(accuracies)):
            accuracies[j] = (accuracies[j] + ret[j])/float(2.0)
    for i in range(0,len(name)):
        print(name[i],"{0:.2f} %".format(accuracies[i]))

    print("Running 3a...\n")
    part_3a()
    
    print("Running 4a_b...\n")
    part_4_a_b()
    print("Running 4c..\n")
    part_4_c()