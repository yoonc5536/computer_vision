"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   ([int]): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """
    ext = ".png"
    imgs = []
    imagesFiles = [f for f in os.listdir(folder) if f.endswith(ext)]
    labels = []

    for f in imagesFiles:
        imgs.append(np.array(cv2.imread(os.path.join(folder, f), 0)))
        imgNum = int(f.split('.')[0].split('subject')[1])
        labels.append(imgNum)
    imgs = [np.array(cv2.resize(x, size)).flatten() for x in imgs]
    labels = np.asarray(labels, dtype=int)
    imgs = np.asarray(imgs)
    return imgs,labels

def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    imgIdx = np.arange(X.shape[0])
    np.random.shuffle(imgIdx)
    trainIdx = imgIdx[0:int(len(imgIdx)*p)]
    testIdx = imgIdx[int(len(imgIdx)*p):len(imgIdx)]

    Xtrain = X[trainIdx]
    Ytrain = y[trainIdx]
    Xtest = X[testIdx]
    Ytest = y[testIdx]

    return Xtrain,Ytrain,Xtest,Ytest

def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    return np.mean(x,axis=0)
def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """

    mean_face = get_mean_face(X)
    X = X - mean_face
    C = np.dot(X.T, X)
    w,v = np.linalg.eig(C)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]

    v = v[:,0:k]
    w = w[0:k]
    return v,w 

class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        # Adaboosting algorithm
        for i in range(0, self.num_iterations):
            #a) renormalize the weights
            self.weights /= self.weights.sum()

            #b) instantiate the weak classifier
            h = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            h.train()
            hResult = [h.predict(i) for i in self.Xtrain]
            self.weakClassifiers.append(h)

            #c) find ej for weights where h(xi) != y(i)
            ej = 0.0
            for i in range(0,len(self.ytrain)):
                if(hResult[i] != self.ytrain[i]):
                    ej += self.weights[i]
            
            #d) calculate aj
            aj = np.log((1-ej)/self.eps) * 0.5
            self.alphas.append(aj)

            #e) if ej is greater than the threshold self.eps -> update weights, else stop the loop
            if(ej > self.eps):
                for i in range(0,len(self.ytrain)):
                    if(hResult[i] != self.ytrain[i]):
                        self.weights[i] = self.weights[i] * np.exp(-self.ytrain[i]*hResult[i]*aj)
            else:
                break

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        predicts = self.predict(self.Xtrain)
        good = 0
        for i in range(0,len(self.ytrain)):
            if(predicts[i] == self.ytrain[i]):
                good += 1

        return good,len(self.ytrain)-good

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        predicts = []
        for i in range(0,len(X)):
            sign = 0.0
            for j in range(0,len(self.weakClassifiers)):
                sign += self.weakClassifiers[j].predict(X[i]) * self.alphas[j]
            if(sign < 0):
                predicts.append(-1)
            else:
                predicts.append(+1)
        return np.array(predicts)

class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size
        self.px,self.py = position
        self.sx,self.sy = size
    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        cv2.rectangle(img, (self.py, self.px), (self.py + self.sy - 1, self.px + self.sx/2 - 1), (255), -1)
        cv2.rectangle(img, (self.py, self.px + self.sx/2), (self.py + self.sy - 1, self.px + self.sx - 1), (126), -1)

        return img

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)

        cv2.rectangle(img, (self.py, self.px), (self.py + self.sy/2 - 1, self.px + self.sx - 1), (255), -1)
        cv2.rectangle(img, (self.py + self.sy/2, self.px), (self.py +  self.sy - 1, self.px + self.sx - 1), (126), -1)
        return img

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        cv2.rectangle(img, (self.py, self.px), (self.py + self.sy - 1, self.px + self.sx/3 - 1), (255), -1)
        cv2.rectangle(img, (self.py, self.px + self.sx/3), (self.py + self.sy - 1, self.px + (self.sx/3 * 2) - 1), (126), -1)
        cv2.rectangle(img, (self.py, self.px + (self.sx/3 * 2)), (self.py + self.sy - 1, self.px + self.sx - 1), (255), -1)

        return img

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        cv2.rectangle(img, (self.py, self.px), (self.py + self.sy/3 - 1, self.px + self.sx - 1), (255), -1)
        cv2.rectangle(img, (self.py + self.sy/3, self.px), (self.py + (self.sy/3 * 2) - 1, self.px + self.sx - 1), (126), -1)
        cv2.rectangle(img, (self.py + (self.sy/3 * 2), self.px), (self.py + self.sy - 1, self.px + self.sx - 1), (255), -1)

        return img

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape)
        cv2.rectangle(img, (self.py, self.px), (self.py + self.sy/2 - 1, self.px + self.sx/2 - 1), (126), -1)
        cv2.rectangle(img, (self.py + self.sy/2, self.px), (self.py + self.sy - 1, self.px + self.sx/2 - 1), (255), -1)
        cv2.rectangle(img, (self.py, self.px + self.sx/2), (self.py + self.sy/2 - 1, self.px + self.sx - 1), (255), -1)
        cv2.rectangle(img, (self.py + self.sy/2, self.px + self.sx/2), (self.py + self.sy - 1, self.px + self.sx - 1), (126), -1)

        return img

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def regionSum(self, ii, tl, br):
        ii = ii.astype('float64')
        #tl = (tl[0]+1,tl[1]+1)
        #br = (br[0]+1,br[1]+1)
        return ii[br[1]][br[0]] - ii[br[1]][tl[0]] - ii[tl[1]][br[0]] + ii[tl[1]][tl[0]]

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        
        if self.feat_type == (1, 2):  
            # two_vertical
            sumWhite = self.regionSum(ii, (self.py - 1, self.px - 1), (self.py + self.sy/2 - 1, self.px + self.sx - 1))
            sumGrey = self.regionSum(ii, (self.py + self.sy/2 - 1, self.px - 1), (self.py +  self.sy - 1, self.px + self.sx - 1))
            return (sumWhite - sumGrey)

        elif self.feat_type == (2, 1):  
            # two_horizontal
            sumWhite = self.regionSum(ii, (self.py - 1, self.px - 1), (self.py + self.sy - 1, self.px + self.sx/2 - 1))
            sumGrey = self.regionSum(ii, (self.py - 1, self.px + self.sx/2 - 1), (self.py + self.sy - 1, self.px + self.sx - 1))
            return (sumWhite - sumGrey)

        elif self.feat_type == (3, 1): 
            # three_horizontal
            sumWhite1 = self.regionSum(ii, (self.py - 1, self.px - 1), (self.py + self.sy - 1, self.px + self.sx/3 - 1))
            sumWhite2 = self.regionSum(ii, (self.py - 1, self.px + (self.sx/3 * 2) - 1), (self.py + self.sy - 1, self.px + self.sx - 1))
            sumGrey = self.regionSum(ii, (self.py - 1, self.px + self.sx/3 - 1), (self.py + self.sy - 1, self.px + (self.sx/3 * 2) - 1))
            return (sumWhite1 + sumWhite2 - sumGrey)

        elif self.feat_type == (1, 3): 
            # three_vertical
            sumWhite1 = self.regionSum(ii, (self.py - 1, self.px - 1), (self.py + self.sy/3  - 1, self.px + self.sx - 1))
            sumWhite2 = self.regionSum(ii, (self.py + (self.sy/3 * 2) - 1, self.px - 1), (self.py + self.sy - 1, self.px + self.sx - 1))
            sumGrey = self.regionSum(ii, (self.py + self.sy/3  - 1, self.px - 1), (self.py + (self.sy/3  * 2) - 1, self.px + self.sx - 1))
            return (sumWhite1 + sumWhite2 - sumGrey)

        elif self.feat_type == (2, 2): 
            # four_square 
            sumWhite1 = self.regionSum(ii, (self.py + self.sy/2 - 1, self.px - 1), (self.py + self.sy - 1, self.px + self.sx/2 - 1))
            sumWhite2 = self.regionSum(ii, (self.py - 1, self.px + self.sx/2 - 1), (self.py + self.sy/2 - 1, self.px + self.sx - 1))
            sumGrey1 = self.regionSum(ii, (self.py - 1, self.px - 1), (self.py + self.sy/2 - 1, self.px + self.sx/2 - 1))
            sumGrey2 = self.regionSum(ii, (self.py + self.sy/2 - 1, self.px + self.sx/2 - 1), (self.py + self.sy - 1, self.px + self.sx - 1))
            return (sumWhite1 + sumWhite2 - sumGrey2 - sumGrey1)

        return 0


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """

    i_img = []
    for img in images:
        img = np.cumsum(np.cumsum(img,0), 1)
        i_img.append(img)
    return i_img

class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.iteritems():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print " -- compute all scores --"
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print " -- select classifiers --"

        for i in range(num_classifiers):
            weights_pos /= weights_pos.sum()
            weights_neg /= weights_neg.sum()
            hj = VJ_Classifier(scores, self.labels, weights)
            hj.train()
            self.classifiers.append(hj) 
            beta = hj.error / (1-hj.error)
            alpha = np.log(1/beta)
            for i in range(0, len(weights)):
                ei = 1
                if(hj.predict(scores[i]) == self.labels[i]):
                    ei = -1
                weights[i] = weights[i] * (beta**(1-ei))
            
            self.alphas.append(alpha)

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)
        scores = np.zeros((len(ii), len(self.haarFeatures)))

        for clf in self.classifiers:
            featureID = clf.feature
            hf = self.haarFeatures[featureID]
            for x, im in enumerate(ii):
                scores[x, featureID] = hf.evaluate(im)
        result = []

        threshold = np.array(self.alphas).sum()
        for x in scores:
            scoreSum = 0
            i = 0
            for clf in self.classifiers:
                scoreSum += clf.predict(x) * self.alphas[i]
                i += 1
            if(scoreSum  < threshold*0.5):
                result.append(-1)
            else:
                result.append(1)

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x_points = []
        y_points = []
        slices = []
        for x in range(0, img.shape[0] - 24):
            for y in range(0, img.shape[1] - 24):
                small_slice = img[x:x+24, y:y+24]
                prediction = self.predict([small_slice])
                if(prediction[0] == 1):
                    x_points.append(x)
                    y_points.append(y)
                    resized_x,resized_y = (x,y)
                    #cv2.rectangle(image, (resized_y, resized_x), (resized_y + 24, resized_x + 24), (255, 0, 0))
        average_point = (int(np.average(x_points)), int(np.average(y_points)))
        resized_x,resized_y = average_point
        cv2.rectangle(image, (resized_y, resized_x), (resized_y + 24, resized_x + 24), (0, 255, 0))

        cv2.imwrite("output/{}.png".format(filename), image)
