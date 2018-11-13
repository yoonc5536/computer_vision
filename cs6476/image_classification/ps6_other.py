"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier

def showImage(img):
  cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

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

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]

    X = []
    y = []

    for image_file in images_files:
        img = cv2.imread(os.path.join(folder, image_file), 0)
        small_image = cv2.resize(img, size)
        small_image = small_image.flatten()
        X.append(small_image)

        label = image_file[image_file.find('.') + 1 : image_file.rfind('.')]
        y.append(label)

    X = np.array(X, np.uint8)
    y = np.array(y, np.int8)

    return (X, y)



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




    
    seed = np.arange(X.shape[0])

    np.random.shuffle(seed)

    midpoint = int(seed.shape[0] * p)

    train_seed = seed[0:midpoint]
    test_seed = seed[midpoint:seed.shape[0]]

    Xtrain = X[train_seed]
    Ytrain = y[train_seed]

    Xtest = X[test_seed]
    Ytest = y[test_seed]

    return (Xtrain, Ytrain, Xtest, Ytest)
    


    raise NotImplementedError


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    mean_face = np.mean(x, axis=0)

    
    return mean_face


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

    eig_val, eig_vec = np.linalg.eig(C)
    

    idx = eig_val.argsort()[::-1]   
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]


    return (eig_vec[:, 0:k], eig_val[0:k])

    
    
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

        for i in range(0, self.num_iterations):
            self.weights /= self.weights.sum()
            wk_clf = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            wk_clf.train()
            wk_results = [wk_clf.predict(x) for x in self.Xtrain]

            eJ = 0.0

            for i in range(0, len(self.ytrain)):
                if(self.ytrain[i] != wk_results[i]):
                    eJ += self.weights[i]

            aJ = .5 * np.log((1-eJ)/self.eps)

            self.weakClassifiers.append(wk_clf)
            self.alphas.append(aJ)

            if(eJ > self.eps):
                for i in range(0, len(self.ytrain)):
                    if(self.ytrain[i] != wk_results[i]):
                        self.weights[i] = self.weights[i] * np.exp(-self.ytrain[i] * aJ * wk_results[i])
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

        predictions = self.predict(self.Xtrain)

        good = 0
        bad = 0

        for i in range(0, len(self.ytrain)):
            if(self.ytrain[i] == predictions[i]):
                good += 1

            else:
                bad += 1

        return(good, bad)


    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        
        self.predictions = []
        for i in range(0, len(X)):
            sign = 0.0

            for j in range(0, len(self.weakClassifiers)):
                sign += self.alphas[j] * self.weakClassifiers[j].predict(X[i])

            if(sign > 0):
                self.predictions.append(1)
            else:
                self.predictions.append(-1)


        self.predictions = np.array(self.predictions)
        return self.predictions





class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (x, y) position of the feature's top left corner.
        size (tuple): Feature's (width, height)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        row_height = self.size[0] / 2

        black_image = np.zeros(shape)


        cv2.rectangle(black_image, (self.position[1], self.position[0]), (self.position[1] + self.size[1] - 1, self.position[0] + row_height - 1), (255), -1)

        cv2.rectangle(black_image, (self.position[1], self.position[0] + row_height), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1), (126), -1)

        return black_image


    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        column_width = self.size[1] / 2

        black_image = np.zeros(shape)


        cv2.rectangle(black_image, (self.position[1], self.position[0]), (self.position[1] + column_width - 1, self.position[0] + self.size[0] - 1), (255), -1)

        cv2.rectangle(black_image, (self.position[1] + column_width, self.position[0]), (self.position[1] +  self.size[1] - 1, self.position[0] + self.size[0] - 1), (126), -1)

        return black_image


    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        row_height = self.size[0] / 3

        black_image = np.zeros(shape)


        cv2.rectangle(black_image, (self.position[1], self.position[0]), (self.position[1] + self.size[1] - 1, self.position[0] + row_height - 1), (255), -1)

        cv2.rectangle(black_image, (self.position[1], self.position[0] + row_height), (self.position[1] + self.size[1] - 1, self.position[0] + (row_height * 2) - 1), (126), -1)

        cv2.rectangle(black_image, (self.position[1], self.position[0] + (row_height * 2)), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1), (255), -1)

        return black_image

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        column_width = self.size[1] / 3

        black_image = np.zeros(shape)

        cv2.rectangle(black_image, (self.position[1], self.position[0]), (self.position[1] + column_width - 1, self.position[0] + self.size[0] - 1), (255), -1)

        cv2.rectangle(black_image, (self.position[1] + column_width, self.position[0]), (self.position[1] + (column_width * 2) - 1, self.position[0] + self.size[0] - 1), (126), -1)

        cv2.rectangle(black_image, (self.position[1] + (column_width * 2), self.position[0]), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1), (255), -1)

        return black_image

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        row_height = self.size[0] / 2
        column_width = self.size[1] / 2

        black_image = np.zeros(shape)

        cv2.rectangle(black_image, (self.position[1], self.position[0]), (self.position[1] + column_width - 1, self.position[0] + row_height - 1), (126), -1)

        cv2.rectangle(black_image, (self.position[1] + column_width, self.position[0]), (self.position[1] + self.size[1] - 1, self.position[0] + row_height - 1), (255), -1)

        cv2.rectangle(black_image, (self.position[1], self.position[0] + row_height), (self.position[1] + column_width - 1, self.position[0] + self.size[0] - 1), (255), -1)

        cv2.rectangle(black_image, (self.position[1] + column_width, self.position[0] + row_height), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1), (126), -1)

        return black_image


    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (x, y) and (width, height) format.

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

    def calc_sum(self, ii, top_left, bottom_right):

        s4 = ii[bottom_right[1]][bottom_right[0]]

        s2_pos = (bottom_right[1], top_left[0])
        s2 = ii[s2_pos[0]][s2_pos[1]]

        s3_pos = (top_left[1], bottom_right[0])
        s3 = ii[s3_pos[0]][s3_pos[1]]

        s1 = ii[top_left[1]][top_left[0]]

        whole_sum = s4 - s2 - s3 + s1
        return whole_sum



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

        whole_sum = 0

        if self.feat_type == (2, 1):  # two_horizontal
            row_height = self.size[0] / 2
            sum_of_white = self.calc_sum(ii, (self.position[1] - 1, self.position[0] - 1), (self.position[1] + self.size[1] - 1, self.position[0] + row_height - 1))
            sum_of_grey = self.calc_sum(ii, (self.position[1] - 1, self.position[0] + row_height - 1), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1))

            whole_sum = sum_of_white - sum_of_grey

        elif self.feat_type == (1, 2):  # two_vertical
            column_width = self.size[1] / 2
            sum_of_white = self.calc_sum(ii, (self.position[1] - 1, self.position[0] - 1), (self.position[1] + column_width - 1, self.position[0] + self.size[0] - 1))
            sum_of_grey = self.calc_sum(ii, (self.position[1] + column_width - 1, self.position[0] - 1), (self.position[1] +  self.size[1] - 1, self.position[0] + self.size[0] - 1))

            whole_sum = sum_of_white - sum_of_grey

        elif self.feat_type == (3, 1):  # three_horizontal
            row_height = self.size[0] / 3       
            sum_of_white1 = self.calc_sum(ii, (self.position[1] - 1, self.position[0] - 1), (self.position[1] + self.size[1] - 1, self.position[0] + row_height - 1))
            sum_of_grey = self.calc_sum(ii, (self.position[1] - 1, self.position[0] + row_height - 1), (self.position[1] + self.size[1] - 1, self.position[0] + (row_height * 2) - 1))
            sum_of_white2 = self.calc_sum(ii, (self.position[1] - 1, self.position[0] + (row_height * 2) - 1), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1))

            whole_sum = sum_of_white1 + sum_of_white2 - sum_of_grey

        elif self.feat_type == (1, 3):  # three_vertical
            column_width = self.size[1] / 3
            sum_of_white1 = self.calc_sum(ii, (self.position[1] - 1, self.position[0] - 1), (self.position[1] + column_width - 1, self.position[0] + self.size[0] - 1))
            sum_of_grey = self.calc_sum(ii, (self.position[1] + column_width - 1, self.position[0] - 1), (self.position[1] + (column_width * 2) - 1, self.position[0] + self.size[0] - 1))
            sum_of_white2 = self.calc_sum(ii, (self.position[1] + (column_width * 2) - 1, self.position[0] - 1), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1))

            whole_sum = sum_of_white1 + sum_of_white2 - sum_of_grey

        elif self.feat_type == (2, 2):  # four_square
            row_height = self.size[0] / 2
            column_width = self.size[1] / 2

            sum_of_grey1 = self.calc_sum(ii, (self.position[1] - 1, self.position[0] - 1), (self.position[1] + column_width - 1, self.position[0] + row_height - 1))
            sum_of_white1 = self.calc_sum(ii, (self.position[1] + column_width - 1, self.position[0] - 1), (self.position[1] + self.size[1] - 1, self.position[0] + row_height - 1))
            sum_of_white2 = self.calc_sum(ii, (self.position[1] - 1, self.position[0] + row_height - 1), (self.position[1] + column_width - 1, self.position[0] + self.size[0] - 1))
            sum_of_grey2 = self.calc_sum(ii, (self.position[1] + column_width - 1, self.position[0] + row_height - 1), (self.position[1] + self.size[1] - 1, self.position[0] + self.size[0] - 1))
            
            whole_sum = - sum_of_grey1 + sum_of_white1 + sum_of_white2 - sum_of_grey2

        return whole_sum



def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """

    integral_images = []

    for image in images:
        img = np.cumsum(np.cumsum(image, 0), 1)

        integral_images.append(img)

    return integral_images


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

            hJ = VJ_Classifier(scores, self.labels, weights)
            hJ.train()
            self.classifiers.append(hJ) 


            beta = hJ.error / (1-hJ.error)
            alpha = np.log(1/beta)

            for i in range(0, len(weights)):
                ei = 1
                if(hJ.predict(scores[i]) == self.labels[i]):
                    ei = 0

                weights[i] = weights[i] * (beta**(1-ei))

            
            self.alphas.append(alpha)



            # TODO: Complete the Viola Jones algorithm

            #raise NotImplementedError

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        for clf in self.classifiers:
            feat_id = clf.feature
            hf = self.haarFeatures[feat_id]

            for x, im in enumerate(ii):
                scores[x, feat_id] = hf.evaluate(im)

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).

        threshold = np.array(self.alphas).sum()

        

        for x in scores:
            score_sum = 0

            i = 0
            for clf in self.classifiers:
                score_sum += clf.predict(x) * self.alphas[i]
                i += 1

            if(score_sum  >= threshold * .5):
                result.append(1)
            else:
                result.append(-1)


            #raise NotImplementedError



        return result

    def faceDetection(self, image, filename, original_image):
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

        slices = []

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x_points = []
        y_points = []

        for x in range(0, img.shape[0] - 24):
            for y in range(0, img.shape[1] - 24):
                small_slice = img[x:x+24, y:y+24]
                slices.append(small_slice)
                prediction = self.predict(slices)
                slices = []
                if(prediction[0] == 1):
                    x_points.append(x)
                    y_points.append(y)

        average_point = (int(np.average(x_points)), int(np.average(y_points)))

        resized_x = average_point[0]
        resized_y = average_point[1]

        print((resized_x, resized_y))

        resized_x = int(resized_x * original_image.shape[0] / image.shape[0])
        resized_y = int(resized_y * original_image.shape[1] / image.shape[1])

        print((resized_x, resized_y))

        resized_width = int(24 * original_image.shape[1] / image.shape[1])
        resized_height = int(24 * original_image.shape[0] / image.shape[0])



        cv2.rectangle(original_image, (resized_y, resized_x), (resized_y + 24, resized_x + 24), (255, 0, 0))

        for i in x_points:
            cv2.rectangle(image, (y_points[i], x_points[i]), (y_points[i] + 24, x_points[i] + 24), (255, 0, 0))

        #cv2.rectangle(image, (average_point[1], average_point[0]), (average_point[1] + 24, average_point[0] + 24), (255, 0, 0))
        #showImage(image)
        
        cv2.imwrite("output/{}.png".format(filename), original_image)



