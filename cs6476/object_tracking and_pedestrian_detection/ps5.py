"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2
import os
output_dir = "output"

# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([[init_x], [init_y], [0.], [0.]])  # state
        self.d = np.asmatrix(np.array([[1, 0, 1, 0],[0, 1, 0, 1],[0, 0, 1, 0],[0, 0, 0, 1]])) # state vector transition 
        self.m = np.asmatrix(np.array([[1, 0, 0, 0],[0, 1, 0, 0]])) # measurement 
        self.covU = np.asmatrix(1000*np.eye(4)) # a convariance matrix
        self.covQ = np.asmatrix(Q) # process noise
        self.covR = np.asmatrix(R) # measurement noise
        self.identity = np.asmatrix(np.eye(4))

    def predict(self):
        # predict state
        self.state = self.d * self.state
        # uncertainty matrix
        self.covU = self.d * self.covU * self.d.T + self.covQ

    def correct(self, meas_x, meas_y):
        # measurement
        measurement = np.asmatrix(np.array([[meas_x, meas_y]]))
        current = self.m * self.state
        error = measurement - current

        # compute K
        s = self.m * self.covU * self.m.T + self.covR # residual convariance
        k = self.covU * self.m.T * s.I

        # next state
        self.state = self.state + k * error
        self.covU = (self.identity - k * self.m) * self.covU

    def process(self, measurement_x, measurement_y):
        self.predict()
        self.correct(measurement_x, measurement_y)
        return self.state.A1[0], self.state.A1[1]

class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = np.average(template,axis=2).astype('uint8')
        self.frame = frame

        self.particles = np.zeros((self.num_particles,2)) # (x,y) locations
        self.weights = np.ones((self.num_particles,1))*(1.0/self.num_particles) # same weight
        self.frameCount = 0
        self.particles[:,0] = self.template_rect['x'] + (np.random.rand(self.num_particles)*2*self.template.shape[0]).astype('int')
        self.particles[:,1] = self.template_rect['y'] + (np.random.rand(self.num_particles)*2*self.template.shape[1]).astype('int')

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        rows,cols = template.shape[0],template.shape[1]
        squareError = np.sum((template.astype("float") - frame_cutout.astype("float"))**2)
        meanSquareErorr = squareError/float(rows*cols)

        return meanSquareErorr

    def std_dev(self,avg):
        diff = np.zeros((self.num_particles,1))
        for i in range(self.num_particles):
            diff[i,0] = np.sqrt((self.particles[i,0]-avg[0])**2 + (self.particles[i,1]-avg[1])**2)
        weighted_diff = np.sum(diff*self.weights)/float(np.sum(self.weights))
        return weighted_diff

    def get_frame_cutout(self, u, v, template,frame):
        basicAlgo = True
        if basicAlgo:
            # if the window is smaller than the template, then just fill it with zero values 
            rows = template.shape[0]
            cols = template.shape[1]
        
            upper = u-rows/2
            lower = u+rows/2
            left = v-cols/2
            right = v+cols/2

            # edge cases
            if upper < 0:
                upper = 0
            elif upper >= frame.shape[0]:
                upper = frame.shape[0]-1
        
            if lower > frame.shape[0]:
                lower = frame.shape[0]-1
            elif lower < 0:
                lower = 0
        
            if left < 0:
                left = 0
            elif left > frame.shape[1]:
                left = frame.shape[1]-1
        
            if right > frame.shape[1]:
                right = frame.shape[1]-1
            elif right < 0:
                right = 0
            upper = int(upper)
            lower = int(lower)
            left = int(left)
            right = int(right)
            cutoutRows = lower - upper
            cutoutCols = right - left
        
            #if cutoutCols != rows or cutoutRows != cols:
            #    print lower-upper, right-left
            # garantee the size
            frame_cutout = np.zeros(template.shape)
            frame_cutout[0:cutoutRows, 0:cutoutCols] = frame[upper:lower, left:right]
        else:
            # create a new template that matches the window size
            print "TODO"
        return frame_cutout

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        newParticles = np.zeros((self.num_particles,2))

        # the number of particles to each location based on the weight
        particleDistribution = np.random.multinomial(self.num_particles, self.weights.reshape(self.num_particles).tolist(), size=1)
        particleCount = 0
        # more particles will be updated to locations with higher weight
        for i in range(self.num_particles):
            newParticles[particleCount:particleDistribution[0][i]+particleCount] = self.particles[i]
            particleCount += particleDistribution[0][i]
        return newParticles

    def process(self, frame):
        frame = np.average(frame,axis=2).astype('uint8')  
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        # 1. resample particles
        newParticles = self.resample_particles()

        # 2. update the particles with gaussian noise
        newParticles[:, 0] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        newParticles[:, 1] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        self.particles = newParticles

        # 3. update the weight
        '''
        Explaination:
        1. The frame_cout is the portion of the image frame to be compared with template.
        2. Each particle is the center of the frame_cutout
        3. The frame_cutout will be the same size as the template. When the particle is at the edge, the template may cut as well
        '''
        newWeight = np.zeros((self.num_particles, 1)) 

        for i in range(self.num_particles):
            v,u = self.particles[i][0],self.particles[i][1]
            frame_cutout = self.get_frame_cutout(u,v,self.template,frame)
            meanSquareError = self.get_error_metric(self.template,frame_cutout)
            newWeight[i] = np.exp(-1*meanSquareError/float(2*self.sigma_exp**2))
        # normalize
        self.weights = newWeight/np.sum(newWeight)

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        # compute weighted mean
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
        mean = [x_weighted_mean,y_weighted_mean]

        # compute spread distribution
        diff = np.zeros((self.num_particles, 1))
        for i in range(self.num_particles):
            diff[i,0] = np.sqrt((self.particles[i,0]-mean[0])**2 + (self.particles[i,1]-mean[1])**2)
        
        spread = np.sum(diff*self.weights)/float(np.sum(self.weights))

        # draw particles
        for i in range(self.num_particles):
            pt1 = (int(self.particles[i,0]), int(self.particles[i,1]))
            cv2.circle(frame_in, pt1, 1, (0,255,0), thickness=1)
        self.width, self.height = self.template.shape[1],self.template.shape[0]
        # draw rectangle and circle with spread distribution
        pt1 = (int(mean[0]-self.width/2), int(mean[1]-self.height/2))
        pt2 = (int(mean[0]+self.width/2), int(mean[1]+self.height/2))
        cv2.rectangle(frame_in, pt1, pt2, (0,255,0), thickness=1)
        cv2.circle(frame_in, (mean[0], mean[1]), spread.astype('int'), (0,0,255), thickness=2)

        # draw template location on first image
        #pt1 = (int(self.template_rect['x']),int(self.template_rect['y']))
        #pt2 = (int(self.template_rect['x']+self.template_rect['w']),int(self.template_rect['y']+self.template_rect['h']))
        #cv2.rectangle(frame_in, pt1, pt2, (0,0,255), thickness=1)
        
class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        frame = np.average(frame,axis=2).astype('uint8')
        # 1. resample particles
        newParticles = self.resample_particles()

        # 2. update the particles with gaussian noise
        newParticles[:, 0] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        newParticles[:, 1] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        self.particles = newParticles

        # 3. update the weight
        '''
        Explaination:
        1. The frame_cout is the portion of the image frame to be compared with template.
        2. Each particle is the center of the frame_cutout
        3. The frame_cutout will be the same size as the template. When the particle is at the edge, the template may cut as well
        '''
        newWeight = np.zeros((self.num_particles, 1)) 

        for i in range(self.num_particles):
            v,u = self.particles[i][0],self.particles[i][1]
            frame_cutout = self.get_frame_cutout(u,v,self.template,frame)
            meanSquareError = self.get_error_metric(self.template,frame_cutout)
            newWeight[i] = np.exp(-1*meanSquareError/float(2*self.sigma_exp**2))
        # normalize
        self.weights = newWeight/np.sum(newWeight)

        # 4. update template
        self.update_template(frame)
        
        #self.frameCount += 1
        #template_name = 'template' + str(self.count) + '.png'
        #cv2.imwrite("./{}".format(template_name), self.template )

    def update_template(self, frame):
        # compute the best match window
        highestWeight = self.weights[0]
        highestWeightIdx = 0
        for i in range(self.num_particles):
            if self.weights[i] > highestWeight:
                highestWeight = self.weights[i]
                highestWeightIdx = i

        u,v = self.particles[highestWeightIdx,0], self.particles[highestWeightIdx,1]
        bestWindow = self.get_frame_cutout(u,v,self.template,frame)
        
        # The new template is a weighted sum of the two templates, the old one and the new best match
        self.template = (self.alpha*bestWindow + (1-self.alpha)*self.template).astype('uint8')  
        self.template = cv2.normalize(self.template,None,0,255,cv2.NORM_MINMAX)

class MDParticleFilter(AppearanceModelPF):

    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.alpha = kwargs.get('alpha',0.5)
        self.sigma_scale = kwargs.get('sigma_scale',0.1)
        #self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.template = np.average(template,axis=2).astype('uint8')
        #self.template = template
        self.frame = frame
        self.templateRatio = template.shape[0]/template.shape[1]
        self.particles = np.ones((self.num_particles,3)) # (x,y) locations
        self.weights = np.zeros((self.num_particles,1))*(1.0/self.num_particles) # same weight
        self.frameCount = 0
        self.particles[:,0] = self.template_rect['x'] + (np.random.rand(self.num_particles)*self.template_rect['w']).astype('int')
        self.particles[:,1] = self.template_rect['y'] + (np.random.rand(self.num_particles)*self.template_rect['h']).astype('int')
        self.particles[:,2] += np.random.normal(0, self.sigma_scale, self.num_particles)
        self.templateHeight = self.template.shape[0]
        self.templateWidth = self.template.shape[1]

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        newParticles = np.zeros((self.num_particles,3))

        # the number of particles to each location based on the weight
        particleDistribution = np.random.multinomial(self.num_particles, self.weights.reshape(self.num_particles).tolist(), size=1)
        particleCount = 0
        # more particles will be updated to locations with higher weight
        for i in range(self.num_particles):
            newParticles[particleCount:particleDistribution[0][i]+particleCount] = self.particles[i]
            particleCount += particleDistribution[0][i]
        return newParticles

    # sensing model
    def updateWeightParticles(self,frame):
        newWeight = np.zeros((self.num_particles, 1)) 

        for i in range(self.num_particles):
            v,u = self.particles[i][0],self.particles[i][1]
            fz = self.particles[i][2]
            fy = int(self.template.shape[1]*fz)

            fx = int(fy*self.templateRatio)
            template = cv2.resize(self.template,(fy,fx))           
            frame_cutout = self.get_frame_cutout(u,v,template,frame)

            meanSquareError = self.get_error_metric(template,frame_cutout)
            newWeight[i] = np.exp(-1*meanSquareError/float(2*self.sigma_exp**2))

        # normalize
        self.weights = newWeight/np.sum(newWeight)

    def update_template(self, frame):
        # some metric to determine the update 
        template = np.copy(self.template)
        resize_template = np.copy(self.template) 
        highestWeight = self.weights[0]
        highestWeightIdx = 0 
        for i in range(self.num_particles):
            if self.weights[i] > highestWeight:
                highestWeight = self.weights[i]
                highestWeightIdx = i

        x_weighted_mean = 0
        y_weighted_mean = 0
        z_weighted_mean = 0
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            z_weighted_mean += self.particles[i, 2] * self.weights[i]
        mean = [x_weighted_mean,y_weighted_mean, z_weighted_mean]
        fz = z_weighted_mean
        
        fy = int(template.shape[1] * fz)
        fx = int(fy*self.templateRatio)

        if abs(fy - template.shape[1]) > 1:  
            template = cv2.resize(template,(fy,fx))
            resize_template = cv2.resize(self.template,(fy,fx)) 
        u,v = self.particles[highestWeightIdx,0], self.particles[highestWeightIdx,1]
        
        # The new template is a weighted sum of the two templates, the old one and the new best match
        bestWindow = self.get_frame_cutout(u,v,template,frame )

        template = (self.alpha*bestWindow + (1-self.alpha)*template).astype('uint8')  
        template = cv2.normalize(template,None,0,255,cv2.NORM_MINMAX) 

        std = self.std_dev([x_weighted_mean,y_weighted_mean])
        meanSquareError = self.get_error_metric(resize_template,template)
        
        #if meanSquareError < 0.5 and std < 8:
        #    self.template = template
        if meanSquareError < 2.1:
            self.template = template
        #print "Scale:",fz,fx,fy,"window:", self.template.shape, "std",std, "mse", meanSquareError 


    def update_model(self, newParticles):
        newParticles[:, 0] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        newParticles[:, 1] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        newParticles[:, 2] += np.random.normal(0, self.sigma_scale, self.num_particles)
        self.particles = newParticles       

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        frame = np.average(frame,axis=2).astype('uint8')
        # 1. resample particles
        newParticles = self.resample_particles()
        # 2. update the particles with gaussian noise (motion model)
        self.update_model(newParticles)
        # 3. update the weight (sensing model)
        self.updateWeightParticles(frame)        

        # 4. update template
        # compute the best match window
        self.update_template(frame)
        
        self.frameCount += 1
        #template_name = 'template' + str(self.frameCount) + '.png'
        #cv2.imwrite("./{}".format(template_name), self.template )

class MTParticleFilter(AppearanceModelPF):

    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MTParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.alpha = kwargs.get('alpha',0.5)
        self.sigma_scale = kwargs.get('sigma_scale',0.1)
        #self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.template = np.average(template,axis=2).astype('uint8')
        #self.template = template
        self.frame = frame
        self.templateRatio = template.shape[0]/template.shape[1]
        self.particles = np.ones((self.num_particles,3)) # (x,y) locations
        self.weights = np.zeros((self.num_particles,1))*(1.0/self.num_particles) # same weight
        self.frameCount = 0
        self.particles[:,0] = self.template_rect['x'] + (np.random.rand(self.num_particles)*self.template.shape[0]).astype('int')
        self.particles[:,1] = self.template_rect['y'] + (np.random.rand(self.num_particles)*self.template.shape[1]).astype('int')
        self.particles[:,2] += np.random.normal(0, self.sigma_scale, self.num_particles)
        self.templateHeight = self.template.shape[0]
        self.templateWidth = self.template.shape[1]
        print self.templateHeight,self.templateWidth, self.frame.shape
    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        newParticles = np.zeros((self.num_particles,3))

        # the number of particles to each location based on the weight
        particleDistribution = np.random.multinomial(self.num_particles, self.weights.reshape(self.num_particles).tolist(), size=1)
        particleCount = 0
        # more particles will be updated to locations with higher weight
        for i in range(self.num_particles):
            newParticles[particleCount:particleDistribution[0][i]+particleCount] = self.particles[i]
            particleCount += particleDistribution[0][i]
        return newParticles

    # sensing model
    def updateWeightParticles(self,frame):
        template = self.template
        newWeight = np.zeros((self.num_particles, 1)) 

        for i in range(self.num_particles):
            v,u = self.particles[i][0],self.particles[i][1]
            frame_cutout = self.get_frame_cutout(u,v,template,frame)
            meanSquareError = self.get_error_metric(template,frame_cutout)
            newWeight[i] = np.exp(-1*meanSquareError/float(2*self.sigma_exp**2))

        # normalize
        self.weights = newWeight/np.sum(newWeight)

    def update_template(self, frame):
        # some metric to determine the update 
        template = np.copy(self.template)
        resize_template = np.copy(self.template) 
        highestWeight = self.weights[0]
        highestWeightIdx = 0 
        for i in range(self.num_particles):
            if self.weights[i] > highestWeight:
                highestWeight = self.weights[i]
                highestWeightIdx = i

        x_weighted_mean = 0
        y_weighted_mean = 0
        z_weighted_mean = 0
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            z_weighted_mean += self.particles[i, 2] * self.weights[i]
        mean = [x_weighted_mean,y_weighted_mean, z_weighted_mean]
        fz = z_weighted_mean
        
        fy = int(template.shape[1] * fz)
        fx = int(fy*self.templateRatio)

        if abs(fy - template.shape[1]) > 1:  
            template = cv2.resize(template,(fy,fx))
            resize_template = cv2.resize(self.template,(fy,fx)) 
        u,v = self.particles[highestWeightIdx,0], self.particles[highestWeightIdx,1]
        
        # The new template is a weighted sum of the two templates, the old one and the new best match
        bestWindow = self.get_frame_cutout(u,v,template,frame )

        template = (self.alpha*bestWindow + (1-self.alpha)*template).astype('uint8')  
        template = cv2.normalize(template,None,0,255,cv2.NORM_MINMAX) 

        std = self.std_dev([x_weighted_mean,y_weighted_mean])
        meanSquareError = self.get_error_metric(resize_template,template)
        
        #if meanSquareError < 0.5 and std < 8:
        #    self.template = template
        if meanSquareError < 2.1:
            self.template = template
        #print "Scale:",fz,fx,fy,"window:", self.template.shape, "std",std, "mse", meanSquareError 


    def update_model(self, newParticles):
        newParticles[:, 0] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        newParticles[:, 1] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        newParticles[:, 2] += np.random.normal(0, self.sigma_scale, self.num_particles)
        self.particles = newParticles       

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        frame = np.average(frame,axis=2).astype('uint8')
        # 1. resample particles
        newParticles = self.resample_particles()
        # 2. update the particles with gaussian noise (motion model)
        self.update_model(newParticles)
        # 3. update the weight (sensing model)
        self.updateWeightParticles(frame)        

        # 4. update template
        # compute the best match window
        
        #self.update_template(frame)
        
        #self.frameCount += 1
        #template_name = 'template' + str(self.frameCount) + '.png'
        #cv2.imwrite("./{}".format(template_name), self.template )
