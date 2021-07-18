from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=10, maxDistance=150):
        """
        Initialize a Centroid Tracker with
        maxDisappered: the maximum of frames how long it can disappear
        maxDistance: max pixel distance between old centroid with ID and current centroid without ID
        """
        self.nextObjectID = 0
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        print("DEREGISTER", objectID)
        del self.objects[objectID]
        del self.disappeared[objectID]

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        print("REGISTER", self.nextObjectID)
        self.objects[self.nextObjectID] = (int(centroid[0]), int(centroid[1]))
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def update(self, rects, change_dis=True):
        """
        Update tracker with BB from the detection
        """
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                if change_dis:
                    self.disappeared[objectID] += 1
                    # if we have reached a maximum number of consecutive
                    # frames where a given object has been marked as
                    # missing, deregister it
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                # return early as there are no centroids or tracking info
            # to update
            return self.objects
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
            # otherwise, are are currently tracking objects so we need to
            # try to match the input centroids to existing object
            # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # D is the distance from objectCentroids to inputCentroids
            cand_ids, _filter_ids = linear_sum_assignment(D, maximize=False)
            # cand_ids is id for input cent in _filter_ids

            # matched IDs and input centroids
            for i, obj_cur_no in enumerate(cand_ids):
                objectID = objectIDs[obj_cur_no]
                if D[cand_ids[i], _filter_ids[i]] < self.maxDistance:
                    # update id with new centriod
                    self.objects[objectID] = inputCentroids[_filter_ids[i]]
                    self.disappeared[objectID] = 0
                else:
                    # too far away, create new id for input centroid
                    self.register(inputCentroids[_filter_ids[i]])
                    if change_dis:
                        # for id add to disapperance
                        self.disappeared[objectID] += 1
                        if self.disappeared[objectID] > self.maxDisappeared:
                            self.deregister(objectID)
            # not matched IDs
            for i, objectID in enumerate(objectIDs):
                if i not in cand_ids and change_dis:
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # not matched input centroids
            for i in range(0, len(inputCentroids)):
                if i not in _filter_ids:
                    self.register(inputCentroids[i])
        return self.objects

    def new_id_registered(self, rects):
        if len(rects) == 0:
            return False
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            return True
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # D is the distance from objectCentroids to inputCentroids
            cand_ids, _filter_ids = linear_sum_assignment(D, maximize=False)
            # cand_ids is id for input cent in _filter_ids

            # matched IDs and input centroids
            for i, obj_cur_no in enumerate(cand_ids):
                objectID = objectIDs[obj_cur_no]
                if D[cand_ids[i], _filter_ids[i]] > self.maxDistance:
                    # too far away, create new id for input centroid
                    return True
            # not matched input centroids
            for i in range(0, len(inputCentroids)):
                if i not in _filter_ids:
                    return True
        return False



