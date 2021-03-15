import argparse
from BackgroundSubtractor import *
import cv2
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import time

class CentroidTracker:
    ## [parameters_for_tracker]
    # The next unique ID to be assigned to a new object
    nextObjectID = 0
    # Dictionary - Key: registered object ID; Value: centroid of the object in the current frame
    object2Centroid = {}
    # Dictionary - Key: registered object ID; Value: number of consecutive frame the object has been lost
    object2Lost = {}
    
    # Maximum number of consecutive frames allowed for a registered object to be lost for
    max_lost = 50
    ## [parameters_for_tracker]

    ## [functions]
    # Register an object to be tracked
    def register(self, centroid):
        self.object2Centroid[self.nextObjectID] = centroid
        self.object2Lost[self.nextObjectID] = 0
        self.nextObjectID += 1
    
    # Deregister an object being tracked
    def deregister(self, objectID):
        del self.object2Centroid[objectID]
        del self.object2Lost[objectID]
    
    # Update the tracking status of each object in the current frame
    # Parameter - centroids: centroids of objects detected in the current frame
    def update(self, centroids):
        # If no object is detected in the current frame,
        # mark the objects being tracked as lost
        if len(centroids) == 0:
            for objectID in list(self.object2Lost.keys()):
                self.object2Lost[objectID] += 1
                if self.object2Lost[objectID] > self.max_lost:
                    self.deregister(objectID)
    
        # If no object is being tracked,
        # register the objects in the current frame as new
        elif len(self.object2Centroid) == 0:
            for centroid in centroids:
                self.register(centroid)
    
        # If there are objects being tracked,
        # try to match the newly detected objects with them
        else:
            objectIDs = list(self.object2Centroid.keys())
            objectCentroids = list(self.object2Centroid.values())
    
            # Calculate the distance between each pair of centroids:
            # centroids of objects being tracked and the newly detected objects
            D = dist.cdist(np.array(objectCentroids), np.array(centroids))
            # Find the minimum distance in each row
            # and sort the row indexes and column indexes
            # in ascending order of the distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
    
            # Record the examined rows and columns
            usedRows = set()
            usedCols = set()
    
            # Pair up the row indexs and column indexes
            # and examine each of the pairs
            for (row, col) in zip(rows, cols):
                # Ignore the examined pair
                if row in usedRows or col in usedCols:
                    continue
    
                # Update the centroid of the object being tracked
                # and reset its counter for lost frames
                objectID = objectIDs[row]
                self.object2Centroid[objectID] = centroids[col]
                self.object2Lost[objectID] = 0
    
                # Mark the row index and column index as examined
                usedRows.add(row)
                usedCols.add(col)
    
            # Compute the row indexes and column indexes not yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
    
            # If the number of objects being tracked (number of rows) is equal to or greater than
            # the number of newly detected objects in the current frame (number of columns),
            # check and find the potentially lost objects
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.object2Lost[objectID] += 1
    
                    if self.object2Lost[objectID] > self.max_lost:
                        self.deregister(objectID)
            # Otherwise, check and register the new objects
            else:
                for col in unusedCols:
                    self.register(centroids[col])
    ## [functions]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program shows how to track moving objects in an image sequence \
            with centroid tracking algorithm.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='videos/vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (MOG, MOG2, GMG, KNN).', default='MOG2')
    parser.add_argument('--min_area', type=int, help='Least pixel number of a connected region to be detected.', default=200)
    args = parser.parse_args()

    ## [prepare]
    bgSubtractor = BackgroundSubtractor(args.algo)
    ct = CentroidTracker()

    video = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
    if not video.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)
    ## [prepare]

    #record each running time for each frame
    time_record = []
    time_exceed_count = 0

    #record the whole video processing time
    t_start = time.perf_counter()

    while True:
        ### [timer]
        t0 = time.perf_counter()

        ret, frame = video.read()
        if frame is None:
            break

        ## [resize_frame]
        #frame = bgSubtractor.resizeFrame(frame)
        ## [resize_frame]

        ## [background_subtraction]
        fgMask, stats, centroids = bgSubtractor.bgSub(frame)
        ## [background_subtraction]

        ## [update_tracking_status]
        ct.update(centroids)
        ## [update_tracking_status]

        ## [display_bounding_box]
        for stat in stats:
            cv2.rectangle(frame, (stat[0], stat[1]),(stat[0]+stat[2], stat[1]+stat[3]), (0,0,255), 2)
        for (objectID, centroid) in ct.object2Centroid.items():
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 2, (0,0,255), -1)
            cv2.putText(frame, 'ID: ' + str(objectID), (int(centroid[0]) - 10, int(centroid[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        ## [display_bounding_box]

        t1 = time.perf_counter()
        t = t1 - t0
        ### [timer]

        ## [display_frame_number]
        cv2.rectangle(frame, (10, 2), (80, 20), (255,255,255), -1)
        cv2.putText(frame, str(video.get(cv2.CAP_PROP_POS_FRAMES)),
                (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        ## [display_frame_number]

        ## [show]
        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)
        ## [show]

        ## [console]
        print('Frame number: ' + str(video.get(cv2.CAP_PROP_POS_FRAMES)))
        print('Blobs count: ' + str(len(stats)))
        print('Time cost: ' + str((t1 - t0) * 1000) + ' millis')
        print('----------------------------')
        ## [console]

        time_record.append(t)
          
        keyboard = cv2.waitKey(0)
        if keyboard == 113 or keyboard == 27:
            break

    t_end = time.perf_counter()

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)

    max_time = max(time_record)
    avg_time = sum(time_record) / len(time_record)
    for time in time_record:
        if (time * 1000) > (1000.0 / fps):
            time_exceed_count += 1

    print('Resolution: ' + str(int(width)) + ' x ' + str(int(height)))
    print('FPS: ' + str(fps) + '\n')

    print('time_allowed_per_frame: ' + str(1000.0 / fps) + ' millis')
    print('max_time: ' + str(max_time * 1000) + ' millis')
    print('avg_time: ' + str(avg_time * 1000) + ' millis')
    print('time_exceed_rate: ' + str(time_exceed_count / len(time_record)))

    plt.hist(np.array(time_record) * 1000, bins=50)
    plt.gca().set(title='Processing time distribution',
            xlabel='Processing time per frame (millis)',
            ylabel='Frequency')
    plt.axvline(x=avg_time*1000, color='lime', linestyle='solid', linewidth=1, label='Mean')
    plt.axvline(x=1000.0/fps, color='r', linestyle='solid', linewidth=1, label='Maximum allowed')
    plt.legend(prop={'size': 10})
    plt.show()
