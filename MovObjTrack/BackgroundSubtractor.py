import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

class BackgroundSubtractor:

    def __init__(self, algo, min_area=200):
        if algo == 'MOG2':
            self.cv_bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            # Disable shadow detection
            self.cv_bg_subtractor.setShadowValue(0)
            # Threshold default=16.0
            #self.cv_bg_subtractor.setVarThreshold(36)
        elif algo == 'MOG':
            self.cv_bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        elif algo == 'GMG':
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            self.cv_bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
            #default=120
            self.cv_bg_subtractor.setNumFrames(100)
        else:
            self.cv_bg_subtractor = cv2.createBackgroundSubtractorKNN()
            self.cv_bg_subtractor.setShadowValue(0)
            # Threshold default=400.0
            #self.cv_bg_subtractor.setDist2Threshold(150)

        self.algo = algo
        # Desired width after scaling
        self.desired_width = 640
        # Least number of pixels in a connected region
        self.min_area = min_area 

    def resizeFrame(self, frame):
        width = frame.shape[1]
        height = frame.shape[0]

        scaler = self.desired_width / width
        new_width = int(scaler * width)
        new_height = int(scaler * height)
        resized_frame = cv2.resize(frame, (new_width,new_height), interpolation=cv2.INTER_AREA)

        return resized_frame

    # Process background subtraction.
    # return foreground mask, bounding parameters and centroids
    def bgSub(self, frame):
        ## [apply]
        # Update the background model
        fgMask = self.cv_bg_subtractor.apply(frame)
        if self.algo == 'GMG':
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, self.kernel)
        ## [apply]

        ## [segment_connected_component]
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(fgMask, connectivity=8)
        filteredStats = []
        filteredCentroids = []
        for i in range(1, retval):
            if stats[i][4] >= self.min_area:
                filteredStats.append(stats[i])
                filteredCentroids.append(centroids[i])
        ## [segment_connected_component]

        return fgMask, filteredStats, filteredCentroids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
            OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='videos/vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (MOG, MOG2, GMG, KNN).', default='MOG2')
    parser.add_argument('--min_area', type=int, help='Least pixel number of a connected region to be detected.', default=200)
    args = parser.parse_args()

    ## [create]
    #create a BackgroundSubtractor object
    bgSubtractor = BackgroundSubtractor(args.algo, args.min_area)
    ## [create]

    ## [read_video]
    video = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
    if not video.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)
    ## [read_video]

    #record each running time for each frame
    time_record = []
    time_exceed_count = 0

    #record the whole video playing time
    t_start = time.perf_counter()

    while True:
        ### [timer]
        t0 = time.perf_counter()

        ret, frame = video.read()
        if frame is None:
            break 

        ## [resize_frame]
        frame = bgSubtractor.resizeFrame(frame)
        ## [resize_frame]

        ## [background_subtraction]
        fgMask, stats, centroids = bgSubtractor.bgSub(frame)
        ## [background_subtraction]

        ## [display_bounding_box]
        for stat in stats:
            cv2.rectangle(frame, (stat[0], stat[1]),(stat[0]+stat[2], stat[1]+stat[3]), (0,0,255), 2)
        for centroid in centroids:
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 2, (0,0,255), -1)
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
        print('Time cost: ' + str(t * 1000) + ' millis')
        print('----------------------------')
        ## [console]
          
        time_record.append(t)

        keyboard = cv2.waitKey(30)
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
