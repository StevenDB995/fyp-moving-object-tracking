import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from BackgroundSubtractor import *
from CentroidTracker import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
            OpenCV. You can process both videos and images.')
    parser.add_argument('--algo', type=str, help='Background subtraction method (MOG, MOG2, GMG, KNN).', default='MOG2')
    parser.add_argument('--min_area', type=int, help='Least pixel number of a connected region to be detected.', default=200)
    parser.add_argument('--enable_ct', type=int, help='Enable centroid tracker or not. 0 for disable and 1 for enable.', default=0)
    args = parser.parse_args()

    bgSubtractor = BackgroundSubtractor(args.algo, args.min_area)
    if args.enable_ct:
        ct = CentroidTracker()
    else:
        ct = None

    ## [read_video_from_webcam]
    #url = 'rtsp://192.168.31.100:8080/h264_pcm.sdp'
    url = 'https://192.168.31.100:8080/video'
    video = cv2.VideoCapture(url)
    if not video.isOpened:
        print('Unable to open the video from webcam.')
        exit(0)
    ## [read_video_from_webcam]

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

        ## [update_tracking_status]
        if args.enable_ct:
            ct.update(centroids)
        ## [update_tracking_status]

        ## [display_bounding_box]
        for stat in stats:
            cv2.rectangle(frame, (stat[0], stat[1]),(stat[0]+stat[2], stat[1]+stat[3]), (0,0,255), 2)
        if args.enable_ct:
            for (objectID, centroid) in ct.object2Centroid.items():
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 2, (0,0,255), -1)
                cv2.putText(frame, 'ID: ' + str(objectID), (int(centroid[0]) - 10, int(centroid[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        else:
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

        keyboard = cv2.waitKey(1)
        if keyboard == 113 or keyboard == 27:
            break

    t_end = time.perf_counter()

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = len(time_record) / (t_end - t_start)

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

    plt.hist(np.array(time_record) * 1000, bins=100)
    plt.gca().set(title='Processing time distribution',
            xlabel='Processing time per frame (millis)',
            ylabel='Frequency')
    plt.axvline(x=avg_time*1000, color='lime', linestyle='solid', linewidth=1, label='Mean')
    plt.axvline(x=1000.0/fps, color='r', linestyle='solid', linewidth=1, label='Maximum allowed')
    plt.legend(prop={'size': 10})
    plt.show()
