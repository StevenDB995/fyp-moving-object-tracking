import cv2
import time

#url = 'rtsp://192.168.31.100:8080/h264_pcm.sdp'
url = 'https://192.168.31.100:8080/video'

t_start = time.perf_counter()

video = cv2.VideoCapture(url)
if not video.isOpened:
    print('Unable to open the video from webcam.')
    exit(0)

t_end = time.perf_counter()
#time for fetching the video from server
t_setup = t_end - t_start

#record each running time in each frame
time_record = []
time_exceed_count = 0

#record the whole video playing time
t_start = time.perf_counter()

while True:
    t0 = time.perf_counter()

    ret, frame = video.read()
    if frame is None:
        break
    cv2.imshow('Frame', frame)

    t1 = time.perf_counter()
    #t: frame reading time
    t = t1 - t0

    print('Frame number: ' + str(video.get(cv2.CAP_PROP_POS_FRAMES)))
    print('Time cost: ' + str(t * 1000) + ' millis')
    print('----------------------------')

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

print('Setup time: ' + str(t_setup) + ' seconds\n')

print('Resolution: ' + str(int(width)) + ' x ' + str(int(height)))
print('FPS: ' + str(fps) + '\n')

print('time_allowed_per_frame: ' + str(1000.0 / fps) + ' millis')
print('max_time: ' + str(max_time * 1000) + ' millis')
print('avg_time: ' + str(avg_time * 1000) + ' millis')
print('time_exceed_rate: ' + str(time_exceed_count / len(time_record)))

f = open('test_record/delay_test.txt', 'a+')
f.write('Resolution: ' + str(int(width)) + ' x ' + str(int(height)) + '\n')
f.write('Frame rate: {0:.2f}\n'.format(fps))
f.write('time_allowed_per_frame: {0:.2f} millis\n\n'.format(1000.0 / fps))

if url[:4] == 'rtsp':
    protocol = 'rtsp (h264_pcm)'
else:
    protocol = 'https'
f.write('Protocol: ' + protocol + '\n')
f.write('max_time: {0:.2f} millis\n'.format(max_time * 1000))
f.write('avg_time: {0:.2f} millis\n'.format(avg_time * 1000))
f.write('time_exceed_rate: {0:.4f}\n'.format(time_exceed_count / len(time_record)))

f.write('\n\n\n')
