import cv2
import numpy as np

def boundary_detection(input, output_video, output_image):
    video_capture = cv2.VideoCapture(input)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # print(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_count += 1

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_scale, 15, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        egde_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(egde_frame, [largest_contour], -1, (0, 0, 255), 1)

            # if frame_count == 304:                         # Optional
            #     cv2.imwrite("original_frame.png", frame)
            #     cv2.imwrite("edge_frame.png", egde_frame)

            if frame_count % 2 == 0:
                cv2.drawContours(canvas, [largest_contour], -1, (0, 0, 0), 1)

        output_video.write(egde_frame)
        cv2.imwrite(output_image, canvas)



    video_capture.release()
    output_video.release()





if __name__ == '__main__':
    input_video = "raw_video_waterflow.mp4"
    output_video = "waterflow_contour.mp4"
    output_image = "waterflow_accumulation.png"

    boundary_detection(input_video, output_video, output_image)
