
# import cv2 as cv
# from PIL import Image
# from deepface import DeepFace
# from deepface.detectors import FaceDetector
import config as config
# query_img_path = './data/npt.jpg'
# # detected_faces = DeepFace.detectFace(img_path=query_img_path,detector_backend='opencv')


# # detector = FaceDetector.build_model('opencv')
# # faces_1 = FaceDetector.detect_face(detector, 'opencv', query_img_path)
# # faces_2 = FaceDetector.detect_faces(detector, 'opencv', query_img_path)
# # print(len(faces_1))
# # print(len(faces_2))



# # DeepFace.find()


# df = DeepFace.stream(source='./query/demo.mp4', 
#                         db_path = config.DB_PATH, 
#                         model_name = "ArcFace", 
#                         distance_metric = "euclidean_l2", 
#                         detector_backend = "opencv",
#                         enable_face_analysis= False,
#                         frame_threshold=15,time_threshold=7
                        
#                         )

# print(df)


import os
# if not os.path.exists("query_video_output"):
#     os.makedirs("query_video_output")
#     os.makedirs(os.path.join("query_video_output", "video_file.filename"))
    
import uuid
# print(uuid.uuid1())

# out_temp = uuid.uuid1()
# query_video_out = os.makedirs(os.path.join(config.DB_OUT,str(out_temp)))

t = [file for file in os.listdir(os.path.join(config.DB_OUT,str('4dc8d206-3372-11ee-b2c0-40ec99184e9d')))]
print(t)