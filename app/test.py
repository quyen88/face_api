
import cv2 as cv
from PIL import Image
from deepface import DeepFace
from deepface.detectors import FaceDetector

query_img_path = './data/npt.jpg'
# detected_faces = DeepFace.detectFace(img_path=query_img_path,detector_backend='opencv')


detector = FaceDetector.build_model('opencv')
faces_1 = FaceDetector.detect_face(detector, 'opencv', query_img_path)
faces_2 = FaceDetector.detect_faces(detector, 'opencv', query_img_path)
print(len(faces_1))
print(len(faces_2))

# image = cv.imread(query_img_path)
# DeepFace.de
#     # Đánh dấu lại các khuôn mặt đã phát hiện bằng hình chữ nhật màu xanh lá cây
# for face in detected_faces:
#     (x, y, w, h) = face['box']
#     cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # Lưu hình ảnh đã được đánh dấu lại
# cv.imwrite('output_path'.jpg, image)

DeepFace.find()