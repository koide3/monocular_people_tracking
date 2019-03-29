#!/usr/bin/python
import cv2
import math
import numpy

import rospy
import cv_bridge
import message_filters

from sensor_msgs.msg import *


def callback(camera_info, image_msg):
	camera_matrix = numpy.float32(camera_info.K).reshape(3, 3)
	distortion = numpy.float32(camera_info.D).flatten()

	rows = rospy.get_param('~rows', 6)
	cols = rospy.get_param('~cols', 5)
	size = rospy.get_param('~size', 0.109)

	image = cv_bridge.CvBridge().imgmsg_to_cv2(image_msg, 'bgr8')
	undistorted = cv2.undistort(image, camera_matrix, distortion)
	gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

	det, corners_2d = cv2.findChessboardCorners(gray, (rows, cols), cv2.CALIB_CB_FAST_CHECK)
	if det:
		cv2.cornerSubPix(gray, corners_2d, (11, 11), (0, 0), (cv2.TermCriteria_COUNT | cv2.TermCriteria_EPS, 30, 1e-6))
	cv2.drawChessboardCorners(undistorted, (rows, cols), corners_2d, det)

	if not det:
		cv2.imshow('image', undistorted)
		cv2.waitKey(10)

	corners_3d = []
	for i in range(cols):
		for j in range(rows):
			corners_3d.append([i * size, j * size, 0])

	corners_3d = numpy.float32(corners_3d).reshape(-1, 1, 3)

	ret, rvec, tvec = cv2.solvePnP(corners_3d, corners_2d, camera_matrix, None)
	rotation, jacobian = cv2.Rodrigues(rvec)

	cam2base = numpy.eye(4, dtype=numpy.float32)
	cam2base[:3, :3] = rotation
	cam2base[:3, 3] = tvec.flatten()

	base2cam = numpy.linalg.inv(cam2base)
	camera_height = base2cam[2, 3]

	up_vector = cam2base[:3, 2]  # [0, -1, 0]
	up_vector[0] = 0
	up_vector = up_vector / numpy.linalg.norm(up_vector)

	theta = -math.atan2(up_vector[2], -up_vector[1])

	print 'tilt_theta:%.5f height:%.5f' % (theta, camera_height)


def main():
	print '--- calibration_camera_pose_node ---'
	rospy.init_node('calibration_camera_pose_node')
	subs = [
		message_filters.Subscriber('/csi_cam_0/camera_info', CameraInfo),
		message_filters.Subscriber('/csi_cam_0/image_raw', Image)
	]
	sync = message_filters.TimeSynchronizer(subs, 30)
	sync.registerCallback(callback)

	rospy.spin()


if __name__ == '__main__':
	main()
