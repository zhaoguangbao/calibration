import argparse
from calibrate import HandInEyeChessCalibrate


parser = argparse.ArgumentParser(description='camera calibration')
parser.add_argument('--num_frame',
                    type=int,
                    default=30,
                    help='number of images used for calibration')
parser.add_argument('--height',
                    type=int,
                    default=8,
                    help='height of the chessboard')
parser.add_argument('--width',
                    type=int,
                    default=11,
                    help='width of the chessboard')
parser.add_argument('--img_height',
                    type=int,
                    default=480,
                    help='pixel height of image')
parser.add_argument('--img_width',
                    type=int,
                    default=640,
                    help='pixel width of image')
parser.add_argument('--size',
                    type=float,
                    default=0.025,
                    help='square size for each square of the chessboard')
parser.add_argument('--dir',
                    type=str,
                    default='/home/zhao/MultiFinGraspExWs/realsense_calibrate/calibration/simulation/output/camera_calibration/',
                    help='directory to load image and save result')
parser.add_argument('--a',
                    type=int,
                    default=15,
                    help='one of the motions for which '
                         'the angle of rotation axes is in [a, pi-a] will be discarded.')
parser.add_argument('--sc',
                    type=float,
                    default=0.0002,
                    help='the scalar threshold for the difference '
                         'between two scalar parts of the quaternions of hand and eye motions.')
args = parser.parse_args()

c = HandInEyeChessCalibrate(args)
c.calibrate()
