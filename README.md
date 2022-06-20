手眼相机标定

# Usage

代码：calibrate_demo.py

配置图像路径及标定板参数

```python
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
                    default='output/camera_calibration/',
                    help='directory to load image and save result')
```

其中dir用于配置图像及位姿路径，图像的命名应按照0~num_frame.png，对应位姿文件pose.txt，其中pose.txt中每行数据满足格式

```bash
pose_process[0:3] -- tran (m), pose_process[3:7] -- quat
```

convert_pose.py文件将指定pose_ori.txt位姿格式转换为需要的pose.txt格式，其中pose_ori.txt格式满足

```bash
pose[0:3] -- tran (mm), pose[3:6] -- xyz(rpy)
```

**注意：height和width等于标定板网格实际数量-1**

# Output

```bash
tcT.txt  # 相机坐标系相对与末端坐标系的变换矩阵
mtx.txt/new_mtx.txt  # 内参矩阵
```

