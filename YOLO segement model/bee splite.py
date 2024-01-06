from ultralytics import YOLO
import cv2
import copy
import pandas as pd
import os

model = YOLO('best.pt')
predictions = model.predict(source='orign_imgs', save=True, save_txt=True)

# 獲取偵測框的座標值
def detection_frame_coordinates(label_file_path):
  # 紀錄偵測框座標的表
  coordinates_table = pd.DataFrame(columns=['x_center', 'y_center', 'width', 'height'])

  with open(label_file_path, 'r') as file:
    # 將每一行讀取到列表中
    lines = file.readlines()

    # 顯示列表中的內容
    for line in lines:
      line = line.split(" ")
      coordinates_table.loc[len(coordinates_table.index)] = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
  return coordinates_table


# 圖片資料夾的路徑
folder_path = "orign_imgs"

# 資料夾中所有圖片的名字
image_files = os.listdir(folder_path)

for image_file in image_files:
    # 構建完整的文件路徑
    image_path = os.path.join(folder_path, image_file)

    # 使用OpenCV讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    try:
        # 獲取偵測框的座標值
        coordinates_table = detection_frame_coordinates(f'runs/detect/predict/labels/{image_file[:-3]}txt')
        for i in range(coordinates_table.shape[0]):
            x1 = int((coordinates_table.iloc[i,0]-coordinates_table.iloc[i,2]/2)*image.shape[1])
            x2 = int((coordinates_table.iloc[i,0]+coordinates_table.iloc[i,2]/2)*image.shape[1])
            y1 = int((coordinates_table.iloc[i,1]-coordinates_table.iloc[i,3]/2)*image.shape[0])
            y2 = int((coordinates_table.iloc[i,1]+coordinates_table.iloc[i,3]/2)*image.shape[0])
            image_split = copy.deepcopy(image)
            image_split = image_split[y1:y2,x1:x2]
            # 保存圖像
            cv2.imwrite('split_imgs/%s'%image_file[:-4]+'_'+str(i)+image_file[-4:], image_split)

    except FileNotFoundError:
        continue