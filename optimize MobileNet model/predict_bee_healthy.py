import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

# 模型檔案的路徑和名稱
model = load_model('model.h5')

# 待預測圖片的資料夾路徑
images_folder = 'split_imgs'

class_indices = {'1_healthy': 0, '2_fvar': 1, '3_var': 2, '4_ant': 3, '5_robbed': 4, '6_queen': 5}

# 初始化每個類別的計數器
class_counts = {class_name: 0 for class_name in class_indices}

# 遍歷資料夾中的每張圖片，並進行預測
for filename in os.listdir(images_folder):
    img_path = os.path.join(images_folder, filename)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # 進行預測，使用 class_indices
    predictions = model.predict(img_array)
    
    # 取得預測結果對應的類別
    predicted_class = np.argmax(predictions)
    
    # 根據 class_indices 反查類別名稱
    for class_name, class_index in class_indices.items():
        if class_index == predicted_class:
            predicted_label = class_name
            break
    
    # 更新計數器
    class_counts[predicted_label] += 1

# 將每個類別的預測總數畫成長條圖
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel('health category')
plt.ylabel('quantity')
plt.title('Bee health pridection')

# 儲存長條圖
plt.savefig('bar_chart.png')

# 顯示長條圖
plt.show()