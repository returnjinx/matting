import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
import paddlehub as hub
from PIL import Image, ImageSequence
import numpy as np
import os

# 测试图片路径和输出路径
test_path = ''  #按需更改路径
output_path = ''  #按需更改路径 

# 待预测图片
test_img_path = ["2.png"]  #按需更改文件名
test_img_path = [test_path + img for img in test_img_path]

out_img_path = 'humanseg_output'+os.sep + os.path.basename(test_img_path[0]).split('.')[0] + '.png'
out_path = 'humanseg_output'
module = hub.Module(name="deeplabv3p_xception65_humanseg")

input_dict = {"image": test_img_path}
# execute predict and print the result
results = module.segmentation(data=input_dict,output_dir=out_path,visualization=True)
print(results)
for result in results:
    print(result)




# 预测结果展示
# out_img_path = 'zhou.png'   #输出图片的位置
print(out_img_path)

img = mpimg.imread(out_img_path)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()
