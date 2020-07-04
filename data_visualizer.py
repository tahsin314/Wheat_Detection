import numpy as np
from matplotlib import pyplot as plt
from config import *
from WheatDataset import WheatDataset

train_dataset = WheatDataset(df_folds.index.values, markings=marking, dim=1024, transforms=train_aug)
image, target, image_id = train_dataset[109]
boxes = target['boxes'].cpu().numpy().astype(np.int32)
# boxes = target['boxes'].astype(np.int32)
numpy_image = 255*image.permute(1,2,0).cpu().numpy()
# numpy_image = (255.0*np.transpose(image, (1, 0, 2))).astype(int)
# print(boxes)
for box in boxes:
    # print(box)
    cv2.rectangle(numpy_image, (box[1], box[0]), (box[3],  box[2]), (0, 255, 0), 2)
    
cv2.imwrite('Aug.png', numpy_image)
# ax.show()