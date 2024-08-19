import cv2
import numpy as np
from euclideanTransform import EuclideanTransform
from interpolation import nearestNeighbour, BicubicInterpolation

if __name__ == "__main__":
    """
    複数回
    """
    euclidean_transformer = EuclideanTransform() # 剛体変換用のオブジェクトを作成
    #b = BicubicInterpolation()
    inter = nearestNeighbour # 補間法を選択．(双三次補間なら，inter = BicubicInterpolation().bicubicInterpolatuon)
    euclidean_transformer.before_translation_image = cv2.imread('images/rotate.jpg', flags=cv2.IMREAD_GRAYSCALE) # 変換用の画像を読み込み

    theta = np.pi/(10*2) # 回転角

    theta2 = (np.pi-theta)/2 - np.pi/4
    dx = np.sqrt((1500**2)*(1-np.cos(theta)))*np.cos(theta2) + 0 # 移動距離．今回は，回転の中心が画像の中央になるように調整している
    dy = -np.sqrt((1500**2)*(1-np.cos(theta)))*np.sin(theta2) + 0

    # 10回繰り返し
    for i in range(10):
        image = euclidean_transformer.transformWithInterpolation(theta, dx, dy, inter)
        euclidean_transformer.before_translation_image = np.copy(image)
        euclidean_transformer.after_translation_image = np.zeros((1500,1500))
    cv2.imwrite('nearest.jpg', image)
