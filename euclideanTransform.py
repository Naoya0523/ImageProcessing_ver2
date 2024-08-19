import cv2
import numpy as np
from translation import Translation
from interpolation import nearestNeighbour, BicubicInterpolation
from tqdm import tqdm
"""
剛体変換

T = [[cosθ, -sinθ, dx],
     [sinθ,  cosθ, dy],
     [0,     0,     1]]

方針は平行移動と同じ．

thetaは回転角度（ラジアン）
dx,dyはそれぞれ移動距離（floot型）

今回も愚直に計算しているのでかなりスピードは遅い．
"""

class EuclideanTransform(Translation):
    def __init__(self):
        super().__init__() # translation.pyからベースとなる変数を継承

    def getT(self, theta:float, dx:int, dy:int) -> np.ndarray:
        return np.array([[np.cos(theta), -np.sin(theta), dx],
                         [np.sin(theta),  np.cos(theta), dy],
                         [0,              0,              1]])
    
    def getInverseT(self, theta:float, dx:int, dy:int) -> np.ndarray:
        return np.linalg.inv(self.getT(theta, dx, dy))
    
    def transformNoInterpolation(self, theta:float, dx: int, dy: int) -> None:
        # 画素補完なし．単純に移動先の画素を求めていく．

        self.T = self.getT(theta, dx, dy) # 行列Tを回転角と移動距離に基づいて取得

        # Tを用いて回転，移動後の座標を取得する
        for before_y in tqdm(range(self.before_translation_image.shape[1])):
            for before_x in range(self.before_translation_image.shape[0]):

                # 計算後の座標は連続的なものとして算出されているので，浮動点少数になっている．
                # しかし離散的に処理しなくてはならないので，整数型に丸め込む．
                after = np.dot(self.T, np.array([[before_x],[before_y],[1]]))
                after_x, after_y = int(after[0][0]), int(after[1][0])

                # 画素値をコピー
                try:
                    self.after_translation_image[after_y][after_x] = self.before_translation_image[before_y][before_x]
                except IndexError:
                    pass
                    # 画像サイズ的に参照不可能なエリアは無視する．

        cv2.imwrite('images/euclideanTransformImages/after_euclideanTransform_no_interpolation.jpg', self.after_translation_image)

    def transformWithInterpolation(self, theta:float, dx:int, dy:int, interpolation) -> np.ndarray:
        # 逆行列を求める
        self.inverse_T = self.getInverseT(theta, dx, dy)

        # 逆行列を元に，変換前の座標を求めていく
        for after_y in tqdm(range(self.after_translation_image.shape[1])):
            for after_x in range(self.after_translation_image.shape[0]):

                # 変換前の座標
                before = np.dot(self.inverse_T, np.array([[after_x], [after_y], [1]]))
                before_x, before_y = before[0][0], before[1][0]
                try:
                    # 画素補完をしながら画素値をコピーしていく
                    pixcel = interpolation(before_x, before_y, self.before_translation_image)
                    self.after_translation_image[after_y][after_x] = pixcel
                except IndexError:
                    # サイズ的に参照不可能なエリアは無視する
                    pass

        return self.after_translation_image