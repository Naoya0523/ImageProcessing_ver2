import cv2
import numpy as np

"""
平行移動

T = [[1, 0, dx],
     [0, 1, dy],
     [0, 0,  1]]

変化がわかりやすいように，1500x1500の真っ黒な画像の左上(0,0)に500x500の画像を置き，そこから平行移動させる
今回使用する画像は実家に飾ってあったぬいぐるみたち（SSRB）

dx,dyはそれぞれ移動距離（整数型）
今回は愚直に計算しているのでかなり計算スピードは遅い．
"""

class Translation:
    def __init__(self) -> None:
        self.base_image = np.zeros((1500,1500))                            # shape=(1500,1500,3)
        self.image = cv2.imread('images/resizedImage.jpg', cv2.IMREAD_GRAYSCALE).astype('float32') # shape=(500,500,3)
        self.before_translation_image = self.getBeforeTranslationImage()     # shape=(1500,1500,3)
        self.after_translation_image = np.zeros((1500,1500))               # shape=(1500,1500,3)
        self.T = None                                                        # 平行移動に使う行列T shape=(3,3,)
        self.inverse_T = None                                                # Tの逆行列

    def getBeforeTranslationImage(self) -> np.ndarray:
        background = np.copy(self.base_image)
        for y in range(self.image.shape[1]):
            for x in range(self.image.shape[0]):
                background[y][x] = self.image[y][x]
        #cv2.imwrite('images/before_transform.jpg', background)
        return background
    
    def getT(self, dx:int, dy:int) -> np.ndarray:
        return np.array([[1,0,dx],
                         [0,1,dy],
                         [0,0,1]])
    
    def transform(self, dx:int, dy:int) -> None:
        self.T = self.getT(dx,dy) # 行列Tを移動距離に基づいて取得

        # Tを用いて平行移動後の座標を取得していく
        for before_y in range(self.before_translation_image.shape[1]):
            for before_x in range(self.before_translation_image.shape[0]):

                # 移動後の座標を計算
                after = np.dot(self.T,np.array([[before_x],[before_y],[1]]))
                after_x, after_y = after[0][0], after[1][0]

                # 画素値を移動
                try:
                    self.after_translation_image[after_y][after_x] = self.before_translation_image[before_y][before_x]
                except IndexError:
                    pass
                    # 画像サイズ的に参照不可能なエリアは無視する
        
        # 画像を保存
        cv2.imwrite('images/translationImages/after_translation.jpg', self.after_translation_image)


if __name__ == "__main__":
    dx, dy = 500, 600
    translator = Translation()
    translator.transform(dx,dy)