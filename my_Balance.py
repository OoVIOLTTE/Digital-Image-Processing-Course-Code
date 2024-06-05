# 源代码:https://blog.csdn.net/xiexinzhe12345/article/details/134408948

import numpy as np
import cv2

#封装成类
class Color_Balance_and_Fusion_for_Underwater_Image_Enhancement():
    def __init__(self):
        pass

    #读图片
    def read_img_by_cv2(self,img_path):
        img=cv2.imread(img_path)
        return img
    
    #色彩补偿
    def color_balance(self,img,alpha=0.3):
        R = img[:, :, 2]
        G = img[:, :, 1]
        B = img[:, :, 0]
        # 三颜色通道均值再归一化，对应 I¯r I¯g I¯b
        Irm = np.mean(R)/256.0
        Igm = np.mean(G)/256.0
        Ibm = np.mean(B)/256.0
    
        Irc = R + (alpha) * (Igm-Irm)*(1-Irm)*G  # 补偿红色通道
        Irc = np.array(Irc.reshape(G.shape), np.uint8)
        Ibc = B + (alpha/2) * (Igm-Ibm)*(1-Ibm)*G  # 补偿蓝色通道
        Ibc = np.array(Ibc.reshape(G.shape), np.uint8)

        img = cv2.merge([Ibc,G,Irc])

        img=np.clip(img,0,255)
        img=img.astype(np.uint8)
        return img
    
    #白平衡
    def gray_world(self,img,alpha=1,beta=1.1,gamma=0.9):
    # 将图像转换为浮点格式
        img_float = img.astype(float)
        # 计算图像的各通道平均值
        avg_b = np.mean(img_float[:, :, 0])
        avg_g = np.mean(img_float[:, :, 1])
        avg_r = np.mean(img_float[:, :, 2])

        # 计算各通道的增益
        gain_b = (avg_g / avg_b)*alpha
        gain_r = (avg_g / avg_r)*gamma

        # 应用增益来进行白平衡
        balanced = cv2.merge([img_float[:, :, 0] * gain_b, img_float[:, :, 1]*beta, img_float[:, :, 2] * gain_r])

        # 将结果限制在0到255的范围内
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        return balanced
    
    # 保存图片  
    def save_image(self, img, filename):  
        cv2.imwrite(filename, img)  
        print(f"Image saved to {filename}")  

#  
if __name__ == '__main__':  
    enhancer = Color_Balance_and_Fusion_for_Underwater_Image_Enhancement()  
    img = cv2.imread("8.png")  
    img_balenced_1 = enhancer.color_balance(img)  
    img_balenced_ours=enhancer.gray_world(img_balenced_1)

    enhancer.save_image(img_balenced_ours, "8_img_balanced_ours.png")  
   
    cv2.imshow('result', img_balenced_ours)  
    cv2.waitKey(0)
    cv2.destroyAllWindows() 