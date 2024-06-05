# 源代码:https://github.com/falrom/MSRCR_Python/blob/master/MSRCR.py

import numpy as np
import argparse
import cv2

#封装成类
class Color_Balance_and_Fusion_for_Underwater_Image_Enhancement():
    def __init__(self):
        pass
    def retinex_scales_distribution(self,max_scale, nscales):
      scales = []
      scale_step = max_scale / nscales
      for s in range(nscales):
        scales.append(scale_step * s + 2.0)
      return scales


    def CR(self,im_ori, im_log, alpha=128., gain=1., offset=0.):
      im_cr = im_log * gain * (
            np.log(alpha * (im_ori + 1.0)) - np.log(np.sum(im_ori, axis=2) + 3.0)[:, :, np.newaxis]) + offset
      return im_cr

    def MSRCR(self,image_path, max_scale, nscales, dynamic=2.0, do_CR=True):
      im_ori = np.float32(cv2.imread(image_path)[:, :, (0, 1, 2)])
      scales = self.retinex_scales_distribution(max_scale, nscales)

      im_blur = np.zeros([len(scales), im_ori.shape[0], im_ori.shape[1], im_ori.shape[2]])
      im_mlog = np.zeros([len(scales), im_ori.shape[0], im_ori.shape[1], im_ori.shape[2]])

      for channel in range(3):
        for s, scale in enumerate(scales):
            # If sigma==0, it will be automatically calculated based on scale
            im_blur[s, :, :, channel] = cv2.GaussianBlur(im_ori[:, :, channel], (0, 0), scale)
            im_mlog[s, :, :, channel] = np.log(im_ori[:, :, channel] + 1.) - np.log(im_blur[s, :, :, channel] + 1.)

      im_retinex = np.mean(im_mlog, 0)
      if do_CR:
        im_retinex = self.CR(im_ori, im_retinex)

      im_rtx_mean = np.mean(im_retinex)
      im_rtx_std = np.std(im_retinex)
      im_rtx_min = im_rtx_mean - dynamic * im_rtx_std
      im_rtx_max = im_rtx_mean + dynamic * im_rtx_std

      im_rtx_range = im_rtx_max - im_rtx_min

      im_out = np.uint8(np.clip((im_retinex - im_rtx_min) / im_rtx_range * 255.0, 0, 255))

      return im_out

    # 保存图片  
    def save_image(self,img, filename):  
        cv2.imwrite(filename, img)  
        print(f"Image saved to {filename}")  
        pass

#  
if __name__ == '__main__':  
    result = Color_Balance_and_Fusion_for_Underwater_Image_Enhancement()  
    img =r'8_img_balanced_ours.png'
    img_res = result.MSRCR(img,max_scale=300, nscales=3, dynamic=2, do_CR=True)  
    result.save_image(img_res, "8_result.png")  
   
    cv2.imshow('result', img_res)  
    cv2.waitKey(0)
    cv2.destroyAllWindows() 