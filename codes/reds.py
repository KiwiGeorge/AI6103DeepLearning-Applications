# import os
# from PIL import Image
# from ffmpy3 import FFmpeg
#
# inputPath = '/tzz/tzz/datasets/test/test_blur'
# outputPath = '/tzz/tzz/datasets/test1/test_blur'
#
#
#
# def delAll(path):
#     if os.path.isdir(path):
#         files = os.listdir(path)  # ['a.doc', 'b.xls', 'c.ppt']
#         # 遍历并删除文件
#         for file in files:
#             p = os.path.join(path, file)
#             if os.path.isdir(p):
#                 # 递归
#                 delAll(p)
#             else:
#                 os.remove(p)
#         # 删除文件夹
#         os.rmdir(path)
#     else:
#         os.remove(path)
#
# piclist = os.listdir(inputPath)
# for pic in piclist:  # pic: /te/test_blur/000
#     inputPath1 = os.path.join(inputPath, pic)  # /000/
#     print('inputPath1', inputPath1)
#     outputPath1 = os.path.join(outputPath, pic)
#     print('outputPath1', outputPath1)
#     pic1 = os.listdir(inputPath1)
#     for pic2 in pic1:  # 00000001.png
#          picpath = os.path.join(inputPath1, pic2)  # /000/00000001.png
#          print('picpath', picpath)
#          img = Image.open(picpath)
#          in_wid, in_hei = img.size
#          out_wid = in_wid // 6 * 6
#          out_hei = in_hei // 6 * 6
#          # size = '{}x{}'.format(out_wid, out_hei)  # 输出文件会缩放成这个大小
#          outname = outputPath1 + '/' + pic2
#          print(outname)
#          os.system('mkdir -p {}'.format(outputPath1))
#
#          os.system('ffmpeg -i {} -vf scale=1224:720 {}'.format(picpath, outname))

#
import os
from PIL import Image
from ffmpy3 import FFmpeg

inputPath = '/tzz/tzz/datasets/210'
outputYUVPath = '/tzz/tzz/datasets/210_1'


piclist = os.listdir(inputPath)
for pic in piclist:
    picpath = os.path.join(inputPath, pic)
    img = Image.open(picpath)
    in_wid, in_hei = img.size
    out_wid = in_wid // 2 * 2
    out_hei = in_hei // 2 * 2
    size = '{}x{}'.format(out_wid, out_hei)
    outname = outputYUVPath + '/' + pic
    os.system('ffmpeg -i {} -vf scale=1224:720 {}'.format(picpath, outname))