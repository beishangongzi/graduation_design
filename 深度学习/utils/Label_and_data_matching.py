# create by andy at 2022/4/18
# reference:
import os


def process(data_dir, label_dir):
    ls_data = set(os.listdir(data_dir))
    ls_label = set(os.listdir(label_dir))
    res = ls_data.difference(ls_label).union(ls_label.difference(ls_data))
    print(res)
    for file in res:
        try:
            os.remove(os.path.join(data_dir, file))
        except:
            os.remove(os.path.join(label_dir, file))


if __name__ == '__main__':
    process("/media/andy/z/python/graduation_design/深度学习/others/CV-Papers-Codes/FCN/data/BagImages",
            "/media/andy/z/python/graduation_design/深度学习/others/CV-Papers-Codes/FCN/data/BagImagesMasks")

    process("/media/andy/z/python/graduation_design/深度学习/others/CV-Papers-Codes/FCN/data/testImages",
            "/media/andy/z/python/graduation_design/深度学习/others/CV-Papers-Codes/FCN/data/testMasks")