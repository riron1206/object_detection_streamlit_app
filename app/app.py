"""
efficientdet_d1で物体検出
Usage:
    $ cd ../work/app
    $ streamlit run app.py --server.port 8889
"""

# ====================================================
# Library
# ====================================================
import os
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset

from effdet import create_model, create_dataset, create_loader
from effdet.data import resolve_input_config

import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# https://github.com/amikelive/coco-labels/blob/master/coco-labels-paper.txt
COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
              "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
              "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis",
              "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
              "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
              "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    if data == 'test':
        return A.Compose(
            [
                A.Resize(height=640, width=640, p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2()
            ],
            p=1.0
        )
    else:
        return A.Compose(
            [
                A.Resize(height=640, width=640, p=1.0)
            ],
            p=1.0
        )


# ====================================================
# Model
# ====================================================
def get_bench_model(device):
    """imagenetの学習済みのeffdetロード"""
    # https://zenn.dev/nnabeyang/articles/30330c3ab7d449c66338
    bench = create_model(
        'tf_efficientdet_d1', # tf_efficientdet_d1 はうまくいく。他のモデルは画像サイズが違うのか全然ダメ
        bench_task='predict',
        pretrained=True,
        redundant_bias=None,
        soft_nms=None,
        checkpoint_ema='use_ema',
    )
    bench = bench.to(device)
    return bench


# ====================================================
# Helper functions
# ====================================================
def pred_top(bench, images):
    """effdetのモデルで予測して、各画像の確信度1位のbboxだけ返す"""
    bench.eval()
    with torch.no_grad():
        output_batch = bench(images)
    top_outs = []
    for output in output_batch:
        out = output[0]
        out = [o.cpu().numpy() for o in out]
        xmin, ymin, xmax, ymax, score, label = out
        top_outs.append([int(xmin), int(ymin), int(xmax), int(ymax), score, COCO_NAMES[int(label)-1]])
    return top_outs


def pred_score_th(bench, images, score_th=0.5):
    """effdetのモデルで予測して、各画像の閾値以上の確信度のbboxだけ返す"""
    bench.eval()
    with torch.no_grad():
        output_batch = bench(images)
    outs = []
    for output in output_batch:
        for out in output:
            out = [o.cpu().numpy() for o in out]
            xmin, ymin, xmax, ymax, score, label = out
            if score < score_th: # 閾値
                break
            outs.append([int(xmin), int(ymin), int(xmax), int(ymax), score, COCO_NAMES[int(label)-1]])
    return outs


def pred_cv2_rectangle(image, xmin, ymin, xmax, ymax, pred, label):
    """画像にbboxとラベル名書き込む"""
    cv2.rectangle(image,
                  pt1=(int(xmin), int(ymin)),
                  pt2=(int(xmax), int(ymax)),
                  color=(0, 220, 0),
                  thickness=5)
    cv2.putText(image,
                text=str(label) + ":" + str(round(pred, 3)),
                org=(int(xmin), int(ymin)-10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 220, 0),
                thickness=5)
    return image


def img_transform(cv2_img, transform=None):
    """albmentationで変換"""
    image = cv2_img.astype(np.float32)
    if transform:
        sample = transform(**{'image': image})
        return sample['image']
    else:
        return image


def one_img_bbox_show_save(outs, cv2_img, is_show=True, save_crop_dir=None):
    """画像1枚分のモデル出力から可視化及びbboxを画像に保存する"""
    if is_show:
        # 描画用に画像ロード
        image = img_transform(cv2_img, transform=get_transforms(data=""))
        image = image/255
        for out in outs:
            image = pred_cv2_rectangle(image,
                                       out[0], out[1],
                                       out[2], out[3],
                                       out[4], out[5])
        plt.imshow(image)
        plt.show()

    if save_crop_dir is not None:
        # bboxを切り抜いて保存する
        image = cv2_img(cv2_img, transform=get_transforms(data=""))
        for i, out in enumerate(outs):
            crop_bbox = image[int(out[1]):int(out[3]), int(out[0]):int(out[2])]
            crop_bbox = cv2.cvtColor(crop_bbox, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{save_crop_dir}/{Path(img_path).stem}_{str(i)}_{str(out[5])}.png', crop_bbox)


def run_pred(bench, cv2_img,
             score_th=None, is_show=True, save_crop_dir=None):
    """画像1件だけ予測する場合"""
    # ====================================================
    # 画像1件だけ予測
    # ====================================================
    image = img_transform(cv2_img, transform=get_transforms(data="test"))
    images = image.unsqueeze(dim=0).to(device)
    if score_th is not None:
        # 確信度の閾値決める。bboxはN件ある
        outs = pred_score_th(bench, images, score_th=score_th)
    else:
        # bbox1件だけ
        outs = pred_top(bench, images)

    # ====================================================
    # 可視化及びbboxを画像に保存
    # ====================================================
    one_img_bbox_show_save(outs, cv2_img, is_show=is_show, save_crop_dir=save_crop_dir)

    # ====================================================
    # 予測結果をデータフレームに詰める
    # ====================================================
    outs = np.array(outs)
    out_df = pd.DataFrame(
        {#"image_id":"",
         #"file_path":"",
         "resize_h":640,
         "resize_w":640,
         "xmin":outs[:,0], "ymin":outs[:,1],
         "xmax":outs[:,2], "ymax":outs[:,3],
         "score":outs[:,4], "label":outs[:,5]})
    return out_df

# ==============================================
# app.py
# ==============================================
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_model():
    model = get_bench_model(device)
    return model


def load_file_up_image(file_up, size=640):
    pillow_img = Image.open(file_up).convert("RGB")
    pillow_img = pillow_img.resize((size, size)) if size is not None else pillow_img
    cv2_img = pil2cv(pillow_img)
    return pillow_img, cv2_img


def pil2cv(pillow_img):
    """ PIL型 -> OpenCV型
    https://qiita.com/derodero24/items/f22c22b22451609908ee"""
    new_image = np.array(pillow_img, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def main():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.title("Simple Object Detection App (EfficientDet-d1)")
    st.write("")

    # ファイルupload
    file_up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    # サイドバー スライダー
    st_score_th = st.sidebar.slider("BBox Score Threshold", 0.0, 1.0, step=0.01, value=0.5)

    # サイドバー
    st.sidebar.write("GPU Not available on hosted demo, try on your local!")
    st.sidebar.markdown("Clone Demo [Code](https://github.com/riron1206/object_detection_streamlit_app)")

    def run():
        pillow_img, cv2_img = load_file_up_image(file_up)

        st.image(
            pillow_img,
            caption="Uploaded Image. Resize (640, 640).",
            use_column_width=True,
        )

        st.write("")
        bench = load_model()
        try:
            out_df = run_pred(bench, cv2_img, score_th=st_score_th, is_show=False, save_crop_dir=None)
        except:
            out_df = None

        if out_df is not None:
            # 描画用に画像ロード
            image = img_transform(cv2_img, transform=get_transforms(data=""))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for idx, row in out_df.iterrows():
                image = pred_cv2_rectangle(image,
                                        row["xmin"], row["ymin"],
                                        row["xmax"], row["ymax"],
                                        float(row["score"]), row["label"]
                                        )
            st.write("## Prediction Result")
            st.image(image/255)
            st.write("## BBox info")
            st.dataframe(out_df)
        else:
            st.write("## BBox None")

    if file_up is not None:
        run()
    else:
        img_url = "https://github.com/riron1206/image_classification_streamlit_app/blob/master/image/dog.jpg?raw=true"
        st.image(
            img_url,
            caption="Sample Image. Please download and upload.",
            use_column_width=True,
        )


if __name__ == "__main__":
    main()
