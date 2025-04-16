from utils import image_preprocess, pred_bbox, sam_init, sam_out_nosave, resize_image
import os
from PIL import Image
import argparse
from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--ckpt_path", default="./../checkpoints/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    # load SAM checkpoint
    # gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    sam_predictor = sam_init(args.ckpt_path, 0)
    print("load sam ckpt done.")

    for image_path in tqdm(sorted(os.listdir(args.input_folder))):

        save_path = os.path.join(args.save_folder,image_path)
        args.image_path = os.path.join(args.input_folder,image_path)

        input_raw = Image.open(args.image_path)
        # input_raw.thumbnail([512, 512], Image.Resampling.LANCZOS)
        input_raw = resize_image(input_raw, 512)
        image_sam = sam_out_nosave(
            sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw)
        )
        image_preprocess(image_sam, save_path, lower_contrast=True, rescale=True)
