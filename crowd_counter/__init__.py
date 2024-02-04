import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2

from .engine import *
from .models import build_model
import os
import warnings

warnings.filterwarnings("ignore")


class Args:
    def __init__(
        self,
        backbone: str,
        row: int,
        line: int,
        output_dir: str,
        weight_path: str,
        # gpu_id: int,
    ) -> None:
        self.backbone = backbone
        self.row = row
        self.line = line
        self.output_dir = output_dir
        self.weight_path = weight_path
        # self.gpu_id = gpu_id


class CrowdCounter:
    def __init__(self) -> None:
        # Create the Args object
        self.args = Args(
            backbone="vgg16_bn",
            row=2,
            line=2,
            output_dir="./crowd_counter/preds",
            weight_path="./crowd_counter/weights/SHTechA.pth",
            # gpu_id=0,
        )

        # device = torch.device('cuda')
        self.device = torch.device("cpu")
        # get the P2PNet
        self.model = build_model(self.args)
        # move to GPU
        self.model.to(self.device)
        # load trained model
        if self.args.weight_path is not None:
            checkpoint = torch.load(self.args.weight_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
        # convert to eval mode
        self.model.eval()
        # create the pre-processing transform
        self.transform = standard_transforms.Compose(
            [
                standard_transforms.ToTensor(),
                standard_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def test(
        self, args: Args, img_path: str, debug: bool = False, returned: bool = False
    ) -> None or tuple[any, Image.Image, torch.Tensor]:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_id)
        # print(img_path)

        # set your image path here
        clean_img_path = img_path.split("/")[-1]

        # load the images
        img_raw = Image.open(img_path).convert("RGB")
        # round the size
        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)
        # pre-proccessing
        img = self.transform(img_raw)

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(self.device)
        # run inference
        outputs = self.model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[
            :, :, 1
        ][0]

        outputs_points = outputs["pred_points"][0]

        threshold = 0.5
        # filter the predictions
        conf = outputs_scores[outputs_scores > threshold]
        points = (
            outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        )
        predict_cnt = int((outputs_scores > threshold).sum())

        outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[
            :, :, 1
        ][0]

        outputs_points = outputs["pred_points"][0]

        if not returned:
            # draw the predictions
            size = 5
            img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
            output_file = open(
                args.output_dir + "/" + clean_img_path.split(".")[0] + ".txt", "w"
            )
            for p in points:
                img_to_draw = cv2.circle(
                    img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1
                )
                output_file.write(str(p[0]) + " ")
                output_file.write(str(p[1]) + "\n")
            # save the visualized image
            cv2.imwrite(os.path.join(args.output_dir, clean_img_path), img_to_draw)
        else:
            return points, img_raw, conf

    # Function to process and save images
    def inference(self, file_path: str) -> tuple[int, float]:

        # Predict points on the image
        points, img_raw, conf = self.test(self.args, file_path, returned=True)

        # Draw the predictions
        size_cover = 20
        img_to_draw_cover = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        for p in points:
            img_to_draw_cover = cv2.circle(
                img_to_draw_cover, (int(p[0]), int(p[1])), size_cover, (0, 0, 255), -1
            )

        red_lower = np.array([0, 0, 250], np.uint8)
        red_upper = np.array([5, 5, 255], np.uint8)
        red_mask = cv2.inRange(img_to_draw_cover, red_lower, red_upper)

        # Calculate the percentage of the image covered by red points
        covered_area = np.sum(red_mask > 0)
        total_area = red_mask.size
        coverage_percentage = (
            (covered_area / total_area) * 100 + 15 + 3
        )  # constant area mekah + celah
        if coverage_percentage > 95:
            coverage_percentage = 95.12

        # Prepare text for the number of points
        num_points = len(points)
        # pilgrims_text = f"Pilgrims   : {num_points}"
        # coverage_text = f"Crowdness : {coverage_percentage:.2f}%"

        # Pilgrims, Crowdness %
        return num_points, coverage_percentage
