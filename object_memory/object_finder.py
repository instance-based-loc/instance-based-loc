import numpy as np
import torch
import PIL
from huggingface_hub import hf_hub_download

from utils.logging import conditional_log
from .object_finder_phrases import filter_caption

from ram.models import ram
from ram import inference_ram
from ram import get_transform as get_transform_ram

# VSCode might not detect this despite being installed (but the code will run) 
# Add the path to grounding dino if needed to VSCode's path (will be suggested)
from groundingdino.models import build_model as gd_build_model
from groundingdino.util.slconfig import SLConfig as gd_SLConfig
from groundingdino.util.inference import annotate as gd_annotate 
from groundingdino.util.inference import load_image as gd_load_image
from groundingdino.util.inference import predict as gd_predict
from groundingdino.util.utils import clean_state_dict as gd_clean_state_dict
from groundingdino.util.box_ops import box_cxcywh_to_xyxy as gd_box_cxcywh_to_xyxy


from segment_anything import build_sam, SamPredictor

print("\033[34mLoaded modules for object_memory.object_finder\033[0m")


class ObjectFinder():
    """
        Class that detects objects in a given image.
    """

    @classmethod
    def _log(cls, statement: any) -> None:
        conditional_log(statement, cls.log_enabled)


    @classmethod
    def _setup_ram(cls):
        cls._log("Loading RAM")
        cls.ram_model = ram(pretrained=cls.ram_pretrained_path, image_size=cls.image_size, vit=cls.ram_model)
        cls.ram_model.eval()
        cls.ram_model.to(cls.device)
        cls.ram_transform = get_transform_ram(image_size=cls.image_size)

    @classmethod
    def _setup_grounding_dino(cls):
        cls._log("Loading Grounding Dino")

        cache_config_file = hf_hub_download(repo_id=cls.ckpt_repo_id, filename=cls.ckpt_config_filename)
        args = gd_SLConfig.fromfile(cache_config_file)
        args.device = cls.device
        cls.groundingdino_model = gd_build_model(args)

        cache_file = hf_hub_download(repo_id=cls.ckpt_repo_id, filename=cls.ckpt_filenmae)
        checkpoint = torch.load(cache_file, map_location=cls.device)
        _ = cls.groundingdino_model.load_state_dict(gd_clean_state_dict(checkpoint['model']), strict=False)
        cls.groundingdino_model.eval()

    @classmethod
    def _setup_sam(cls):
        cls._log("Loading SAM")
        cls.sam_predictor = SamPredictor(build_sam(checkpoint=cls.sam_checkpoint_path).to(cls.device).eval())

    @classmethod
    def setup(
        cls,
        device,
        ram_pretrained_path,
        sam_checkpoint_path,
        gd_detection_box_threshold = 0.35,
        gd_detection_text_threshold = 0.55,
        gd_box_processing_intersection_threshold = 0.7,
        gd_box_processing_size_threshold = 0.75, 
        image_size = 384,
        ram_model = 'swin_l',
        gd_ckpt_repo_id = "ShilongLiu/GroundingDINO",
        gd_ckpt_filename = "groundingdino_swinb_cogcoor.pth",
        gd_ckpt_config_filename = "GroundingDINO_SwinB.cfg.py",
        log_enabled = True

    ) -> None:
        cls.device = device
        cls.ram_pretrained_path = ram_pretrained_path
        cls.sam_checkpoint_path = sam_checkpoint_path
        cls.image_size = image_size
        cls.gd_detection_box_threshold = gd_detection_box_threshold
        cls.gd_detection_text_threshold = gd_detection_text_threshold
        cls.gd_box_processing_intersection_threshold = gd_box_processing_intersection_threshold
        cls.gd_box_processing_size_threshold = gd_box_processing_size_threshold
        cls.ram_model = ram_model
        cls.gd_ckpt_repo_id = gd_ckpt_repo_id
        cls.gd_ckpt_filename = gd_ckpt_filename
        cls.gd_ckpt_config_filename = gd_ckpt_config_filename
        cls.log_enabled = log_enabled

        cls._setup_ram()
        cls._setup_grounding_dino()
        cls._setup_sam()

    @classmethod
    def get_bounding_boxes_and_phrases(cls, image: torch.Tensor, keywords: list[str]) -> tuple[torch.Tensor, list]:
        def get_box_iou(rect1, rect2) -> float:
            area_rect1 = rect1[2]*rect1[3]
            area_rect2 = rect2[2]*rect2[3]

            overlap_top_left = (max(rect1[0], rect2[0]), max(rect1[1], rect2[1]))
            overlap_bottom_right = (min(rect1[0] + rect1[2], rect2[0] + rect2[2]), min(rect1[1] + rect1[3], rect2[1] + rect2[3]))

            # no overload condition
            if (overlap_bottom_right[0] <= overlap_top_left[0]) or (overlap_bottom_right[1] <= overlap_top_left[1]):
                return 0.0

            # Calculate the area of the overlap rectangle
            overlap_area = abs((overlap_bottom_right[0] - overlap_top_left[0]) * (overlap_bottom_right[1] - overlap_top_left[1]))
            return (overlap_area / min(area_rect1, area_rect2))

        def get_box_comparison(rect1, rect2) -> float:
            area_rect1 = rect1[2]*rect1[3]
            area_rect2 = rect2[2]*rect2[3]

            return min(area_rect1, area_rect2)/max(area_rect1, area_rect2)

        returned_boxes, returned_phrases = [], []

        with torch.no_grad():
            for i, word in enumerate(keywords):
                detected_boxes, _, detected_phrases = gd_predict(
                    model=cls.groundingdino_model,
                    image=image,
                    caption=word,
                    box_threshold=cls.gd_detection_box_threshold,
                    text_threshold=cls.gd_detection_text_threshold,
                    device=cls.device
                )

                cls._log(f"Index {i} / Word {word}: {detected_phrases}")

                if detected_boxes is not None and len(detected_phrases) > 0:
                    if len(returned_boxes) > 0:
                        for box in detected_boxes:
                            box_is_unique = True

                            for prev_box in returned_boxes:
                                if (get_box_iou(box, prev_box) > cls.gd_box_processing_intersection_threshold and 
                                    get_box_comparison(box, prev_box) > cls.gd_box_processing_size_threshold):
                                    box_is_unique = False
                                    break

                            if box_is_unique:
                                returned_boxes.append(box)
                                returned_phrases.append(word)

                    else:
                        # First Box
                        for box in detected_boxes:
                            returned_boxes.append(box)
                            returned_phrases.append(word)

        if len(returned_boxes) > 0:
            # torch.stack cannot work on an empty lis
            return torch.stack(returned_boxes), returned_phrases
        
        return None, None
    
    @classmethod
    def segment_from_bounding_boxes(cls, image: np.array, cxcy_boxes: torch.Tensor):
        with torch.no_grad():
            cls.sam_predictor.set_image(image)
            H, W, _ = image.shape
            boxes_xyxy = gd_box_cxcywh_to_xyxy(cxcy_boxes) * torch.Tensor([W, H, W, H])

            transformed_boxes = cls.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(cls.device), image.shape[:2])
            masks, _, _ = cls.sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
                )
            return boxes_xyxy, masks

    @classmethod
    def find(cls, image_path: str, caption: list[str] = None) -> tuple[list, torch.Tensor, torch.Tensor, list]:

        image_source, image = gd_load_image(image_path)

        # Use of RAM
        if caption is None or len(caption) == 0: 
            img_ram = cls.ram_transform(PIL.Image.fromarray(image_source)).unsqueeze(0).to(cls.device)
            caption = inference_ram(img_ram, cls.ram_model)[0].split("|")

        filtered_caption = filter_caption(caption)
        cls._log(f"Filtered caption post RAM: {filtered_caption}")

        # Use of GroundingDINO
        cxcy_boxes, phrases = cls.get_bounding_boxes_and_phrases(image, filtered_caption)

        if cxcy_boxes is None or phrases is None:
            cls._log("Grounding DINO could not find anything.")
            return None, None, None, None

        # Use of SAM
        boxes, masks = cls.segment_from_bounding_boxes(image_source, cxcy_boxes)

        # ground objects
        grounded_objects = [image_source[int(bb[1]):int(bb[3]),
                                         int(bb[0]):int(bb[2]), :] for bb in boxes]

        return grounded_objects, boxes, masks, phrases