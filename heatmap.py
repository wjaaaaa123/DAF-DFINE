import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np
np.random.seed(0)
from tqdm import trange
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from engine.core import YAMLConfig
from engine.obbdeim.box_ops import xywhr_to_poly
from tools.inference.utils import draw
from tools.visualization.val_preprocess import build_visual_preprocessor, select_cam_input

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
CLASS_NAME = None


def unpack_postprocessor_outputs(outputs):
    if isinstance(outputs, tuple):
        if len(outputs) == 3:
            labels, boxes, scores = outputs
            return labels, boxes, scores, None
        if len(outputs) == 4:
            labels, boxes, scores, masks = outputs
            return labels, boxes, scores, masks
    raise ValueError(f"Unsupported postprocessor output format: {type(outputs)}")


def get_draw_kwargs(is_seg=False, is_obb=False, masks=None):
    if is_obb:
        return {"box_format": "xywhr"}
    if is_seg:
        return {"masks": masks, "box_format": "xyxy"}
    return {"box_format": "xyxy"}


def boxes_for_cam_renormalize(boxes, is_obb=False):
    if not is_obb or boxes.numel() == 0:
        return boxes
    polygons = xywhr_to_poly(boxes).reshape(-1, 4, 2)
    min_xy = polygons.min(dim=1).values
    max_xy = polygons.max(dim=1).values
    return torch.cat([min_xy, max_xy], dim=-1)


class MultimodalGradCAMWrapper(torch.nn.Module):
    def __init__(self, model, cam_input_key="rgb"):
        super().__init__()
        self.model = model
        self.cam_input_key = cam_input_key
        self.samples = None

    def set_samples(self, samples):
        if not isinstance(samples, dict):
            raise TypeError("Multimodal Grad-CAM wrapper expects dict samples.")
        if self.cam_input_key not in samples:
            raise KeyError(f"Missing CAM input key '{self.cam_input_key}'. Available keys: {list(samples.keys())}")
        self.samples = samples

    def forward(self, cam_tensor):
        if self.samples is None:
            raise RuntimeError("Multimodal Grad-CAM wrapper samples were not initialized.")
        samples = dict(self.samples)
        samples[self.cam_input_key] = cam_tensor
        return self.model(samples)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        boxes, logits = result['pred_boxes'], result['pred_logits']
        sorted, indices = torch.sort(logits.max(2)[0], descending=True)
        return logits[0, indices][0], boxes[0, indices][0]
  
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        logits, boxes = self.post_process(model_output)
        return [[logits, boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()

class deim_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        logits, boxes = data
        result = []

        # 最多取前3个高置信度 query
        num_targets = max(
            1,
            min(3, int(logits.size(0) * self.ratio))
        )

        # 用最高置信度 query 的类别作为主要解释类别
        scores = logits.softmax(-1).max(-1)[0]

        best_id = scores.argmax()

        target_class = int(
            logits[best_id].argmax().item()
        )

        topk = torch.topk(scores, k=3).indices

        for i in topk:

            # 所有 query 统一解释同一个类别
            cls_score = logits[i, target_class]

            if float(cls_score) < self.conf:
                continue

            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(cls_score)

            if self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(boxes.shape[-1]):
                    result.append(boxes[i, j])

        if not result:
            return logits[0, target_class]

        return sum(result)

def get_param_by_string(model, param_str):
    # 分割字符串，按 '.' 进行分割，得到各个层次
    keys = param_str.split('.')
    
    # 从模型开始，逐步获取每一层
    param = model
    for key in keys:  # 逐层访问，直到最后一层
        if key.isdigit():  # 如果是数字，说明是一个列表的索引
            key = int(key)  # 将字符串转换为整数索引
            param = param[key]
        else:
            param = getattr(param, key)  # 动态访问属性

    return param

class deim_heatmap:
    def __init__(
        self,
        config,
        weight,
        device,
        method,
        layer,
        backward_type,
        conf_threshold,
        ratio,
        show_box,
        renormalize,
        isUltralytics,
        modality_path=None,
        cam_input_key="rgb",
    ):
        device = torch.device(device)

        model, postprocessor, cfg = self.init_model(config, weight)
        model.to(device)
        model.eval()
        visual_preprocessor = build_visual_preprocessor(cfg)

        target = deim_target(backward_type, conf_threshold, ratio)
        if isUltralytics:
            target_layers = []
            for l in layer:
                target_layers.append(eval(l))
        else:
            target_layers = [get_param_by_string(model, l) for l in layer]
        cam_model = (
            MultimodalGradCAMWrapper(model, cam_input_key=cam_input_key)
            if visual_preprocessor.is_multimodal
            else model
        )
        method = eval(method)(cam_model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(cam_model, target_layers, None)

        self.__dict__.update(locals())
    
    def init_model(self, config, weight):
        global CLASS_NAME
        cfg = YAMLConfig(config, resume=weight)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
        checkpoint = torch.load(weight, map_location='cpu')
        if checkpoint.get('name', None) != None:
            CLASS_NAME = checkpoint['name']
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        
        # Load train mode state and convert to deploy mode
        cfg.model.load_state_dict(state)
        return cfg.model, cfg.postprocessor.deploy(), cfg

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        h, w, _ = image_float_np.shape
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            try:
                x1, y1 = max(x1 , 0) , max(y1, 0) 
                x2, y2 = min(grayscale_cam.shape[1] - 1, x2) , min(grayscale_cam.shape[0] - 1, y2) 
                renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy()) 
            except Exception as e:
                continue  
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam)
        return eigencam_image_renormalized
    
    def run_postprocessor(self, pred, orig_size, resize_pad=None):
        if resize_pad is not None:
            return self.postprocessor(pred, orig_size, resize_pad=resize_pad)
        return self.postprocessor(pred, orig_size)

    def post_process(self, pred, orig_size, resize_pad=None, is_obb=False):
        labels, boxes, scores, _ = unpack_postprocessor_outputs(
            self.run_postprocessor(pred, orig_size, resize_pad=resize_pad)
        )
        boxes = boxes[scores > self.conf_threshold]
        return boxes_for_cam_renormalize(boxes, is_obb=is_obb)

    def process(self, img_path, save_path, modality_path=None):
        bundle = self.visual_preprocessor.prepare(
            img_path,
            device=self.device,
            modality_path=modality_path,
            model_input_key=self.cam_input_key,
        )
        im_pil = bundle.vis_image
        w, h = im_pil.size
        cam_input = select_cam_input(bundle.samples, key=self.cam_input_key)
        if bundle.is_multimodal:
            self.cam_model.set_samples(bundle.samples)
        
        try:
            grayscale_cam = self.method(cam_input, [self.target])
        except AttributeError as e:
            print(f"Warning... self.method(tensor, [self.target]) failure.")
            return
        
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (w, h))

        grayscale_cam=cv2.GaussianBlur(
            grayscale_cam,
            (3,3),
            0
        )

        # 抑制低响应区域，突出真正高响应位置
        grayscale_cam = np.clip(grayscale_cam, 0, 1)
        grayscale_cam = np.power(grayscale_cam, 1.5)

        pred = self.model(bundle.model_input)
        if self.renormalize:
            boxes = self.post_process(
                pred,
                bundle.orig_size,
                resize_pad=bundle.resize_pad,
                is_obb=bundle.is_obb,
            )
            cam_image = self.renormalize_cam_in_bounding_boxes(boxes.cpu().detach().numpy().astype(np.int32), np.array(im_pil) / 255.0, grayscale_cam)
        else:
            cam_image = show_cam_on_image(np.array(im_pil) / 255.0, grayscale_cam)
        cam_image = Image.fromarray(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
        if self.show_box:
            labels, boxes, scores, masks = unpack_postprocessor_outputs(
                self.run_postprocessor(pred, bundle.orig_size, resize_pad=bundle.resize_pad)
            )
            cam_image = draw(
                [cam_image],
                labels,
                boxes,
                scores,
                **get_draw_kwargs(bundle.is_seg, bundle.is_obb, masks),
                thrh=self.conf_threshold,
                class_name=CLASS_NAME,
            )
        cam_image.save(save_path)
    
    def show_layer(self):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'ModuleList':
                continue
            print(BLUE + f"Layer Name: " + ORANGE + name + BLUE + ", Layer Type: ", ORANGE, module.__class__.__name__, RESET)
    
    def __call__(self, img_path, save_path, modality_path=None):
        modality_path = modality_path if modality_path is not None else self.modality_path
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            if modality_path is not None and os.path.isfile(modality_path):
                raise ValueError("modality_path must be a directory when img_path is a directory.")
            for img_path_ in os.listdir(img_path):
                if os.path.splitext(img_path_)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}', modality_path=modality_path)
        else:
            self.process(img_path, f'{save_path}/result.png', modality_path=modality_path)

def get_params():

    params = {

        'config':
        r'configs\ultralytics-yaml\dfine_hgnetv2_s_mg_all_50.yml',

        'weight':
        r'outputs\dfine_hgnetv2_s_all_cmd\best_stg2.pth',

        'device':
        'cuda:0',

        'method':
        'GradCAMPlusPlus',

        'layer':
        [
        'encoder.17'
        ],

        'backward_type':
        'class',

        'conf_threshold':
        0.2,

        'ratio':
        0.1,

        'show_box':
        False,

        'renormalize':
        False,

        'isUltralytics':
        False,
    }

    return params

# 需要安装grad-cam==1.5.4



if __name__ == '__main__':

    model = deim_heatmap(**get_params())

    # 第一次运行查看模型层
   # model.show_layer()

    # 生成热力图
    model(
        r'D:\Users\Administrator\Desktop\研究生\论文\sic\deim\图片\grad',
        r'D:\Users\Administrator\Desktop\研究生\论文\sic\deim\图片\grad_results\dfinecmd17',
    )