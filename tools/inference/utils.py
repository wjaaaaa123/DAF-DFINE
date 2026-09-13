import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Ultralytics YOLO palette
# Directly adapted from the user's Ultralytics plotting.py
# ============================================================

class Colors:
    """Ultralytics default color palette."""

    def __init__(self):
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

        # Ultralytics text-color logic
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

    def get_txt_color(self, color=(128, 128, 128), txt_color=(255, 255, 255)):
        """
        仅白色/浅灰色标签背景使用黑色文字，
        其余所有标签背景统一使用白色文字。
        Ultralytics 当前调色板中的浅灰色为 #F3F3F3 = (243, 243, 243)。
        """
        if tuple(color) == (243, 243, 243):
            return (0, 0, 0)
        return (255, 255, 255)


colors = Colors()


def get_color_by_class(class_id):
    """Compatibility helper: returns exact Ultralytics palette RGB color."""
    return colors(class_id)


def _load_yolo_font(font_size):
    """
    Use Arial like Ultralytics.
    Falls back to common Windows fonts, then PIL default.
    """
    font_paths = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def _text_size(draw_obj, text, font):
    bbox = draw_obj.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw(
    images,
    labels,
    boxes,
    scores,
    masks=None,
    tracks=None,
    thrh=0.4,
    font_scale=1.0,
    box_thickness=0,
    class_name=None,
    mask_alpha=0.5,
    box_format="xyxy",
):
    """
    D-FINE / DEIM drawing with Ultralytics YOLO visualization style.

    Key behavior copied from the user's YOLO plotting.py:
      line width = max(round((w + h) / 2 * 0.003), 2)
      font size  = max(round((w + h) / 2 * 0.035), 12)
      exact YOLO palette
      solid label background
      class-dependent text color

    Args:
        font_scale:
            1.0 = same dynamic font size rule as YOLO.
            1.1 / 1.2 = larger than YOLO if desired.
        box_thickness:
            0 = use YOLO automatic line width.
            Positive integer = force that line width.
    """

    if box_format not in {"xyxy", "xywhr"}:
        raise ValueError(f"Unsupported box_format: {box_format}")

    if box_format == "xywhr" and (masks is not None or tracks is not None):
        raise ValueError('box_format="xywhr" does not support masks or tracks')

    results = []

    for i, im in enumerate(images):
        # D-FINE sends PIL images here.
        if not isinstance(im, Image.Image):
            if isinstance(im, np.ndarray):
                im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            else:
                im = Image.fromarray(np.asarray(im))

        out_im = im.copy()
        w, h = out_im.size

        # Exact Ultralytics sizing formulas.
        yolo_lw = max(round((w + h) / 2 * 0.003), 2)
        lw = int(box_thickness) if box_thickness and box_thickness > 0 else int(yolo_lw)

        base_font_size = max(round((w + h) / 2 * 0.035), 12)
        font_size = max(round(base_font_size * float(font_scale)), 12)

        font = _load_yolo_font(font_size)
        draw_obj = ImageDraw.Draw(out_im)

        # ----------------------------------------------------
        # Select detections
        # ----------------------------------------------------
        if tracks is None:
            scr = scores[i]
            keep = scr > thrh
            lab = labels[i][keep]
            box = boxes[i][keep]
            scrs = scr[keep]
        else:
            scrs = tracks[i][:, 5]
            lab = tracks[i][:, 6]
            box = tracks[i][:, :4]
            track_id = tracks[i][:, 4]

        if masks is not None and tracks is None:
            msk = masks[i][keep]

        # ----------------------------------------------------
        # OBB polygons
        # ----------------------------------------------------
        if box_format == "xywhr" and len(box):
            from engine.obbdeim.box_ops import xywhr_to_poly
            polygons = (
                xywhr_to_poly(box)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, 4, 2)
            )
        else:
            polygons = None

        # ----------------------------------------------------
        # Draw each detection
        # ----------------------------------------------------
        for j, b in enumerate(box):
            if torch.is_tensor(lab[j]):
                lab_id = int(lab[j].item())
            else:
                lab_id = int(lab[j])

            color = colors(lab_id)  # exact YOLO RGB palette

            if torch.is_tensor(scrs[j]):
                score = float(scrs[j].item())
            else:
                score = float(scrs[j])

            if class_name is not None and lab_id < len(class_name):
                name = str(class_name[lab_id]).strip()
            else:
                name = str(lab_id)

            # ============================================================
            # 仅修改可视化显示名称，与 YOLO 图中的类别名称保持一致
            # 不改变 class_id、模型预测结果、评价指标或权重
            # ============================================================
            DISPLAY_NAME_MAP = {
                "Nonconductive": "Nonconductive",
                "Scratch": "Scratch",
                "CornerLeak": "Corner Leak",
                "Corner_Leak": "Corner Leak",
                "OrangePeel": "Orange Peel",
                "Orange_Peel": "Orange Peel",
                "Leakage": "Leakage",
                "Jet": "Jet",
                "PaintBubble": "Paint Bubble",
                "Paint_Bubble": "Paint Bubble",
                "Crater": "Crater",
                "Parti-color": "Parti-color",
                "PartiColor": "Parti-color",
                "DirtyPoint": "Dirty Point",
                "Dirty_Point": "Dirty Point",
            }
            name = DISPLAY_NAME_MAP.get(name, name)

            text = f"{name} {score:.2f}"

            if tracks is not None:
                if torch.is_tensor(track_id[j]):
                    tid = int(track_id[j].item())
                else:
                    tid = int(track_id[j])
                text = f"{tid} {text}"

            # ------------------------------
            # Bounding box
            # ------------------------------
            if box_format == "xywhr":
                points = np.round(polygons[j]).astype(np.int32)
                pts = [tuple(map(int, p)) for p in points]
                draw_obj.line(pts + [pts[0]], fill=color, width=lw, joint="curve")
                p1 = (
                    int(np.clip(points[:, 0].min(), 0, w - 1)),
                    int(np.clip(points[:, 1].min(), 0, h - 1)),
                )
            else:
                if torch.is_tensor(b):
                    b = b.detach().cpu().numpy()

                x1, y1, x2, y2 = [int(round(float(v))) for v in b]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                p1 = (x1, y1)
                draw_obj.rectangle((x1, y1, x2, y2), width=lw, outline=color)

            # ------------------------------
            # Exact YOLO-style label layout
            # ------------------------------
            text_w, text_h = _text_size(draw_obj, text, font)

            # YOLO: label outside if enough space above.
            outside = p1[1] >= text_h

            label_x = p1[0]

            # Avoid overflowing right edge.
            if label_x > w - text_w:
                label_x = max(0, w - text_w)

            if outside:
                label_y1 = p1[1] - text_h
                label_y2 = p1[1] + 1
                text_y = p1[1] - text_h
            else:
                label_y1 = p1[1]
                label_y2 = p1[1] + text_h + 1
                text_y = p1[1]

            label_x2 = min(w - 1, label_x + text_w + 1)

            draw_obj.rectangle(
                (label_x, label_y1, label_x2, label_y2),
                fill=color,
            )

            # 按类别固定文字颜色：
            # a/e/g/j -> 白色文字，对应 class_id 0/4/6/9
            # b/c/d/f/h/i -> 黑色文字，对应 class_id 1/2/3/5/7/8
            WHITE_TEXT_CLASSES = {0, 4, 6, 9}
            txt_color = (255, 255, 255) if lab_id in WHITE_TEXT_CLASSES else (0, 0, 0)

            draw_obj.text(
                (label_x, text_y),
                text,
                fill=txt_color,
                font=font,
            )

        # ----------------------------------------------------
        # Optional masks: preserved for compatibility
        # ----------------------------------------------------
        if masks is not None and tracks is None:
            im_np = np.asarray(out_im).copy()
            overlay = im_np.copy()

            for j, b in enumerate(box):
                mask = msk[j]
                if torch.is_tensor(mask):
                    mask = mask.detach().cpu().numpy()

                if mask.dtype != bool:
                    mask = mask > 0.5

                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                lab_id = int(lab[j].item()) if torch.is_tensor(lab[j]) else int(lab[j])
                color = np.array(colors(lab_id), dtype=np.uint8)
                overlay[mask] = color

            mask_area = np.any(overlay != im_np, axis=2)
            im_np[mask_area] = (
                im_np[mask_area] * (1.0 - mask_alpha)
                + overlay[mask_area] * mask_alpha
            ).astype(np.uint8)

            out_im = Image.fromarray(im_np)

        results.append(out_im)

    return results[0] if len(results) == 1 else results
