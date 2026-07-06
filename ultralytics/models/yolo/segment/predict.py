# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO segmentation model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Apply non-max suppression and process segmentation detections.
        construct_results: Construct a list of result objects from predictions.
        construct_result: Construct a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo26n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

        This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (dict, optional): Dictionary of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """Apply non-max suppression and process segmentation detections for each image in the input batch.

        Args:
            preds (tuple): Model predictions, containing bounding boxes, scores, classes, and mask coefficients.
            img (torch.Tensor): Input image tensor in model format, with shape (B, C, H, W).
            orig_imgs (list | torch.Tensor | np.ndarray): Original image or batch of images.

        Returns:
            (list): List of Results objects containing the segmentation predictions for each image in the batch. Each
                Results object includes both bounding boxes and segmentation masks.

        Examples:
            >>> predictor = SegmentationPredictor(overrides=dict(model="yolo26n-seg.pt"))
            >>> results = predictor.postprocess(preds, img, orig_img)
        """
        # Extract protos - tuple if PyTorch model or array if exported
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        pointrend = self._extract_pointrend(preds)  # (point_head, batch_P3) or None; PyTorch-only
        return super().postprocess(preds[0], img, orig_imgs, protos=protos, pointrend=pointrend)

    def _point_head(self):
        """Return the Segment26 point head for the PyTorch backend, or None if absent/unavailable."""
        if getattr(self.model, "format", "pt") != "pt":
            return None
        try:  # AutoBackend -> backend.model (SegmentationModel) -> .model[-1] (head)
            head = self.model.model.model[-1]
        except (AttributeError, TypeError, IndexError):
            return None
        return getattr(head, "point_head", None)

    def _extract_pointrend(self, preds):
        """Extract (point_head, batch fine feats P3) when inference point-refine is enabled.

        Only the PyTorch backend passes the full model return through ``AutoBackend``, so ``preds[1]``
        is the head dict carrying ``feats``; exported backends return ``[det, proto]`` (no feats) and
        subdivision is auto-disabled. Returns None unless a point head and fine feats are available.
        """
        if not self.args.seg_point_refine_infer:
            return None
        if not (isinstance(preds, (list, tuple)) and len(preds) > 1 and isinstance(preds[1], dict)):
            return None
        point_head = self._point_head()
        if point_head is None:
            return None
        feats_dict = preds[1]
        if "one2one" in feats_dict:  # end2end: feats nested under one2one
            feats_dict = feats_dict["one2one"]
        feats = feats_dict.get("feats")
        if not feats:  # no neck features (e.g. older/non-seg path)
            return None
        return point_head, feats[0]  # P3 = finest neck feature, shape (B, C, Hf, Wf)

    def construct_results(self, preds, img, orig_imgs, protos, pointrend=None):
        """Construct a list of result objects from the predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.
            protos (torch.Tensor): Prototype masks tensor with shape (B, C, H, W).
            pointrend (tuple | None): (point_head, batch P3) for inference subdivision, else None.

        Returns:
            (list[Results]): List of result objects containing the original images, image paths, class names, bounding
                boxes, and masks.
        """
        results = []
        for i, (pred, orig_img, img_path, proto) in enumerate(zip(preds, orig_imgs, self.batch[0], protos)):
            per_image = None
            if pointrend is not None:
                point_head, batch_feats = pointrend
                per_image = (point_head, batch_feats[i : i + 1])  # (1, C, Hf, Wf) for this image
            results.append(self.construct_result(pred, img, orig_img, img_path, proto, per_image))
        return results

    def construct_result(self, pred, img, orig_img, img_path, proto, pointrend=None):
        """Construct a single result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.
            pointrend (tuple | None): (point_head, per-image P3) for subdivision, else None.

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        if pred.shape[0] == 0:  # save empty boxes
            masks = None
        elif pointrend is not None and not self.args.retina_masks:
            point_head, feats = pointrend
            masks = ops.process_mask_pointrend(  # NHW at letterboxed resolution
                proto,
                pred[:, 6:],
                pred[:, :4],
                img.shape[2:],
                point_head,
                feats,
                num_points=int(self.args.seg_point_num),
                oversample_ratio=int(self.args.seg_point_oversample),
                importance_ratio=float(self.args.seg_point_importance),
                subdivisions=int(self.args.seg_point_subdiv_k),
                roi_margin=max(float(self.args.seg_point_roi), 0.0),
            )
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # NHW
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # NHW
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
            if not (all(keep) or getattr(self, "_feats", None) is not None):  # skip filter if native ReID enabled
                pred, masks = pred[keep], masks[keep]  # indexing is slow
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
