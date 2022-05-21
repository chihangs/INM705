# INM705
Object detection by instance segmentation for MS COCO dataset - Focus on detection of birds and airplanes

The code was prepared for server that does not have direct access to Internet

Need to download weight files below manually (too big to upload to github, which limits to 25MB per file)

fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth

Because of the server environment, we also pre-download engine.py, utlis.py, transform.py (and they depend on coco_eval.py, coco_utlis.py, so they also need downloading) from https://github.com/pytorch/vision.git

