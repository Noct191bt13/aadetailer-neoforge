# AADetailer Forge Neo Python 3.13
A small fork with new features and modifications, verified to work with up-to-date Forge Neo Python 3.13 only. (It can potentially work with other webUIs, but i am not promising to maintain them for anything other than Forge Neo Python 3.13)  

Fix for newer Ultralytics included.

Current additional features:  
- Automatically include loras
  If loras are present in prompt, they will be automatically added. If their name include main trigger, it can be included too. Schema for trigger in name: <lora:lora name (trigger) blah blah:1> - basically what is inside () is considered trigger.
- Autotag before inpaint (Autotags crop area, so inpaint is stable, and doesn't require re-prompting each gen)
- Reworked resolution, now based on scaling. (Define a scale(multiplier) for resolution over base, so it's always bigger than original)

Potential future features:
- Class-based detection support for YOLOs.
  
# ADetailer

ADetailer is an extension for the stable diffusion webui that does automatic masking and inpainting. It is similar to the Detection Detailer.

## Install

You can install it directly from the Extensions tab.

download this repo and put it in your expensions(replace original adetailer)

## Options

| Model, Prompts                    |                                                                                    |                                                                                                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ADetailer model                   | Determine what to detect.                                                          | `None` = disable                                                                                                                                       |
| ADetailer model classes           | Comma separated class names to detect. only available when using YOLO World models | If blank, use default values.<br/>default = [COCO 80 classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) |
| ADetailer prompt, negative prompt | Prompts and negative prompts to apply                                              | If left blank, it will use the same as the input.                                                                                                      |
| Skip img2img                      | Skip img2img. In practice, this works by changing the step count of img2img to 1.  | img2img only                                                                                                                                           |
| Apply only on hires.fix           | Skips lowres images, saving a lot of time. Applies only on hires pass.             |                                                                                                                                                        |
| Append main prompt LoRAs          | Append loras to adetailer prompt automatically.                                    | Also picks up [loractl](https://github.com/cheald/loractl)-style scheduled tokens, e.g. `<lora:name:0@0; 0.5@8:hr=0.7:ad=0.5>`. With the loractl fork installed, `ad=` loads the lora at that constant weight during ADetailer passes. |                                                                                                                                                        |
| Append LoRA triggers              | Also add triggers, if any are present in lora name.                                | Loras must follow this naming convention for trigger to work: <lora:lora name (trigger) blah blah:1>                                                   |

| Autotagging                            |                                                                                              | 
| ------------------------------------ | -------------------------------------------------------------------------------------------- |
| Enable Autotagging | Extend prompt by adding tags detected by WDv3 large tagger.| 

| Detection                            |                                                                                              |              |
| ------------------------------------ | -------------------------------------------------------------------------------------------- | ------------ |
| Detection model confidence threshold | Only objects with a detection model confidence above this threshold are used for inpainting. |              |
| Mask min/max ratio                   | Only use masks whose area is between those ratios for the area of the entire image.          |              |
| Mask only the top k largest          | Only use the k objects with the largest area of the bbox.                                    | 0 to disable |

If you want to exclude objects in the background, try setting the min ratio to around `0.01`.

| Mask Preprocessing              |                                                                                                                                     |                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Mask x, y offset                | Moves the mask horizontally and vertically by                                                                                       |                                                                                         |
| Mask erosion (-) / dilation (+) | Enlarge or reduce the detected mask.                                                                                                | [opencv example](https://docs.opencv.org/4.7.0/db/df6/tutorial_erosion_dilatation.html) |
| Mask merge mode                 | `None`: Inpaint each mask<br/>`Merge`: Merge all masks and inpaint<br/>`Merge and Invert`: Merge all masks and Invert, then inpaint |                                                                                         |

Applied in this order: x, y offset → erosion/dilation → merge/invert.

#### Inpainting

Each option corresponds to a corresponding option on the inpaint tab. Therefore, please refer to the inpaint tab for usage details on how to use each option.

## ControlNet Inpainting

You can use the ControlNet extension if you have ControlNet installed and ControlNet models.

Support `inpaint, scribble, lineart, openpose, tile, depth` controlnet models. Once you choose a model, the preprocessor is set automatically. It works separately from the model set by the Controlnet extension.

If you select `Passthrough`, the controlnet settings you set outside of ADetailer will be used.

## Advanced Options

API request example: [wiki/REST-API](https://github.com/Bing-su/adetailer/wiki/REST-API)

`[SEP], [SKIP], [PROMPT]` tokens: [wiki/Advanced](https://github.com/Bing-su/adetailer/wiki/Advanced)

## Media

- 🎥 [どこよりも詳しい After Detailer (adetailer)の使い方 ① 【Stable Diffusion】](https://youtu.be/sF3POwPUWCE)
- 🎥 [どこよりも詳しい After Detailer (adetailer)の使い方 ② 【Stable Diffusion】](https://youtu.be/urNISRdbIEg)

- 📜 [ADetailer Installation and 5 Usage Methods](https://kindanai.com/en/manual-adetailer/)

## Model

| Model                 | Target                | mAP 50                        | mAP 50-95                     |
| --------------------- | --------------------- | ----------------------------- | ----------------------------- |
| face_yolov8n.pt       | 2D / realistic face   | 0.660                         | 0.366                         |
| face_yolov8s.pt       | 2D / realistic face   | 0.713                         | 0.404                         |
| hand_yolov8n.pt       | 2D / realistic hand   | 0.767                         | 0.505                         |
| person_yolov8n-seg.pt | 2D / realistic person | 0.782 (bbox)<br/>0.761 (mask) | 0.555 (bbox)<br/>0.460 (mask) |
| person_yolov8s-seg.pt | 2D / realistic person | 0.824 (bbox)<br/>0.809 (mask) | 0.605 (bbox)<br/>0.508 (mask) |
| mediapipe_face_full   | realistic face        | -                             | -                             |
| mediapipe_face_short  | realistic face        | -                             | -                             |
| mediapipe_face_mesh   | realistic face        | -                             | -                             |

The YOLO models can be found on huggingface [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) and [Anzhc/Anzhcs_YOLOs](https://huggingface.co/Anzhc/Anzhcs_YOLOs)

For a detailed description of the YOLO8 model, see: https://docs.ultralytics.com/models/yolov8/#overview

YOLO World model: https://docs.ultralytics.com/models/yolo-world/

### Additional Model

Put your [ultralytics](https://github.com/ultralytics/ultralytics) yolo model in `models/adetailer`. The model name should end with `.pt`.

It must be a bbox detection or segment model and use all label.

### SAM2 mask refinement

In addition to the classic YOLO/mediapipe detectors, you can refine the
inpaint masks with [SAM2](https://github.com/facebookresearch/sam2) (Segment
Anything 2), using each detected box as a box prompt. This produces object-
shaped masks instead of rectangles, so inpainting stays inside the subject.
It works with every detector: bbox-only YOLOs (face, hand, person) and
mediapipe get real masks, and segmentation YOLOs get tighter masks.

| Option | Description |
| --- | --- |
| ADetailer SAM2 model | `None` = keep the classic detector masks. Pick a SAM2 model to refine. |
| ADetailer SAM2 keep loaded | Keep the SAM2 model in VRAM between generations (default on). Uncheck to offload after every use. |

All SAM2 controls live in the "SAM2 refinement" foldout of each unit:

| Option | Description |
| --- | --- |
| Box expansion | Grow each detection box by N pixels before sending it to SAM2 (more context around the object). |
| Mask threshold | Binarize SAM2 output masks at this confidence. 0 keeps any nonzero mask. |
| Dilation (-erode / +dilate) | Enlarge (positive) or shrink (negative) the refined SAM2 mask. |
| Mask feather | Gaussian-blur the refined mask edges (soft edges). |
| Use detection mask as hint | Pass the detector mask to SAM2 as an extra mask prompt alongside the box. |
| Mask hint threshold | Binarize the detector mask at this level before using it as a hint. |
| Mask hint negative point | Add a negative point prompt sampled from the hint background inside the box. |

Models are downloaded on first use into `models/sam2` in your webui root:

- `sam2_hiera_tiny.pt` / `sam2_hiera_small.pt` / `sam2_hiera_base_plus.pt` / `sam2_hiera_large.pt`
- `sam2.1_hiera_tiny.pt` / `sam2.1_hiera_small.pt` / `sam2.1_hiera_base_plus.pt` / `sam2.1_hiera_large.pt`

Smaller models (tiny/small) are fast and light; `large` gives the most
precise masks but needs more VRAM. The classic pipeline is unchanged when no
SAM2 model is selected, and masks that SAM2 cannot segment fall back to the
detector masks automatically.

**Custom checkpoints:** drop any fine-tuned SAM2 `.pt` file into
`models/sam2` and it appears in the model dropdown (Reload UI if it was added
while the webui was running). The architecture is guessed from the filename:
name it e.g. `sam2_hiera_large_<yours>.pt` or `sam2.1_hiera_tiny_<yours>.pt`.
Unrecognized names fall back to the sam2_hiera_large architecture with a
console warning.

Note: SAM2 runs synchronously during generation, so expect a few seconds of
added latency per image (more on CPU or with larger models). With "keep
loaded" on, the model stays in VRAM for the whole session; only one SAM2
model is kept resident at a time.

## How it works

ADetailer works in three simple steps.

1. Create an image.
2. Detect object with a detection model and create a mask image.
3. Inpaint using the image from 1 and the mask from 2.

## Settings

`Settings -> ADetailer` (the same section as `ad_save_previews`):

| Option | Description |
| --- | --- |
| SAM2 mask preview mode | When SAM2 mask refinement runs, append `[image+box | detector mask | SAM2 mask]` strips to the result gallery (`Gallery`) and/or save them (`Save files`, to `outputs/adetailer-sam-masks/`). |

## Development

AADetailer is developed and tested using the SDXL model, for the latest version of [ReForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) repository only.

## License

ADetailer is a derivative work that uses two AGPL-licensed works (stable-diffusion-webui, ultralytics) and is therefore distributed under the AGPL license.

## See Also

- https://github.com/ototadana/sd-face-editor
- https://github.com/continue-revolution/sd-webui-segment-anything
- https://github.com/portu-sim/sd-webui-bmab
