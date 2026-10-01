from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort


def _hwc(image_chw: np.ndarray) -> np.ndarray:
    a=np.asarray(image_chw)
    if a.ndim!=3 or a.shape[0]!=3 or a.dtype!=np.uint8:
        raise ValueError(f"Expected uint8 CHW RGB, got {a.shape} {a.dtype}")
    return np.transpose(a,(1,2,0))


def _chw(image_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(image_hwc,dtype=np.uint8),(2,0,1))


def _letterbox(rgb: np.ndarray, size: int=512):
    h,w=rgb.shape[:2]
    scale=min(size/w,size/h)
    nw=max(1,min(size,int(round(w*scale))))
    nh=max(1,min(size,int(round(h*scale))))
    resized=np.asarray(
        Image.fromarray(rgb,"RGB").resize((nw,nh),Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    left=(size-nw)//2
    top=(size-nh)//2
    right=size-nw-left
    bottom=size-nh-top
    canvas=np.pad(
        resized,
        ((top,bottom),(left,right),(0,0)),
        mode="edge",
    )
    return canvas,(left,top,nw,nh)


def _letterbox_mask(mask: np.ndarray, geom, size: int=512) -> np.ndarray:
    left,top,nw,nh=geom
    resized=np.asarray(
        Image.fromarray(mask.astype(np.uint8)*255,"L").resize(
            (nw,nh),Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )>=128
    canvas=np.zeros((size,size),dtype=bool)
    canvas[top:top+nh,left:left+nw]=resized
    return canvas


def _unletterbox(rgb: np.ndarray, geom, out_w: int, out_h: int) -> np.ndarray:
    left,top,nw,nh=geom
    crop=rgb[top:top+nh,left:left+nw]
    return np.asarray(
        Image.fromarray(crop.astype(np.uint8),"RGB").resize(
            (out_w,out_h),Image.Resampling.LANCZOS
        ),
        dtype=np.uint8,
    )


class LamaInpainter:
    def __init__(self, model_path: str | Path) -> None:
        self.session=ort.InferenceSession(
            str(Path(model_path)),
            providers=["CPUExecutionProvider"],
        )
        inputs=self.session.get_inputs()
        if len(inputs)!=2:
            raise ValueError(f"Expected 2 inputs, got {len(inputs)}")
        image_inputs=[x for x in inputs if len(x.shape)==4 and x.shape[1]==3]
        mask_inputs=[x for x in inputs if len(x.shape)==4 and x.shape[1]==1]
        if len(image_inputs)!=1 or len(mask_inputs)!=1:
            raise ValueError([(x.name,x.shape) for x in inputs])
        self.image_name=image_inputs[0].name
        self.mask_name=mask_inputs[0].name

    def inpaint(self, image_chw: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rgb=_hwc(image_chw)
        h,w=rgb.shape[:2]
        if mask.shape!=(h,w):
            raise ValueError("Mask/image size mismatch")
        if not np.any(mask):
            return image_chw.copy()

        square,geom=_letterbox(rgb,512)
        mask_square=_letterbox_mask(mask,geom,512)

        image_input=np.transpose(square.astype(np.float32)/255.0,(2,0,1))[None]
        mask_input=mask_square.astype(np.float32)[None,None]

        raw=self.session.run(
            None,
            {self.image_name:image_input,self.mask_name:mask_input},
        )[0]
        out=np.asarray(raw)
        if out.ndim==4:
            out=out[0]
        if out.shape[0]==3:
            out=np.transpose(out,(1,2,0))
        # Known LaMa ONNX exports return 0..255. Be defensive if a 0..1 export is used.
        if float(np.nanmax(out))<=1.5:
            out=out*255.0
        out=np.clip(np.rint(out),0,255).astype(np.uint8)
        fill=_unletterbox(out,geom,w,h)

        # Hard scientific invariant: only masked pixels may change.
        composite=rgb.copy()
        composite[mask]=fill[mask]
        return _chw(composite)


def outside_mask_equal(
    original_chw: np.ndarray,
    inpainted_chw: np.ndarray,
    mask: np.ndarray,
) -> bool:
    a=_hwc(original_chw)
    b=_hwc(inpainted_chw)
    return bool(np.array_equal(a[~mask],b[~mask]))
