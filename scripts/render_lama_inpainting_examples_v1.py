from __future__ import annotations

import argparse
import base64
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from src.scratch_v2 import preprocess_to_uint8, read_csv
from src.hand_mask_v3 import GEOMETRY_CANDIDATES, MultiViewMediaPipeHandMasker, geometry_mask
from src.lama_inpainting_v1 import LamaInpainter, outside_mask_equal

EXAMPLES = [
    "testing/flip/0048_000000011",
    "testing/notflip/0040_000000004",
    "testing/flip/0008_000000029",
    "testing/flip/0059_000000028",
    "training/flip/0018_000000021",
]

def save_png(chw: np.ndarray, path: Path) -> None:
    arr=np.transpose(chw,(1,2,0)).astype(np.uint8)
    Image.fromarray(arr,"RGB").save(path,optimize=True)
    path.with_suffix(path.suffix+".b64.txt").write_text(
        base64.b64encode(path.read_bytes()).decode("ascii")+"\n",
        encoding="ascii",
    )

def mask_overlay(chw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb=np.transpose(chw,(1,2,0)).astype(np.float32)
    out=rgb.copy()
    out[mask]=0.35*rgb[mask]+0.65*np.array([255,0,0],dtype=np.float32)
    return np.transpose(np.clip(np.rint(out),0,255).astype(np.uint8),(2,0,1))

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--archive",required=True,type=Path)
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--hand-model",required=True,type=Path)
    p.add_argument("--lama-model",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    args=p.parse_args()

    manifest={r["sample_id"]:r for r in read_csv(args.manifest)}
    cfg={c.name:c for c in GEOMETRY_CANDIDATES}["mv35_handarm"]
    args.output_dir.mkdir(parents=True,exist_ok=True)

    inpainter=LamaInpainter(args.lama_model)
    with MultiViewMediaPipeHandMasker(args.hand_model,0.35) as masker, ZipFile(args.archive) as z:
        for sample_id in EXAMPLES:
            src=manifest[sample_id]
            with z.open(src["archive_member"]) as fh:
                with Image.open(fh) as im:
                    source_rgb=np.asarray(im.convert("RGB"),dtype=np.uint8)
                    original=preprocess_to_uint8(im)
            _,h,w=original.shape
            detections,_=masker.detections(source_rgb,h,w)
            mask=geometry_mask(detections,h,w,cfg)
            inpainted=inpainter.inpaint(original,mask)
            assert outside_mask_equal(original,inpainted,mask)

            stem=sample_id.replace("/","__")
            save_png(original,args.output_dir/f"{stem}__original.png")
            save_png(mask_overlay(original,mask),args.output_dir/f"{stem}__mask.png")
            save_png(inpainted,args.output_dir/f"{stem}__lama_inpainted.png")

    return 0

if __name__=="__main__":
    raise SystemExit(main())
