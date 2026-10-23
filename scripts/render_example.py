"""Run the actual ComfyUI sequence node without launching a web server."""
import importlib
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root.parent))
node = importlib.import_module(root.name + ".Paints-UNDO.ZZX_PaintsUndo")
source = Image.open(root / "Paints-UNDO/imgs/1.jpg").convert("RGB")
image = torch.from_numpy(np.array(source)).float().unsqueeze(0) / 255
frames, prompt = node.ZZX_PaintsUndoSequence().generate(
    image, "", "400,600,800,900,950,999", 12345,
    sampling_steps=30, resolution=640,
)
output = root / "outputs"
output.mkdir(exist_ok=True)
pictures = [Image.fromarray((frame.numpy() * 255).round().astype(np.uint8)) for frame in frames]
for i, picture in enumerate(pictures):
    picture.save(output / f"undo_{i:02d}.png")
pictures[0].save(output / "undo.gif", save_all=True, append_images=pictures[1:], duration=700, loop=0)
w, h = pictures[0].size
sheet = Image.new("RGB", (w * 4, (h + 32) * 2), "#ddd")
draw = ImageDraw.Draw(sheet)
for i, (picture, label) in enumerate(zip(pictures, ["Original", "400", "600", "800", "900", "950", "999", "Blank"])):
    x, y = i % 4 * w, i // 4 * (h + 32)
    sheet.paste(picture, (x, y))
    draw.text((x + 10, y + h + 8), label, fill="black")
sheet.save(output / "contact_sheet.jpg")
(output / "prompt.txt").write_text(prompt, encoding="utf-8")
print(f"PASS: {tuple(frames.shape)}, range={frames.min().item()}..{frames.max().item()}")
print(output)
