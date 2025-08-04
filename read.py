import os
import glob
import numpy as np
import json

# Define court corners in world coordinates (X, Y, Z, 1)
# (adjust these to your true court dimensions/coordinates)
world_corners = [
    [0, 0, 0, 1],        # Bottom left
    [0, 28, 0, 1],       # Top left
    [15, 0, 0, 1],       # Bottom right
    [15, 28, 0, 1],      # Top right
]
corner_names = ["BL", "TL", "BR", "TR"]

# Load image size metadata from config/camera_config
def get_image_size(cam_index):
    meta_path = f"data/camera_config/cam_{cam_index}/metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if "imsize" in meta:
            return tuple(meta["imsize"])
    # Default fallback
    return (3840, 2160)

def main():
    proj_files = sorted(glob.glob("output/proj_out*.npy"))
    if not proj_files:
        print("No projection matrices found in output/.")
        return

    for proj_path in proj_files:
        projmat = np.load(proj_path)
        cam_index = os.path.splitext(os.path.basename(proj_path))[0].split("out")[-1]
        im_w, im_h = get_image_size(cam_index)
        print(f"\nLoaded: {os.path.basename(proj_path)}")
        print(f"Shape: {projmat.shape}")
        for idx, corner in enumerate(world_corners):
            corner_np = np.array(corner)
            img_pt = projmat @ corner_np
            img_pt = img_pt / img_pt[2]
            u, v = img_pt[:2]
            status = ""
            if not (0 <= u < im_w) or not (0 <= v < im_h):
                status = "  [OUT OF BOUNDS]"
            print(f"  {corner_names[idx]} world {corner[:3]} --> pixel (u,v): {u:.1f}, {v:.1f}{status}")

if __name__ == "__main__":
    main()