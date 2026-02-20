import torch
import numpy as np
import matplotlib.pyplot as plt

from HeatmapParser import HeatmapParser  

def test_heatmap_parser():
    B, C, H, W = 2, 3, 32, 32  # batch=2, channel=3
    fake_heatmap = torch.rand(B, C, H, W) 

    channel_configs = {
        "anchor": {"topk": 5, "max_keep": 2, "threshold": 0.3},
        "tip": {"topk": 5, "max_keep": 2, "threshold": 0.3},
        "contact": {"max_keep": 4, "threshold": 0.5}
    }
    parser = HeatmapParser(channel_configs)
    results = parser.parse(fake_heatmap)

    for name, batch_results in results.items():
        print(f"\nChannel: {name}")
        for b_idx, keypoints in enumerate(batch_results):
            print(f"  Batch {b_idx}:")
            if keypoints.shape[0] == 0:
                print("    No keypoints detected.")
            else:
                for i, (x, y, score) in enumerate(keypoints):
                    print(f"    kp{i}: x={x:.2f}, y={y:.2f}, score={score:.3f}")
    

    hm = fake_heatmap[0, 0].cpu().numpy()  
    plt.imshow(hm, cmap="hot")
    for (x, y, score) in results["anchor"][0]:
        plt.scatter(x, y, color="cyan", s=30, marker="x")
    plt.show()


if __name__ == "__main__":
    test_heatmap_parser()
