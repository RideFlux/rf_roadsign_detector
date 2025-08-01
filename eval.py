import torch
import hydra
from omegaconf import DictConfig
from evaluation.predict import  inference_test_imgs_qtt

@hydra.main(version_base=None, config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    ckpt_path = cfg.eval_checkpoint

    num_classes = cfg.data.datasets.num_classes
    
    eval_mode = cfg.eval_mode
    if eval_mode == 0:
        print("\nChoose options")
        print("1. Visualize in bird's eye view")
        print("2. Visualize in 3D point cloud")
        print("3. Only quantitative evaluation mode\n")
        eval_mode = int(input("Select (1/2/3): "))
    print("")
    
    if eval_mode in [1, 2, 3]:
        # Initialize model
        model = hydra.utils.instantiate(
            cfg.model.roadsign_detector.module,
            training_model=False,
            exporting_onnx=False,
            num_classes=num_classes
        )
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, weights_only=False)
        loaded_state_dict = checkpoint['state_dict']

        model.load_state_dict(loaded_state_dict, strict=False)
        print("Checkpoint loaded successfully.")

        inference_test_imgs_qtt(model, cfg, eval_mode)

if __name__ == '__main__':
    main()