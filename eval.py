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
        print("\n평가 모드를 선택해 주세요")
        print("1. 전방 view & Bird's eye view로 시각화하기")
        print("2. 3D로 시각화하기")
        print("3. 정량 평가만 진행하기\n")
        eval_mode = int(input("1/2/3 중 하나 입력: "))
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
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location=map_location)
        loaded_state_dict = checkpoint['state_dict']

        model.load_state_dict(loaded_state_dict, strict=False)
        print("Checkpoint loaded successfully.")

        inference_test_imgs_qtt(model, cfg, eval_mode)

if __name__ == '__main__':
    main()