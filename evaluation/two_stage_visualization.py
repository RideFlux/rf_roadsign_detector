"""Front-view and BEV inspection of the two-stage model's final predictions."""

from pathlib import Path

import cv2
import numpy as np
import torch


def select_eval_mode(mode):
    if mode not in (0, 1, 2):
        raise ValueError('eval_mode must be 0 (menu), 1 (front + BEV), or 2 (quantitative)')
    if mode:
        return mode
    print('\n평가 모드를 선택해 주세요')
    print('1. 전방 + BEV로 시각화하기')
    print('2. 정량 평가만 진행하기\n')
    while True:
        try:
            answer = input('1 또는 2 입력: ').strip()
        except EOFError as error:
            raise ValueError('모드 입력이 없습니다. 실행 옵션으로 eval_mode=1 또는 eval_mode=2를 지정하세요.') from error
        if answer in ('1', '2'):
            return int(answer)
        print('1 또는 2를 입력해 주세요.')


def _class_name(label, no_object_class=8):
    return 'No object' if label == no_object_class else f'Class_{label + 1} (id={label})'


def render_two_stage_sample(points, record, detection_range, no_object_class=8):
    # Import Open3D-dependent legacy drawing only when visualization is requested.
    from pointpillars.utils.vis_o3d import draw_bev

    points = torch.as_tensor(points).detach().cpu().numpy()
    gt_box = np.asarray(record['gt_bbox'], dtype=np.float32) if record['gt_bbox'] is not None else None
    pred_box = np.asarray(record['pred_bbox'], dtype=np.float32) if record['pred_bbox'] is not None else None
    image = draw_bev(
        points, gt_box, pred_box, record['gt_class'], record['pred_class'], mode=3,
        x_range=(detection_range[0], detection_range[3]),
        y_range=(detection_range[1], detection_range[4]),
        z_range=(detection_range[2], detection_range[5]),
        pred_score=record['regression_score'],
    )
    # Replace the legacy header with the final 9-class result and both scores.
    image[:100] = 0
    gt_text = _class_name(record['gt_class'], no_object_class)
    if record['gt_bbox'] is not None and not record['gt_in_range']:
        gt_text += f" (out of range; raw id={record['raw_gt_class']})"
    reg_score = record['regression_score']
    cls_score = record['classifier_score']
    score_text = 'none' if reg_score is None else f'{reg_score:.3f}'
    cls_text = 'skipped' if cls_score is None else f'{cls_score:.3f}'
    status = 'accepted' if record['detected'] else 'rejected'
    distance = record['center_xy_error_m']
    distance_text = 'n/a' if distance is None else f'{distance:.3f} m'
    lines = [
        (f"GT: {gt_text} | Final: {_class_name(record['pred_class'], no_object_class)}", (255, 255, 255)),
        (f'Regression score: {score_text} ({status}) | Classifier score: {cls_text} | XY error: {distance_text}', (255, 255, 255)),
        ('Red: GT box | Cyan: regression candidate (also shown when rejected) | Left: Front | Right: BEV', (255, 255, 255)),
    ]
    for y, (text, color) in zip((24, 50, 76), lines):
        cv2.putText(image, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
    return image


class InteractiveVisualizer:
    def __init__(self, image_path, detection_range, no_object_class=8):
        self.image_path = Path(image_path)
        self.detection_range = detection_range
        self.no_object_class = no_object_class
        self.skip_remaining = 0

    def __call__(self, points, record):
        if self.skip_remaining:
            self.skip_remaining -= 1
            return True
        image = render_two_stage_sample(points, record, self.detection_range, self.no_object_class)
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.image_path), image):
            raise OSError(f'Could not save visualization: {self.image_path}')
        from evaluation.utils import create_hyperlink
        print(f"\nPCD: {record['pcd_path']}")
        print(f"GT: {_class_name(record['gt_class'], self.no_object_class)}, "
              f"Final: {_class_name(record['pred_class'], self.no_object_class)}")
        create_hyperlink(str(self.image_path), 'Front + BEV image')
        print(f'Image: {self.image_path.resolve()}')
        while True:
            try:
                answer = input('Enter: 다음 | 숫자 N: 다음 N개 시각화 생략 | c/q: 종료 후 결과 저장: ').strip().lower()
            except EOFError:
                answer = 'q'
            if answer in ('c', 'q'):
                return False
            if not answer:
                return True
            if answer.isdecimal():
                self.skip_remaining = int(answer)
                return True
            print('Enter, 0 이상의 정수, c 또는 q를 입력해 주세요.')
