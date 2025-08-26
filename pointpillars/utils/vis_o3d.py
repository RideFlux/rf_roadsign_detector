import cv2
import numpy as np
import open3d as o3d
import os
from pointpillars.utils import bbox3d2corners


COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
COLORS_IMG = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]

LINES = [
        [0, 1],
        [1, 2], 
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [2, 6],
        [7, 3],
        [1, 5],
        [4, 0]
    ]


def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    density = npy[:, 3]
    colors = [[x/255, x/255, x/255] for x in density]
    ply.colors = o3d.utility.Vector3dVector(colors)
    return ply

def ply2npy(ply):
    return np.array(ply.points)


def bbox_obj(points, color=[1, 0, 0]):
    colors = [color for _ in range(len(LINES))]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points)),
        lines=o3d.utility.Vector2iVector(LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def vis_core(plys):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = [0, 0, 0]
    
    # Set point size smaller
    vis.get_render_option().point_size = 0.5  # Smaller points for better visibility

    PAR = os.path.dirname(os.path.abspath(__file__))
    ctr = vis.get_view_control()
    
    # Check if viewpoint.json exists
    viewpoint_path = os.path.join(PAR, 'viewpoint.json')
    if os.path.exists(viewpoint_path):
        try:
            param = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
            
            # Set the window size to match the saved parameters
            intrinsic = param.intrinsic
            vis.create_window(width=intrinsic.width, height=intrinsic.height)
            
            for ply in plys:
                vis.add_geometry(ply)
            ctr.convert_from_pinhole_camera_parameters(param)
        except Exception as e:
            print(f"Failed to load viewpoint: {e}")
            # If reading viewpoint fails, just add geometries
            for ply in plys:
                vis.add_geometry(ply)
    else:
        for ply in plys:
            vis.add_geometry(ply)
    
    print("3D visualization opened. Press 'q' to quit, 'ESC' to close, or close the window.")
    
    # Use poll_events and update_renderer instead of run() to avoid blocking
    vis.poll_events()
    vis.update_renderer()
    
    # Run the visualization loop
    try:
        vis.run()
    except KeyboardInterrupt:
        print("Visualization interrupted by user.")
    finally:
        vis.destroy_window()

def draw_bev(points, gt_box, box=None, gt_label=None, pred_label=None, mode=3, x_range=(-8, 0), y_range=(-48, 0), z_range=(-3, 1)):

    # FV, BEV 크기
    w1, h1, w2, h2 = 640, 320, 1920, 320
    bev_img = np.zeros((460, 40 + (w1 + 40) * (mode & 0b01 != 0) + (w2 + 40) * (mode & 0b10 != 0), 3), dtype=np.uint8)  # BGR 이미지

    def xy_to_pixel(x, y, z):
        # px1, py1: FV 상에서의 위치, px2, py2: BEV 상에서의 위치
        px1 = np.clip(w1 - ((x - x_range[0]) / (x_range[1] - x_range[0]) * w1).astype(np.int32), 0, w1) + 40
        py1 = np.clip(h1 - ((z - z_range[0]) / (z_range[1] - z_range[0]) * h1).astype(np.int32), 0, h1) + 100
        px2 = np.clip(w2 - ((y - y_range[0]) / (y_range[1] - y_range[0]) * w2).astype(np.int32), 0, w2) + 40 + (w1 + 40) * (mode & 0b01 != 0)
        py2 = np.clip(h2 - ((x - x_range[0]) / (x_range[1] - x_range[0]) * h2).astype(np.int32), 0, h2) + 100
        return px1, py1, px2, py2

    # 점 찍기
    x, y, z, r = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    px1, py1, px2, py2 = xy_to_pixel(x, y, z)
    valid = (x >= x_range[0]) & (x <= x_range[1]) \
            & (y >= y_range[0]) & (y <= y_range[1]) \
            & (z >= z_range[0]) & (z <= z_range[1])
    
    # Add bounds checking for pixel coordinates
    valid1 = valid & (px1 >= 40) & (px1 < 40 + w1) & (py1 >= 100) & (py1 < bev_img.shape[0] - 40) & (mode & 0b01 != 0)
    valid2 = valid & (px2 >= 40 + (w1 + 40) * (mode & 0b01 != 0)) & (px2 < bev_img.shape[1] - 40) & (py2 >= 100) & (py2 < bev_img.shape[0]-40) & (mode & 0b10 != 0)
    r = np.minimum(r * 20, 255)
    r_valid1 = r[valid1]
    r_valid2 = r[valid2]
    bev_img[py1[valid1], px1[valid1]] = np.stack([r_valid1, r_valid1, r_valid1], axis=-1)
    bev_img[py2[valid2], px2[valid2]] = np.stack([r_valid2, r_valid2, r_valid2], axis=-1)

    if gt_box is not None:
        cx, cy, cz = gt_box[:3]
        bl, bw, bh = gt_box[3:6]

        half_w, half_l = bw / 2, bl / 2
        corners1 = np.array([
            [cx - half_l, cy - half_w, cz],
            [cx + half_l, cy - half_w, cz],
            [cx + half_l, cy - half_w, cz + bh],
            [cx - half_l, cy - half_w, cz + bh]
        ])
        corners2 = np.array([
            [cx - half_l, cy - half_w, cz],
            [cx - half_l, cy + half_w, cz],
            [cx + half_l, cy + half_w, cz],
            [cx + half_l, cy - half_w, cz],
        ])
        pxs1, pys1, _, _= xy_to_pixel(corners1[:, 0], corners1[:, 1], corners1[:, 2])
        polygon1 = np.stack([pxs1, pys1], axis=1).astype(np.int32).reshape(-1, 1, 2)
        _, _, pxs2, pys2= xy_to_pixel(corners2[:, 0], corners2[:, 1], corners2[:, 2])
        polygon2 = np.stack([pxs2, pys2], axis=1).astype(np.int32).reshape(-1, 1, 2)

        bev_img = np.ascontiguousarray(bev_img)
        if mode & 0b01 != 0:
            cv2.polylines(bev_img, [polygon1], isClosed=True, color=(0, 0, 255), thickness=2)
        if mode & 0b10 != 0:
            cv2.polylines(bev_img, [polygon2], isClosed=True, color=(0, 0, 255), thickness=2)

    if box is not None:
        cx, cy, cz = box[:3]
        bl, bw, bh = box[3:6]

        half_w, half_l = bw / 2, bl / 2
        corners1 = np.array([
            [cx - half_l, cy - half_w, cz],
            [cx + half_l, cy - half_w, cz],
            [cx + half_l, cy - half_w, cz + bh],
            [cx - half_l, cy - half_w, cz + bh]
        ])
        corners2 = np.array([
            [cx - half_l, cy - half_w, cz],
            [cx - half_l, cy + half_w, cz],
            [cx + half_l, cy + half_w, cz],
            [cx + half_l, cy - half_w, cz],
        ])
        pxs1, pys1, _, _= xy_to_pixel(corners1[:, 0], corners1[:, 1], corners1[:, 2])
        polygon1 = np.stack([pxs1, pys1], axis=1).astype(np.int32).reshape(-1, 1, 2)
        _, _, pxs2, pys2= xy_to_pixel(corners2[:, 0], corners2[:, 1], corners2[:, 2])
        polygon2 = np.stack([pxs2, pys2], axis=1).astype(np.int32).reshape(-1, 1, 2)

        bev_img = np.ascontiguousarray(bev_img)
        if mode & 0b01 != 0:
            cv2.polylines(bev_img, [polygon1], isClosed=True, color=(255, 255, 0), thickness=2)
        if mode & 0b10 != 0:
            cv2.polylines(bev_img, [polygon2], isClosed=True, color=(255, 255, 0), thickness=2)

    if mode & 0b01 != 0:
        # cv2.rectangle(bev_img, (40, 100), (680, 420), (0, 255, 255), 1)
        # cv2.rectangle(bev_img, (20, 400), (60, 440), (0, 255, 255), -1)
        # cv2.arrowedLine(bev_img, (40, 436), (40, 404), (0, 0, 0), 2, tipLength=0.3)
        
        cv2.rectangle(bev_img, (40, 100), (40 + w1, 100 + h1), (0, 255, 255), 1)
        cv2.rectangle(bev_img, (20, 80 + h1), (60, 120 + h1), (0, 255, 255), -1)
        cv2.arrowedLine(bev_img, (40, 116 + h1), (40, 84 + h1), (0, 0, 0), 2, tipLength=0.3)
        
    if mode & 0b10 != 0:
        cv2.rectangle(bev_img, (40 + (w1 + 40) * (mode & 0b01 != 0), 100), \
                                (40 + (w1 + 40) * (mode & 0b01 != 0) + w2, 100 + h2), \
                                (0, 255, 255), 1)
        cv2.rectangle(bev_img, (20 + (w1 + 40) * (mode & 0b01 != 0), 80), \
                                (60 + (w1 + 40) * (mode & 0b01 != 0), 120), \
                                (0, 255, 255), -1)
        cv2.arrowedLine(bev_img, (24 + (w1 + 40) * (mode & 0b01 != 0), 100), (56 + (w1 + 40) * (mode & 0b01 != 0), 100), (0, 0, 0), 2, tipLength=0.3)
    
    
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.8
    fontColor = (255,255,255)
    thickness = 1
    text_width, text_height = cv2.getTextSize(f'GT Class: {gt_label+1}, Pred Class: {pred_label+1}', font, fontScale, thickness)[0]
    mid = (40 + (w1 + 40) * (mode & 0b01 != 0) + (w2 + 40) * (mode & 0b10 != 0)) // 2
    cv2.putText(bev_img, f'GT Class: {gt_label+1}, Pred Class: {pred_label+1}', (mid-text_width//2, 50+text_height), font, fontScale, fontColor, thickness, cv2.LINE_AA)

    return bev_img

def vis_pc(pc, bbox=None, label=None):
    '''
    pc: ply or np.ndarray (N, 4)
    bboxes: np.ndarray, (n, 7) or (n, 8, 3)
    labels: (n, )
    scores: (n, )
    '''
    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)
    
    # Create coordinate frame (origin)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0])

    if bbox is None:
        vis_core([pc, mesh_frame])
        return

    bbox = bbox3d2corners(bbox)

    vis_objs = [pc, mesh_frame]
    if label is None:
        color = [1, 1, 0]
    else:
        if label >= 0 and label < 9:
            color = COLORS[label]
        else:
            color = COLORS[-1]
    vis_objs.append(bbox_obj(bbox, color=color))
    vis_core(vis_objs)


def vis_img_3d(img, image_points, labels, rt=True):
    '''
    img: (h, w, 3)
    image_points: (n, 8, 2)
    labels: (n, )
    '''

    for i in range(len(image_points)):
        label = labels[i]
        bbox_points = image_points[i] # (8, 2)
        if label >= 0 and label < 3:
            color = COLORS_IMG[label]
        else:
            color = COLORS_IMG[-1]
        for line_id in LINES:
            x1, y1 = bbox_points[line_id[0]]
            x2, y2 = bbox_points[line_id[1]]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
    if rt:
        return img
    cv2.imshow('bbox', img)
    cv2.waitKey(0)