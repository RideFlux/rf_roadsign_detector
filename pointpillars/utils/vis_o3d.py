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

def draw_bev(points, gt_box, box=None, x_range=(-8, 0), y_range=(-48, 0)):
    h, w = 960, 160
    bev_img = np.zeros((160, 960, 3), dtype=np.uint8)  # BGR 이미지

    def xy_to_pixel(x, y):
        px = ((x - x_range[0]) / (x_range[1] - x_range[0]) * w).astype(np.int32)
        py = h - ((y - y_range[0]) / (y_range[1] - y_range[0]) * h).astype(np.int32)
        return px, py

    # 점 찍기
    x, y, r = points[:, 0], points[:, 1], points[:, 3]
    px, py = xy_to_pixel(x, y)
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    bev_img[px[valid], py[valid]] = (180, 180, 180)  # 흰 점
    
    # GT 박스 그리기
    if gt_box is not None:
        cx, cy = gt_box[:2]
        bl, bw = gt_box[3:5]

        # 네 꼭짓점 좌표
        half_w, half_l = bw / 2, bl / 2
        corners = np.array([
            [cx - half_l, cy - half_w],
            [cx - half_l, cy + half_w],
            [cx + half_l, cy + half_w],
            [cx + half_l, cy - half_w]
        ])
        pxs, pys = xy_to_pixel(corners[:, 0], corners[:, 1])
        polygon = np.stack([pys, pxs], axis=1).astype(np.int32).reshape(-1, 1, 2)

        # 폴리라인 그리기 (GT는 빨간색으로)
        bev_img = np.ascontiguousarray(bev_img)
        cv2.polylines(bev_img, [polygon], isClosed=True, color=(0, 0, 255), thickness=1)
            
    # 박스 그리기
    if box is not None:
        cx, cy = box[:2]
        bl, bw = box[3:5]

        # 네 꼭짓점 좌표
        half_w, half_l = bw / 2, bl / 2
        corners = np.array([
            [cx - half_l, cy - half_w],
            [cx - half_l, cy + half_w],
            [cx + half_l, cy + half_w],
            [cx + half_l, cy - half_w]
        ])
        pxs, pys = xy_to_pixel(corners[:, 0], corners[:, 1])
        polygon = np.stack([pys, pxs], axis=1).astype(np.int32).reshape(-1, 1, 2)

        # 폴리라인 그리기
        bev_img = np.ascontiguousarray(bev_img)
        cv2.polylines(bev_img, [polygon], isClosed=True, color=(0, 255, 255), thickness=1)

    return bev_img

def vis_pc(pc, bbox=None, label=None):
    '''
    pc: ply or np.ndarray (N, 4)
    bboxes: np.ndarray, (n, 7) or (n, 8, 3)
    labels: (n, )
    scores: (n, )
    '''
    print (bbox, label)
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