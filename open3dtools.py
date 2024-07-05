import numpy as np
import open3d as o3d
import time
import os


def np2o3dpcd(nppcd: np.ndarray) -> o3d.geometry.PointCloud:
    '''
    Convert numpy array point cloud data to Open3D format

    Args:
        nppcd (numpy.ndarray): Point cloud data in 2 dimensional numpy array, 1st dimension indicates different points, 2nd dimension indicates x,y,z values.
    '''

    if nppcd.ndim != 2:
        raise ValueError("input should be 2 dim")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nppcd)

    return pcd


def o3dpcd2np(o3dpcd: o3d.geometry.PointCloud) -> np.ndarray:
    '''
    Convert Open3D point cloud to numpy array

    Args:
        o3dpcd (open3d.geometry.PointCloud): Open3D point cloud.
    '''

    if not isinstance(o3dpcd, o3d.geometry.PointCloud):
        raise ValueError("o3dpcd should be o3d.geometry.PointCloud")

    return np.asarray(o3dpcd.points)


def load_from_file(path: str) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(path)

def save_to_file(path: str, o3dpcd: o3d.geometry.PointCloud):
    o3d.io.write_point_cloud(path, o3dpcd)
    return


def uniform_grid_points(step=[0.05, 0.05, 0.05], range=np.array([[-1, 1], [-1, 1], [-1, 1]])) -> o3d.geometry.PointCloud:
    '''
    Output grid points in uniform

    Args:
        step : steps for x,y,z
        range : range of the domain
    '''

    x = np.arange(range[0, 0], range[0, 1], step[0])
    y = np.arange(range[1, 0], range[1, 1], step[1])
    z = np.arange(range[2, 0], range[2, 1], step[2])

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def rotation_animation(pcd: o3d.geometry.PointCloud, axis: str = "y", num_frames: int = 100, fps=24, angle_per_frame=2*np.pi/100, save_frame: bool = False, save_path: str = None, quit_after_play: bool = True):
    '''
    Play animation that a point cloud rotating along some axis (x, y, or z)

    Args:
        pcd (open3d.geometry.PointCloud): The point cloud
        axis (str): The axis that the point cloud rotate along with. Should be one of the value: "x", "y", "z".
        num_frames (int): Number of frames that needs to generate.
        fps : Frames per second. Note that the actual fps may be lower than this.
        angle_per_frame: The angle that the point cloud rotate in each frame, in radian measure.
        save_frame (bool): Indicate whether save the frames to disk or not. If True, save_path should be a valid path.
        save_path (str): The path that the frames should be saved to. It only works if save_frame == True.
        quit_after_play (bool): Close the window when finishing paying the animation.
    '''
    
    try:
        pcd_combined = o3d.geometry.PointCloud()
        for single_pcd in pcd:
            pcd_combined += single_pcd
    except:
        pcd_combined = pcd
    frame_time = 1.0/fps
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_combined)
    if save_frame == True:
        if save_path == None:
            raise ValueError("please indicate save_path")
        os.makedirs(save_path, exist_ok=True)

    for i in range(num_frames):
        start_time = time.time()
        match axis:
            case "x":
                rotation_matrix = pcd_combined.get_rotation_matrix_from_xyz(
                    (angle_per_frame, 0, 0))
            case "y":
                rotation_matrix = pcd_combined.get_rotation_matrix_from_xyz(
                    (0, angle_per_frame, 0))
            case "z":
                rotation_matrix = pcd_combined.get_rotation_matrix_from_xyz(
                    (0, 0, angle_per_frame))
            case _:
                raise ValueError(
                    "axis should be one of the value: \"x\", \"y\", or \"z\".")
        pcd_combined.rotate(rotation_matrix, center=(0, 0, 0))

        vis.update_geometry(pcd_combined)
        vis.poll_events()
        vis.update_renderer()
        if save_frame == True:
            frame_path = os.path.join(save_path, f"frame_{i:05d}.png")
            vis.capture_screen_image(frame_path)
        processing_time = time.time()-start_time
        time.sleep(max(frame_time-processing_time, 0))

    if quit_after_play:
        vis.destroy_window()

    return

