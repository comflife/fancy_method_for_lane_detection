import pyzed.sl as sl
import cv2
import time
import math
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size=5):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold = 50, high_threshold = 150):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def get_white_pixels(img):
    # Get the indices of the white pixels
    white_pixels = np.argwhere(img > 0)
    return np.flip(white_pixels, axis=1)

def perform_dbscan_and_get_clusters(points, eps=5, min_samples=5):
    # Check if there are points to cluster
    """
    Perform DBSCAN clustering on a given point cloud, remove noise, and return points in clusters.

    :param point_cloud: List of points in the format [[x1, y1, z1], [x2, y2, z2], ...]
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: A list of clusters, where each cluster is a list of points.
    """
    if len(points) == 0:
        return []

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    # Organize points into clusters
    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])
    cluster_list = list(clusters.values())
    return cluster_list


def calculate_cluster_centers(clusters):
    centers = []
    for cluster in clusters:
        centers.append(np.mean(cluster, axis=0))
    return np.array(centers).astype(int)

# def mark_centers_on_image(image, centers):
#     for center in centers:
#         cv2.circle(image, tuple(center), 3, (0, 0, 255), -1)
#     return image

def mark_centers_on_image(image, centers):
    """Marks the center points on the image."""
    marked_image = np.copy(image)
    for center in centers:
        # Convert the coordinates to integers for visualization
        int_center = [int(round(coord)) for coord in center]
        cv2.circle(marked_image, tuple(int_center), 3, (0, 0, 255), -1)
    return marked_image



def mark_valid_on_image(image, centers):
    """Marks the center points on the image."""
    marked_image = np.copy(image)
    for center in centers:
        # Convert the coordinates to integers for visualization
        int_center = [int(round(coord)) for coord in center]
        cv2.circle(marked_image, tuple(int_center), 3, (0, 0, 255), -1)
    return marked_image


def bird_eye_view_transform(image, src_points, dst_points):
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    return transformed_image

def get_center_3d_coordinates(point_cloud, centers):
    """
    Get the 3D coordinates (x, y) of the center pixels from a point cloud.

    :param point_cloud: The point cloud object to extract 3D coordinates from.
    :param centers: List of pixel coordinates representing the centers.
    :return: List of 3D coordinates.
    """
    center_3d_coordinates = []

    # Loop through all the centers
    for center in centers:
        i, j = center
        # Get the 3D point cloud values for pixel (i, j)
        status, point3D = point_cloud.get_value(i, j)
        x, y, z, color = point3D
        
        # Check if the point is valid (not NaN)
        if status and not np.isnan(x) and not np.isnan(y):
            # Append the 3D coordinates (x, y) to the list
            center_3d_coordinates.append([x, y])

    return np.array(center_3d_coordinates)

def detect_left_turn(points, curvature_threshold=0.1):
    if len(points) == 0:
        return None, None

    # 분리 x, y 좌표
    x = points[:, 0]
    y = points[:, 1]

    # x를 기준으로 정렬
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    # Spline fitting
    spline = UnivariateSpline(x, y, k=4, s=0)

    # x 범위에 대해 y 값을 계산
    x_range = np.linspace(np.min(x), np.max(x), 100)
    y_spline = spline(x_range)

    # Spline의 두 번째 도함수 (곡률) 계산
    curvature = spline.derivative(2)(x_range)

    # 좌회전을 시작하는 지점과 끝나는 지점 찾기
    turning_started = False
    start_idx = 0
    end_idx = 0
    for i in range(len(curvature)):
        if curvature[i] > curvature_threshold and not turning_started:
            turning_started = True
            start_idx = i

        if curvature[i] < curvature_threshold and turning_started:
            end_idx = i
            break

    # 유효한 점들과 곡률 값을 반환
    valid_points = np.column_stack((x_range[start_idx:end_idx + 1], y_spline[start_idx:end_idx + 1]))
    valid_curvature = curvature[start_idx:end_idx + 1]

    return valid_points, valid_curvature




def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_minimum_distance = 1.0
    init_params.camera_fps = 30  # Set fps at 30
    
    zed.open(init_params)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()

    runtime_parameters.confidence_threshold = 80
    runtime_parameters.texture_confidence_threshold = 100

    # Example source points - (x, y) format
    src_points = np.array([[200, 720], [600, 450], [680, 450], [1100, 720]], dtype="float32")

    # Example destination points - we want to transform the source points into a rectangle
    dst_points = np.array([[300, 720], [300, 0], [900, 0], [900, 720]], dtype="float32")


    while True:
        # 현재 카메라 화면 읽어오기
        image_zed = sl.Mat()
        depth_zed = sl.Mat()
        point_cloud = sl.Mat()
        # img 에 현재 카메라가 담고있는 프레임 들어감
        # err = zed.grab(runtime_parameters)
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            img = image_zed.get_data()
            gray_img = grayscale(img)
            gray_img = gaussian_blur(gray_img, kernel_size=5)
            cany_img = canny(gray_img, low_threshold=50, high_threshold=150)
            
            white_pixels = get_white_pixels(cany_img)
            clusters = perform_dbscan_and_get_clusters(white_pixels, eps=5, min_samples=8)
            centers = calculate_cluster_centers(clusters)
            center_coord = get_center_3d_coordinates(point_cloud, centers)
            valid_points, valid_curvature = detect_left_turn(center_coord, curvature_threshold=0)
            valid = np.array(valid_points)
            

            # Bird eye view transform
            # bird_eye_view = bird_eye_view_transform(cany_img, src_points, dst_points)
            # cv2.imshow("Bird Eye View", bird_eye_view)


            # Show the canny image
            # cv2.imshow("Image", cany_img)


            # Create a color version of the canny image to mark the centers
            canny_color = cv2.cvtColor(cany_img, cv2.COLOR_GRAY2BGR)
            result_image = mark_centers_on_image(canny_color, centers)
            # result_image = mark_valid_on_image(canny_color, valid_points)
            print(len(valid))
            # print(valid_points)
            # print(center_coord)
            print(len(center_coord))
            # print(centers)

            cv2.imshow('Cluster Centers', result_image)

            if cv2.waitKey(1) == ord('q'):
                break

    # Close the camera
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
