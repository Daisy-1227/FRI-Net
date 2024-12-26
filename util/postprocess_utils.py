import os
import cv2
import numpy as np
import copy
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
import sys


def is_clockwise(points):
    """Check whether a sequence of points is clockwise ordered
    """
    # points is a list of 2d points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0

def resort_corners(corners):
    """Resort a sequence of corners so that the first corner starts
       from upper-left and counterclockwise ordered in image
    """
    corners = corners.reshape(-1, 2)
    x_y_square_sum = corners[:,0]**2 + corners[:,1]**2 
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])

    ## sort points clockwise (counterclockwise in image)
    if not is_clockwise(corners_sorted[:,:2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)

    return corners_sorted.reshape(-1)

def simplify_polygon(input_poly):
    def is_angle_change(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        return np.abs(angle) > 1e-2  # Adjust the threshold as needed

    simplified_poly_np = []
    for i in range(len(input_poly)):
        if is_angle_change(input_poly[(i - 1)%len(input_poly)], input_poly[i], input_poly[(i + 1)%len(input_poly)]):
            simplified_poly_np.append(input_poly[i])
    # simplified_poly_np.append(poly_np[-1])
    simplified_poly_np = np.array(simplified_poly_np, dtype=np.uint8)
    
    return simplified_poly_np


def remove_multi_polygon(polygon_lst):
    for poly_idx, polygon in enumerate(polygon_lst):
        connect_edges = []
        if isinstance(polygon, MultiPolygon):
            poly_eqs = []
            poly_pts = []
            poly_v = []
            for sub_polygon in polygon:
                # There may exists some exterior corners in the vertices, remove them
                poly_np = np.array(sub_polygon.exterior.coords, dtype=np.uint8)[:-1]
                simplified_poly_np = simplify_polygon(poly_np)
                poly_np = simplified_poly_np
                poly_eq = []
                poly_pt = []
                for i in range(len(poly_np)-1):
                    start_point = poly_np[i]
                    end_point = poly_np[(i + 1) % len(poly_np)]
                    if start_point[0] == end_point[0]:  # Vertical line
                        line_eq = [float('inf'), start_point[0]]
                    elif start_point[1] == end_point[1]:  # Horizontal line
                        line_eq = [0, start_point[1]]
                    else:
                        line_eq = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
                    poly_eq.append(line_eq)
                    poly_pt.append([start_point, end_point])
                    
                poly_v.append(poly_np)
                poly_eqs.append(poly_eq)
                poly_pts.append(poly_pt)
            
            assert len(poly_v) == len(poly_eqs) == 2
            

            for i, poly_eq in enumerate(poly_eqs):
                src_polygon = poly_v[i]
                tgt_polygon = poly_v[(i+1)%len(poly_v)]
                poly_pt = poly_pts[i]
                for eq_i, eq in enumerate(poly_eq):
                    # Create a line based on the equation
                    pt = poly_pt[eq_i]

                    if eq[0] == float('inf'):  # Vertical line
                        line = LineString([(eq[1], 0), (eq[1], 255)])
                    elif eq[0] == 0:  # Horizontal line
                        line = LineString([(0, eq[1]), (255, eq[1])])
                    else:  # Diagonal line
                        line = LineString([(0, eq[1]), (255, eq[0] * 255 + eq[1])])
                    
                    intersection = line.intersection(Polygon(tgt_polygon))

                    # If there is an intersection, return the first intersection point
                    if not intersection.is_empty:
                        if intersection.geom_type == 'Point':
                            intersection_point = (intersection.x, intersection.y)
                        elif intersection.geom_type == 'MultiPoint':
                            intersection_point = (intersection[0].x, intersection[0].y)
                        elif intersection.geom_type == 'LineString':
                            intersection_point = [(intersection.coords[0][0], intersection.coords[0][1]), 
                                                  (intersection.coords[1][0], intersection.coords[1][1])]
                        else:
                            continue

                        # Calculate the distance between points in pt and intersection_point
                        distances = []
                        for p in pt:
                            for ip in intersection_point:
                                distance = np.sqrt((p[0] - ip[0])**2 + (p[1] - ip[1])**2)
                                distances.append((distance, p, ip))

                        # Find the pair with the minimum distance
                        min_distance, min_pt, min_ip = min(distances, key=lambda x: x[0])

                        # Check if the line segment intersects with src_polygon
                        line_segment = LineString([min_pt, min_ip])
                        if line_segment.intersection(Polygon(src_polygon)).geom_type == 'Point':
                            connect_edges.append([min_pt, min_ip])
            
            if len(connect_edges) == 1:
                merge_polygon = polygon[0] if polygon[0].area > polygon[1].area else polygon[1]
            else:
                edge_points = np.array([point for edge in connect_edges for point in edge], dtype=np.uint8)
                new_edge_points = []
                new_edge_points.append(edge_points[0])
                edge_points = np.delete(edge_points, 0, axis=0)
                while edge_points.shape[0] > 0:
                    for idx, point in enumerate(edge_points):
                        if point[0] == new_edge_points[-1][0] or point[1] == new_edge_points[-1][1]:
                            new_edge_points.append(point)
                            edge_points = np.delete(edge_points, idx, axis=0)
                            break
                edge_polygon = Polygon(new_edge_points)                
                merge_polygon = unary_union([Polygon(src_polygon), Polygon(tgt_polygon), edge_polygon])

            poly_np = np.array(merge_polygon.exterior.coords, dtype=np.uint8)[:-1]
            simplified_poly_np = simplify_polygon(poly_np)
            # poly_np = simplified_poly_np
            poly_np = np.concatenate([simplified_poly_np, simplified_poly_np[None, 0]])
            update_polygon = Polygon(poly_np)
            polygon_lst[poly_idx] = update_polygon

    for polygon in polygon_lst:
        assert polygon.geom_type == 'Polygon'
    return polygon_lst

def remove_rooms_with_iou(polygon_list):
    # Compute the IOU between each pair
    room_map_list = []
    for room_ind, poly in enumerate(polygon_list):
        room_map = np.zeros((256, 256))
        cv2.fillPoly(room_map, [np.array(poly.exterior.coords, dtype=np.int32)[:-1]], color=1.)
        room_map_list.append(room_map)

    access_mat = np.zeros((len(polygon_list), len(polygon_list)))
    remove_indices = []
    for idx0, polygon0 in enumerate(polygon_list):
        for idx1, polygon1 in enumerate(polygon_list):
            if idx0 == idx1 or access_mat[idx0][idx1] == 1 or access_mat[idx1][idx0] == 1:
                continue
            # compute the iou bewteen polygon0 and polygon1
            intersection = ((room_map_list[idx0] + room_map_list[idx1]) == 2)
            union = ((room_map_list[idx0] + room_map_list[idx1]) >= 1)
            iou = np.sum(intersection) / (np.sum(union) + 1)
            if iou > 0.4:
                remove_indices.append([idx0, idx1])
            # print(f'idx_0: {idx0}, idx1: {idx1}, iou: {iou}')
            access_mat[idx0][idx1] = 1
            access_mat[idx1][idx0] = 1
    
    for remove_index in remove_indices:
        idx0, idx1 = remove_index[0], remove_index[1]
        polygon0, polygon1 = polygon_list[idx0], polygon_list[1]
        poly_area0, poly_area1 = polygon0.area, polygon1.area
        if poly_area0 > poly_area1:
            del polygon_list[idx1]
        else:
            del polygon_list[idx0]
    return polygon_list

def refine_rooms(polygon_list):
    access_mat = np.zeros((len(polygon_list), len(polygon_list)))
    for idx0, polygon0 in enumerate(polygon_list):
        for idx1, polygon1 in enumerate(polygon_list):
            if idx0 == idx1 or access_mat[idx0][idx1] == 1 or access_mat[idx1][idx0] == 1:
                continue
            if polygon0.intersects(polygon1):
                intersection = polygon0.intersection(polygon1)
                intersection_area = intersection.area
                if intersection_area >= 20:
                    # remove intersection from larger polygon
                    area0, area1 = polygon0.area, polygon1.area

                    if area0 > area1 and area1 > intersection_area:
                        polygon0 = polygon0.difference(intersection)
                        polygon1 = polygon1.union(intersection)
                    elif area0 < area1 and area0 > intersection_area:
                        polygon1 = polygon1.difference(intersection)
                        polygon0 = polygon0.union(intersection)
                    elif area0 > area1 and area1 == intersection_area:
                        polygon0 = polygon0.difference(intersection)
                        polygon1 = polygon1.union(intersection)

                        if polygon0.geom_type == 'MultiPolygon':
                            polygon_num = len(polygon0.geoms)
                            max_area = 0
                            smaller_polygon = None
                            for _ in range(polygon_num):
                                area = polygon0.geoms[_].area
                                if area > max_area:
                                    smaller_polygon = polygon0.geoms[_]
                                    max_area = area
                            polygon0 = smaller_polygon
                    elif area0 < area1 and area0 == intersection_area:
                        polygon1 = polygon1.difference(intersection)
                        polygon0 = polygon0.union(intersection)
                        if polygon1.geom_type == 'MultiPolygon':
                            polygon_num = len(polygon1.geoms)
                            max_area = 0
                            smaller_polygon = None
                            for _ in range(polygon_num):
                                area = polygon1.geoms[_].area
                                if area > max_area:
                                    smaller_polygon = polygon1.geoms[_]
                                    max_area = area
                            polygon1 = smaller_polygon
                    polygon_list[idx0] = polygon0
                    polygon_list[idx1] = polygon1
            access_mat[idx0][idx1] = 1
            access_mat[idx1][idx0] = 1
    return polygon_list

def remove_duplicate_corners(polygon_list):
    for poly_idx, polygon in enumerate(polygon_list):
        poly_np = np.array(polygon.exterior.coords, dtype=np.uint8)[:-1]
        simplified_poly_np = simplify_polygon(poly_np)
        # poly_np = simplified_poly_np
        poly_np = np.concatenate([simplified_poly_np, simplified_poly_np[None, 0]])
        update_polygon = Polygon(poly_np)
        polygon_list[poly_idx] = update_polygon    

    return polygon_list


def refine_corners(polygon_list):
    room_polys = []

    for polygon in polygon_list:

        # visualize([polygon], img)

        arr = np.array(polygon.exterior.coords, dtype=np.int32)[:-1]
        arr_len = len(arr)
        valid_corners = []
        for _ in range(arr_len):
            if (arr[(_ - 1) % arr_len][0] == arr[_][0] and arr[(_ + 1) % arr_len][0] == arr[_][0]) or (
                    arr[(_ - 1) % arr_len][1] == arr[_][1] and arr[(_ + 1) % arr_len][1] == arr[_][1]):
                continue
            if arr[_][0] < 0 or arr[_][1] < 0:
                arr[_][arr[_] < 0] = 0
            if arr[_][0] > 255 or arr[_][1] > 255:
                arr[_][arr[_] > 255] = 255
            valid_corners.append(arr[_])
        if len(valid_corners) > 0:
            valid_corners.append(valid_corners[0])
        room = np.array(valid_corners)
        
        refine_corners = []
        _ = 0
        
        while _ < len(room) - 1:
            corner0, corner1 = room[_], room[_ + 1]
            if corner0[0] == corner1[0] and abs(corner0[1] - corner1[1]) <= 3:
                refine_corners.append(corner0)
                corner3 = room[(_ + 2) % len(room)]
                corner3[1] = corner0[1]
                refine_corners.append(corner3)
                _ += 3
            elif corner0[1] == corner1[1] and abs(corner0[0] - corner1[0]) <= 3:
                refine_corners.append(corner0)
                corner3 = room[(_ + 2) % len(room)]
                corner3[0] = corner0[0]
                refine_corners.append(corner3)
                _ += 3
            else:
                refine_corners.append(corner0)
                _ += 1
        # visualize([Polygon(np.array(refine_corners))], img)
        
        # simplify corners
        room_corners = simplify_polygon(refine_corners)

        room_polys.append(Polygon(room_corners))

    return room_polys

def postprocess(polygon_list):
    # First fix the multi_polygon
    polygon_list = remove_multi_polygon(polygon_list)
    
    # visualize(polygon_list, img)
    # Second remove rooms with high iou
    polygon_list = remove_rooms_with_iou(polygon_list)

    # visualize(polygon_list, img)
    # Third refine the shape of rooms with neighbor rooms
    polygon_list = refine_rooms(polygon_list)
    
    # visualize(polygon_list, img)
    # remove duplicate corners
    polygon_list = remove_duplicate_corners(polygon_list)

    # visualize(polygon_list, img)

    # check the validity of each room
    for polygon in polygon_list:
        assert polygon.geom_type == 'Polygon'
    
    polygon_list = refine_corners(polygon_list)
    room_polys = []
    for polygon in polygon_list:
        room = np.array(polygon.exterior.coords, dtype=np.int32)[:-1]
        room_polys.append(room)
    
    return room_polys