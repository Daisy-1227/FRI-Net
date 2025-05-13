import pclpy
from pclpy import pcl

import os

import open3d as o3d
import numpy as np

def func(points):
    pc = get_pc(points)
    all_planes = []
    all_eqs = []

    while True:
        if len(pc.points) < 20:
            break

        thresh = 200

        eqs, lins = pc.segment_plane( distance_threshold=thresh, ransac_n=4, num_iterations=1000 )

        p_num = len(lins)

        plane_pc = pc.select_by_index(lins)

        plane_pc.paint_uniform_color(np.random.rand(3))

        all_planes.append(plane_pc)

        all_eqs.append(eqs)

        pc = pc.select_by_index(lins, invert=True)
        # npc = npc.select_by_index(lins, invert=True)

    return all_planes, all_eqs


def get_plane_eqs(pc, thresh = 200):
    eqs, lins = pc.segment_plane( distance_threshold=thresh, ransac_n=4, num_iterations=1000 )
    return eqs

def get_pc(points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector( points )
    return pc

def vis(points, normals):
    pc = get_pc(points)

    if normals is not None:
        pc.normals = o3d.utility.Vector3dVector( normals )
    
    o3d.visualization.draw_geometries([pc], point_show_normal=normals is not None)

if __name__ == "__main__":
    
    from tqdm import tqdm
    #### Get the point cloud from Structured3D dataset [https://github.com/bertjiazheng/Structured3D]
    origin_path = "F:/dataset/floorplane/pc" # /scene_xxx
    new_path = "F:/dataset/floorplane/height_pc"

    id_list = [ i for i in range(3500)]
    for i in tqdm(id_list):

        plyreader = pclpy.pcl.io.PLYReader()
        point_cloud = pclpy.pcl.PointCloud.PointXYZ()
        point_rgb = pclpy.pcl.PointCloud.PointXYZRGB()

        name = f"{i:05d}"
        
        pcd_path = f"{origin_path}/scene_{name}/point_cloud.ply"

        if not os.path.exists(pcd_path): continue
        
        
        save_path = f"{new_path}/scene_{name}"
        
        os.makedirs(save_path, exist_ok=True)
        
        json_path = f"{origin_path}/scene_{name}/annotation_3d.json"
        
        eq_path = f"{save_path}/eq.npy"
        ids_path = f"{save_path}/ids.npy"
        
        # if os.path.exists(json_path):
        #     cmd = "copy %s %s\\" % (json_path.replace('/', '\\'), save_path.replace('/', '\\'))
        #     # print(cmd)
        #     os.system(cmd)


        out_path = f"{save_path}/point_cloud.ply"

        plyreader.read(pcd_path, point_cloud)
        plyreader.read(pcd_path, point_rgb)

        tree = pcl.search.KdTree.PointXYZ()

        normals = pclpy.compute_normals(point_cloud, k=30)

        reg = pcl.segmentation.RegionGrowing.PointXYZ_Normal()
        reg.setMinClusterSize(40)
        reg.setMaxClusterSize(1000000)
        reg.setSearchMethod(tree)
        # reg.setNumberOfNeighbours(32)
        reg.setNumberOfNeighbours(45)
        reg.setInputCloud(point_cloud)
        reg.setInputNormals(normals)
        reg.setSmoothnessThreshold(3.0 / 180.0 * np.pi)
        reg.setCurvatureThreshold(0.5)

        all_ids = pcl.vectors.PointIndices()
        reg.extract(all_ids)

        all_planes = []
        all_points = []
        all_rgb = []
        all_eqs = []

        origin_xyz = point_cloud.xyz
        ox = origin_xyz[:,0]
        oy = origin_xyz[:,1]

        ox_min, ox_max = ox.min(), ox.max()
        oy_min, oy_max = oy.min(), oy.max()
        

        # all_planes.append( o3d.io.read_point_cloud(pcd_path) )
        
        final_ids = []
        # for ids in []:
        for ids in all_ids:
            # 去掉小区域的
            if len(ids.indices) < 300: continue
            points = point_cloud.xyz[ids.indices]
            rgb = point_rgb.rgb[ids.indices]

            # 去掉横向的
            # if points[:,2].max() - points[:,2].min() < 400: continue

            pc = get_pc(points)

            ap, aq = func(points)
            # print(len(aq))
            eqs = aq[0]
            if abs(eqs[:3]).argmax() == 2: continue

            all_eqs.append(aq[0])
            # o3d.visualization.draw_geometries(ap)

            # pc.paint_uniform_color(np.random.rand(3))
            all_planes.append(pc)
            all_points.append(points)
            all_rgb.append(rgb)
            final_ids.append(np.array(ids.indices))
            
        # print(len(all_planes))

        # o3d.visualization.draw_geometries(all_planes)

        all_points.append(
            np.array([
                [ ox_min, oy_min, origin_xyz[0][2] ],
                [ ox_max, oy_max, origin_xyz[0][2] ]
                ] )
        )
        all_rgb.append(
            np.array([
                [ 0,0,0 ],
                [ 0,0,0 ]
            ] )
        )
        
        all_points = np.concatenate(all_points, axis=0)
        all_rgb = np.concatenate(all_rgb, axis=0)

        result_pc = get_pc(all_points)
        result_pc.colors = o3d.utility.Vector3dVector(all_rgb / 255.0)
        o3d.io.write_point_cloud(out_path, result_pc, compressed=False)

        np.save(eq_path, all_eqs)
        np.save(ids_path, final_ids)

