import copy
import functools
import numpy as np
import os
import csv

from Evaluator.Evaluator import Evaluator
from options import MCSSOptions
from DataRW.S3DRW import S3DRW
from DataRW.wrong_annotatios import wrong_s3d_annotations_list
from planar_graph_utils import get_regions_from_pg


room_polys_def = [np.array([[191, 150],
       [191,  70],
       [222,  70],
       [222, 150],
       [191, 150]]), np.array([[232,  65],
       [232,  11],
       [202,  11],
       [202,  65],
       [232,  65]]), np.array([[ 47,  50],
       [ 47, 150],
       [ 24, 150],
       [ 24,  50],
       [ 47,  50]]), np.array([[199, 156],
       [199, 234],
       [146, 234],
       [146, 156],
       [199, 156]]), np.array([[109, 184],
       [120, 184],
       [120, 156],
       [ 50, 156],
       [ 50, 234],
       [109, 234],
       [109, 184]]), np.array([[110, 234],
       [144, 234],
       [144, 187],
       [110, 187],
       [110, 234]]), np.array([[ 50,  50],
       [ 50, 150],
       [123, 150],
       [123, 184],
       [144, 184],
       [144, 150],
       [190, 150],
       [190,  70],
       [108,  70],
       [108,  50],
       [ 50,  50]])]

pg_base = "./results/npy"

options = MCSSOptions()
opts = options.parse()

if __name__ == '__main__':

    # data_rw = FloorNetRW(opts)

    if opts.scene_id == "val":

        opts.scene_id = "scene_03250" # Temp. value
        data_rw = S3DRW(opts, mode='test')
        scene_list = data_rw.loader.scenes_list

        quant_result_dict = None
        quant_result_maskrcnn_dict = None
        scene_counter = 0

        room_prec, room_rec, corner_prec, corner_rec, angles_prec, angles_rec = list(), list(), list(), list(), list(), list()
        scene_ids = list()

        for scene_ind, scene in enumerate(scene_list):
            if int(scene[6:]) in wrong_s3d_annotations_list:
                continue

            if int(scene[6:]) in []:
                continue

            # if int(scene[6:]) not in eval_list:
            #     continue

            print("------------")
            curr_opts = copy.deepcopy(opts)
            curr_opts.scene_id = scene
            curr_data_rw = S3DRW(curr_opts, mode='test')

            print("Running Evaluation for scene %s" % scene)

            evaluator = Evaluator(curr_data_rw, curr_opts)

            # TODO load your room polygons into room_polys, list of polygons (n x 2)
            # room_polys = np.array([[[0,0], [200, 0], [200, 200]]]) # Placeholder
            pg_path = os.path.join(pg_base, scene[6:] + '.npy')
            if not os.path.exists(pg_path):
                continue

            polygon_list = np.load(pg_path, allow_pickle=True).tolist()
            # polygon_list = np.load(pg_path, allow_pickle=True)['room_polys'].tolist()
            # if polygon_list[0] == None:
                # continue
            room_polys = [] 
            for polygon in polygon_list:
                arr = np.array(polygon, dtype=np.int32)
                room_polys.append(arr)

            try:
                quant_result_dict_scene =\
                    evaluator.evaluate_scene(room_polys=room_polys)
            except:
                continue

            corner_prec_map = quant_result_dict_scene['corner_prec_map']
            corner_rec_map = quant_result_dict_scene['corner_rec_map']

            if quant_result_dict is None:
                quant_result_dict_scene.pop('corner_prec_map')
                quant_result_dict_scene.pop('corner_rec_map')
                quant_result_dict = quant_result_dict_scene
            else:
                for k in quant_result_dict.keys():
                    if k == 'corner_prec_map' or k == 'corner_rec_map':
                        continue
                    quant_result_dict[k] += quant_result_dict_scene[k]

            scene_counter += 1
            
            room_prec.append(quant_result_dict_scene['room_prec'])
            room_rec.append(quant_result_dict_scene['room_rec'])
            corner_prec.append(quant_result_dict_scene['corner_prec'])
            corner_rec.append(quant_result_dict_scene['corner_rec'])
            angles_prec.append(quant_result_dict_scene['angles_prec'])
            angles_rec.append(quant_result_dict_scene['angles_rec'])
            scene_ids.append(scene)

        for k in quant_result_dict.keys():
            quant_result_dict[k] /= float(scene_counter)

        print("Our: ", quant_result_dict)

        print("Ours")
        evaluator.print_res_str_for_latex(quant_result_dict)
