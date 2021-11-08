# Copyright (c) 2018 Andy Zeng

import numpy as np
import torch

from numba import njit, prange
from skimage import measure
import matplotlib.cm as cm
from graphviz import Digraph
import webcolors
import os
import cv2

# For color clustering
import operator
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from stitch import Stitch
from skimage.measure import ransac, EllipseModel

# from data import COLORS
import warnings
warnings.filterwarnings(action='ignore')


try:
  import pycuda.driver as cuda
  import pycuda.autoinit
  from pycuda.compiler import SourceModule
  FUSION_GPU_MODE = 1
except Exception as err:
  print('Warning: {}'.format(err))
  print('Failed to import PyCUDA. Running fusion in CPU mode.')
  FUSION_GPU_MODE = 0


class Get_Color_Info(object):
  def __init__(self):
    self.COLORS = \
      ((244,  67,  54),
       (233,  30,  99),
       (156,  39, 176),
       (103,  58, 183),
       ( 63,  81, 181),
       ( 33, 150, 243),
       (  3, 169, 244),
       (  0, 188, 212),
       (  0, 150, 136),
       ( 76, 175,  80),
       (139, 195,  74),
       (205, 220,  57),
       (255, 235,  59),
       (255, 193,   7),
       (255, 152,   0),
       (255,  87,  34),
       (121,  85,  72),
       (158, 158, 158),
       ( 96, 125, 139))

  def closest_colour(self, requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
      r_c, g_c, b_c = webcolors.hex_to_rgb(key)
      rd = (r_c - requested_colour[0]) ** 2
      gd = (g_c - requested_colour[1]) ** 2
      bd = (b_c - requested_colour[2]) ** 2
      min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

  def get_colour_name(self, requested_colour):
    try:
      closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
      closest_name = self.closest_colour(requested_colour)
      actual_name = None
    return actual_name, closest_name

  def RGB2HEX(self, color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

  def get_color_hist_kmeans(self, color_list):
    clf = KMeans(n_clusters=4)
    labels = clf.fit_predict(color_list)
    counts = Counter(labels)
    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    center_colors = clf.cluster_centers_

    ordered_colors = [center_colors[i[0]] for i in sorted_counts]
    hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    color_hist = []
    for i, rgb in enumerate(rgb_colors):
      actual_name, closest_name = self.get_colour_name(list(np.int_(rgb)))
      if (actual_name == None):
        color_hist.append([hex_colors[i], closest_name])
      else:
        color_hist.append([hex_colors[i], actual_name])

    return color_hist

class Same_Node_Detection(object):
  def __init__(self):
    self.class_w = 8.0 / 20.0
    self.pose_w = 10.0 / 20.0
    self.color_w = 2.0 / 20.0
    self.th = 0.23
    self.node_th = 0.80

  def compare_class(self, curr_cls, prev_cls, cls_score):
    if (curr_cls == prev_cls):
      score = 1.0
    else:
      score = 0.0
    return score

  def compare_position(self, c_m, c_v, p_m, p_v, prev_pt_num, new_pt_num):
    # In standardized normal gaussian distribution
    # Threshold : 0.9 --> -1.65 < Z < 1.65
    #           : 0.8 --> -1.29 < Z < 1.29
    #           : 0.7 --> -1.04 < Z < 1.04

    # curr_min_range = -self.th*c_v + c_m
    # curr_max_range =  self.th*c_v + c_m
    #
    # prev_min_range = -self.th*p_v + p_m
    # prev_max_range =  self.th*p_v + p_m
    #
    # prev_volume = np.product(prev_max_range - prev_min_range)
    # curr_volume = np.product(curr_max_range - curr_min_range)
    #
    # #overlapped_area_x = min(cur_x_range[1],pre_x_range[1]) - max(cur_x_range[0], pre_x_range[0])
    # overlapped_area_x = min(curr_max_range[0], prev_max_range[0]) - max(curr_min_range[0], prev_min_range[0])
    # I_x = 0.0 if overlapped_area_x < 0 else overlapped_area_x
    #
    # overlapped_area_y = min(curr_max_range[1], prev_max_range[1]) - max(curr_min_range[1], prev_min_range[1])
    # I_y = 0.0 if overlapped_area_y < 0 else overlapped_area_y
    #
    # overlapped_area_z = min(curr_max_range[2], prev_max_range[2]) - max(curr_min_range[2], prev_min_range[2])
    # I_z = 0.0 if overlapped_area_z < 0 else overlapped_area_z

    distance = c_m - p_m
    I_x = 1.0 if (np.abs(distance[0]) < self.th) else self.th / (np.abs(distance[0]))
    I_y = 1.0 if (np.abs(distance[1]) < self.th) else self.th / (np.abs(distance[1]))
    I_z = 1.0 if (np.abs(distance[2]) < self.th) else self.th / (np.abs(distance[2]))

    # score = (I_x * I_y * I_z) / max(prev_volume, curr_volume)
    # score = I_x/3 + I_y/3 + I_z/3
    score = I_x * I_y * I_z
    return score

  def compare_color(self, curr_hist, prev_hist):
    curr_rgb = webcolors.name_to_rgb(curr_hist[0][1])
    prev_rgb = webcolors.name_to_rgb(prev_hist[0][1])
    dist = np.sqrt(np.sum(np.power(np.subtract(curr_rgb, prev_rgb), 2))) / (255 * np.sqrt(3))
    score = 1 - dist
    return score

  def node_update(self, global_node, curr_mean, curr_var, curr_pt_num, curr_cls, cls_score, curr_color_hist):
    w1, w2, w3 = self.class_w, self.pose_w, self.color_w
    score = []
    pos_test = []
    cls_test = []
    idx_test = []
    for idx in global_node.keys():
      prev_cls = global_node[str(idx)]['class']
      prev_mean = global_node[str(idx)]['mean']
      prev_var = global_node[str(idx)]['var']
      prev_pt_num = global_node[str(idx)]['pt_num']
      prev_color_hist = global_node[str(idx)]['color_hist']

      score_cls = self.compare_class(curr_cls, prev_cls, cls_score)
      score_pos = self.compare_position(curr_mean, curr_var,
                                        prev_mean, prev_var,
                                        prev_pt_num, curr_pt_num)
      score_col = self.compare_color(curr_color_hist, prev_color_hist)
      # score_col = self.compare_color(curr_color_hist, prev_color_hist)

      score_tot = (w1 * score_cls) + (w2 * score_pos) + (w3 * score_col)
      # score_tot = (w1 * score_cls) + (w2 * score_pos)
      score.append(score_tot)
      pos_test.append(score_pos)
      cls_test.append(score_cls)
      idx_test.append(idx)
    node_score = max(score)
    max_score_index = score.index(max(score))
    matched_idx = idx_test[max_score_index]
    # for testing
    if node_score > self.node_th:
      print("updated node_score and id --> class name: {},{}, {}".format(node_score,
                                                                         max_score_index,
                                                                         global_node[str(matched_idx)]['class']))
      print("cls_score and pos_score: {},{}".format(cls_test[max_score_index], pos_test[max_score_index]))
    else:
      print('!!!!new node was created : {}'.format(curr_cls))
      print("node_score : {}".format(node_score))
      print("cls_score and pos_score: {},{}".format(cls_test[max_score_index], pos_test[max_score_index]))

    return node_score, max_score_index


class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, vol_bnds, voxel_size, use_gpu=True, root_path=None, cfg=None):
    """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    self.cfg = cfg

    # setting root path
    self.root_path = os.path.join(root_path, 'scene_results')
    if not os.path.exists(self.root_path):
      os.mkdir(self.root_path)
    self.f_idx = 0
    # setting save info into folders (scene_graph, bounidng_box)
    self.bbox_path = os.path.join(self.root_path, 'BBOX')
    if not os.path.exists(self.bbox_path):
      os.mkdir(self.bbox_path)
    self.scene_graph_path = os.path.join(self.root_path, 'scene_graph')
    if not os.path.exists(self.scene_graph_path):
      os.mkdir(self.scene_graph_path)

    # Initialize color system
    self.GCI = Get_Color_Info()
    colors = cm.rainbow(np.linspace(0, 1, 80))
    self.class_colors = (colors*255)[:,:3].astype('int')

    # Initialize scene graph data
    self.node_data = {}
    self.rel_data = {}

    # Initialize same node detection system
    self.SND = Same_Node_Detection()
    self.debug_same_node_detector = True
    self.ID_2D = 0
    self.stitch_img = None
    self.wide_class_mask = None
    self.apply_panorama = False
    self.stitch_method = Stitch()
    self.stitch_list = {}
    self.thumbnail_dict = {}


    ''' GFTT-BRIEF '''
    self.feature_detector = cv2.GFTTDetector_create(
      maxCorners=1000, minDistance=12.0,
      qualityLevel=0.001, useHarrisDetector=False)
    self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
      bytes=32, use_orientation=False)
    self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # setting several techniques
    self.is_depth_clustered = True
    ''' cluster technique : ['kmeans', 'meanshift', 'ransac', 'dbscan', 'hdbscan']'''
    self.cluster_technique = 'ransac'
    self.cluster_object = ['dining table', 'bench', 'bottle', 'cup', 'chair', 'bed', 'couch']
    self.mask_data = np.ones([])

    # Define voxel volume parameters
    self._vol_bnds = vol_bnds
    self._voxel_size = float(voxel_size)
    self._trunc_margin = 5*self._voxel_size  # truncation on SDF (orig default : 5)
    self._trunc_margin_mask = 2*self._voxel_size
    self._color_const = 256 * 256

    # Adjust volume bounds and ensure C-order contiguous
    self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
    self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
    self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)
    self._prev_vol_bnds = self._vol_bnds.copy()

    # print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
    #   self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
    #   self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
    # )

    # Initialize pointers to voxel volume in CPU memory
    self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
    # for computing the cumulative moving average of observations per voxel
    self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._class_vol_cpu = -np.ones(self._vol_dim).astype(np.int32)

    self.gpu_mode = use_gpu and FUSION_GPU_MODE

    self._prev_vol_bnds = np.copy(self._vol_bnds)


    # Copy voxel volumes to GPU
    if self.gpu_mode:
      self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
      cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
      self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
      cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
      self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
      cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)
      self._class_vol_gpu = cuda.mem_alloc(self._class_vol_cpu.nbytes)
      cuda.memcpy_htod(self._class_vol_gpu, self._class_vol_cpu)

      # Cuda kernel function (C++)
      self._cuda_src_mod = SourceModule("""
        #include <stdio.h>
        #include <math.h>
       
        __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  int * class_vol,
                                  int * mask_data,
                                  float * mask_color_data,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im) {
          // Get voxel index
          int gpu_loop_idx = (int) other_params[0];
          int num_mask = (int) other_params[6];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];
          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;
              
          // Get voxel grid coordinates (note: be careful when casting)
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
          
          // Voxel grid coordinates to world coordinates
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;
          
          // World coordinates to camera coordinates
          float tmp_pt_x = pt_x-cam_pose[0*4+3];
          float tmp_pt_y = pt_y-cam_pose[1*4+3];
          float tmp_pt_z = pt_z-cam_pose[2*4+3];
          float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
          
          // Camera coordinates to image pixels
          int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
          int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
          
                    
          // Skip if outside view frustum
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
              return;
              
          // Skip invalid depth
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;
          
          // Integrate TSDF
          float trunc_margin = other_params[4];
          float trunc_margin_mask = other_params[7];
          float depth_diff = depth_value-cam_pt_z;
          if (depth_diff < -trunc_margin)
              return;
              
          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
                    
          int update_tsdf = (int) other_params[9];
          int first_masking = (int) other_params[10];
          
          if (update_tsdf == 1){
            weight_vol[voxel_idx] = w_new;
            tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;         
          }
          
          // Integrate color
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;
          //new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
          //new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
          //new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
          if (color_vol[voxel_idx] == 0.0 && class_vol[voxel_idx] == -1 && update_tsdf == 1)
            color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
            
          
          // compare Yolact masked points with current image pixels
          // mask_data : [cls_score][cls_label][2D_instance]
          // class_vol is composed [detect_cnt][cls_score][cls_label][3D_instance]
          int is_first_masked_img = (int) other_params[8];
          if (is_first_masked_img == 0)
            return;
          
          int pixel_xy = pixel_y * im_w + pixel_x;
          if (num_mask == 0 or mask_data[pixel_xy] == -1)
            return;
            
          float mask_depth_diff = cam_pt_z - depth_value;
          if (mask_depth_diff < -trunc_margin_mask)
            return;
          
          int prev_class_vol = class_vol[voxel_idx];
          int prev_mask_data = mask_data[pixel_xy];
          int curr_cls_score = (int) (prev_mask_data/10000);
          int curr_cls_label = ((int)(prev_mask_data/100)) - ((int)(prev_mask_data/10000))*100;
          int instance_2D_ID = ((int) prev_mask_data) - ((int)(prev_mask_data/100))*100;
          
          int detected_cnt_num = 1;
          if (first_masking) {
            class_vol[voxel_idx] = mask_data[pixel_xy] + 1000000;
          }
          else{
            int new_id = 1;
            int voxel_neigh_idx = 0;
            float neigh_class = -1.0;
            // dining table : 60, bed: 59, couch : 57, chair:56, refrigerator : 72, bicycle : 1
            int neigh_gap = 4;
            if (curr_cls_label == 60 || curr_cls_label == 59 || curr_cls_label == 57 || curr_cls_label == 72)
              neigh_gap = 10;
            if (curr_cls_label == 56 || curr_cls_label == 1)
              neigh_gap = 6;
            for (int i = -neigh_gap; i < neigh_gap; i++){
              for (int j = -neigh_gap; j < neigh_gap; j ++){
                for (int k = -neigh_gap; k < neigh_gap; k ++){
                  if (voxel_x+i >= 0 && voxel_y+j >= 0 && voxel_z+k >= 0 && voxel_x+i <= vol_dim_x && voxel_y+j <= vol_dim_y && voxel_z+k <= vol_dim_z){
                    voxel_neigh_idx = ((int)voxel_x + i)*vol_dim_y*vol_dim_z + ((int)voxel_y + j)*vol_dim_z + ((int)voxel_z + k);
                    neigh_class = class_vol[voxel_neigh_idx];
                    if (neigh_class != -1.0){
                      int prev_detected_cnt = (int) (neigh_class/1000000);
                      int prev_cls_score = ((int)(neigh_class/10000)) - ((int)(neigh_class/1000000))*100;
                      int prev_cls_label = ((int)(neigh_class/100)) - ((int)(neigh_class/10000))*100;
                      int prev_instance_3D_ID = ((int) prev_class_vol) - ((int)(neigh_class/100))*100;
                      if (prev_cls_label == curr_cls_label){
                        class_vol[voxel_idx] = neigh_class;
                        new_id = 0;
                      }
                      
                    }
                  }
                }
              }
            }
            if (new_id){
              // update new_id
              class_vol[voxel_idx] = mask_data[pixel_xy] + 1000000;
            }
          }

          
          if (detected_cnt_num >= 1){
            new_color = - (mask_color_data[3*curr_cls_label + 2] + mask_color_data[3*curr_cls_label + 1]*256 + mask_color_data[3*curr_cls_label + 0]*256*256);
            new_b = floorf(new_color/(256*256));
            new_g = floorf((new_color-new_b*256*256)/256);
            new_r = new_color-new_b*256*256-new_g*256;
            color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
          }
          
        }
        
        """)

      self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

      # Determine block/grid size on GPU
      gpu_dev = cuda.Device(0)
      self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
      n_blocks = int(np.ceil(float(np.prod(self._vol_dim))/float(self._max_gpu_threads_per_block)))
      grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,int(np.floor(np.cbrt(n_blocks))))
      grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y,int(np.floor(np.sqrt(n_blocks/grid_dim_x))))
      grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z,int(np.ceil(float(n_blocks)/float(grid_dim_x*grid_dim_y))))
      self._max_gpu_grid_dim = np.array([grid_dim_x,grid_dim_y,grid_dim_z]).astype(int)
      self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim))/float(np.prod(self._max_gpu_grid_dim)*self._max_gpu_threads_per_block)))

    else:
      # Get voxel grid coordinates
      xv, yv, zv = np.meshgrid(
        range(self._vol_dim[0]),
        range(self._vol_dim[1]),
        range(self._vol_dim[2]),
        indexing='ij'
      )
      self.vox_coords = np.concatenate([
        xv.reshape(1,-1),
        yv.reshape(1,-1),
        zv.reshape(1,-1)
      ], axis=0).astype(int).T

  def update(self, vol_bnds, prev_vol_bnds, vol_min, vol_max):
    # Update voxel volume parameters
    #self._prev_vol_bnds = prev_vol_bnds
    self._vol_bnds = self._prev_vol_bnds.copy()

    if self.gpu_mode:
      self._tsdf_vol_cpu, self._color_vol_cpu = self.get_volume()
      cuda.memcpy_dtoh(self._weight_vol_cpu, self._weight_vol_gpu)
      cuda.memcpy_dtoh(self._class_vol_cpu, self._class_vol_gpu)
      self._tsdf_vol_gpu.free()
      self._color_vol_gpu.free()
      self._weight_vol_gpu.free()
      self._class_vol_gpu.free()

    x_extend_b, y_extend_b, z_extend_b = (vol_min/self._voxel_size)
    x_extend_u, y_extend_u, z_extend_u = (vol_max/self._voxel_size)

    self.test1 = (x_extend_b, y_extend_b, z_extend_b)
    self.test2 = (x_extend_u, y_extend_u, z_extend_u)
    # self._vol_bnds = np.add(self._vol_bnds, vol_min)
    # self._vol_origin = np.add(self._vol_origin, np.array(self.test1).astype('float')*self._voxel_size)
    # print('extend method2 volume left below dir : {}, right upper dir : {}\n'.format(self.test1, self.test2))


    if (x_extend_b > 0):
      x_extend_b = np.ceil(x_extend_b).copy(order='C').astype(int)
      self._vol_bnds[:, 0] = self._prev_vol_bnds[:, 0] - np.array([x_extend_b*self._voxel_size, 0., 0.])
      self._vol_origin[0] = self._prev_vol_bnds[:, 0][0] - x_extend_b*self._voxel_size
      self._vol_bnds = np.add(self._vol_bnds, np.array([x_extend_b * self._voxel_size, 0., 0.]).reshape(3, 1))
      self._vol_dim += np.array([x_extend_b, 0, 0])
      for i in range(x_extend_b):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, 0, 1, axis=0)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, 0, 0, axis=0)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, 0, 0, axis=0)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, 0, -1, axis=0)
    if (y_extend_b > 0):
      y_extend_b = np.ceil(y_extend_b).copy(order='C').astype(int)
      self._vol_bnds[:, 0] = self._prev_vol_bnds[:, 0] - np.array([0., y_extend_b * self._voxel_size, 0.])
      self._vol_origin[1] = self._prev_vol_bnds[:, 0][1] - y_extend_b * self._voxel_size
      self._vol_bnds = np.add(self._vol_bnds, np.array([0., y_extend_b*self._voxel_size, 0.]).reshape(3, 1))
      self._vol_dim += np.array([0, y_extend_b, 0])
      for j in range(y_extend_b):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, 0, 1, axis=1)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, 0, 0, axis=1)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, 0, 0, axis=1)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, 0, -1, axis=1)
    if (z_extend_b > 0):
      z_extend_b = np.ceil(z_extend_b).copy(order='C').astype(int)
      self._vol_bnds[:, 0] = self._prev_vol_bnds[:, 0] - np.array([0., 0., z_extend_b * self._voxel_size])
      self._vol_origin[2] = self._prev_vol_bnds[:, 0][2] - z_extend_b * self._voxel_size
      #self._vol_bnds = np.add(self._vol_bnds, np.array([0., 0., z_extend_b*self._voxel_size]).reshape(3, 1))
      self._vol_dim += np.array([0, 0, z_extend_b])
      for k in range(z_extend_b):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, 0, 1, axis=2)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, 0, 0, axis=2)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, 0, 0, axis=2)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, 0, -1, axis=2)

    if (x_extend_u > 0):
      x_extend_u = np.ceil(x_extend_u).copy(order='C').astype(int)
      self._vol_bnds[:, 1] += np.array([x_extend_u, 0, 0])
      self._vol_dim += np.array([x_extend_u, 0, 0])
      for i in range(x_extend_u):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, self._tsdf_vol_cpu.shape[0], 1, axis=0)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, self._weight_vol_cpu.shape[0], 0, axis=0)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, self._color_vol_cpu.shape[0], 0, axis=0)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, self._class_vol_cpu.shape[0], -1, axis=0)
    if (y_extend_u > 0):
      y_extend_u = np.ceil(y_extend_u).copy(order='C').astype(int)
      self._vol_bnds[:, 1] += np.array([0, y_extend_u, 0])
      self._vol_dim += np.array([0, y_extend_u, 0])
      for j in range(y_extend_u):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, self._tsdf_vol_cpu.shape[1], 1, axis=1)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, self._weight_vol_cpu.shape[1], 0, axis=1)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, self._color_vol_cpu.shape[1], 0, axis=1)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, self._class_vol_cpu.shape[1], -1, axis=1)
    if (z_extend_u > 0):
      z_extend_u = np.ceil(z_extend_u).copy(order='C').astype(int)
      self._vol_bnds[:, 1] += np.array([0, 0, z_extend_u])
      self._vol_dim += np.array([0, 0, z_extend_u])
      for k in range(z_extend_u):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, self._tsdf_vol_cpu.shape[2], 1, axis=2)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, self._weight_vol_cpu.shape[2], 0, axis=2)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, self._color_vol_cpu.shape[2], 0, axis=2)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, self._class_vol_cpu.shape[2], -1, axis=2)
    self.move_pose = self._vol_origin - self._vol_bnds[:, 0]
    self._vol_bnds = np.add(self._vol_bnds, self.move_pose.reshape(3, 1))
    self._prev_vol_bnds = self._vol_bnds.copy()


    # Copy voxel volumes to GPU
    if self.gpu_mode:
      self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
      cuda.memcpy_htod(self._tsdf_vol_gpu,self._tsdf_vol_cpu)
      self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
      cuda.memcpy_htod(self._weight_vol_gpu,self._weight_vol_cpu)
      self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
      cuda.memcpy_htod(self._color_vol_gpu,self._color_vol_cpu)
      self._class_vol_gpu = cuda.mem_alloc(self._class_vol_cpu.nbytes)
      cuda.memcpy_htod(self._class_vol_gpu, self._class_vol_cpu)

      gpu_dev = cuda.Device(0)
      n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) / float(self._max_gpu_threads_per_block)))
      grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
      grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
      grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
      self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
      self._n_gpu_loops = int(
        np.ceil(float(np.prod(self._vol_dim)) / float(np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))
    else:
      # Get voxel grid coordinates
      xv, yv, zv = np.meshgrid(
        range(self._vol_dim[0]),
        range(self._vol_dim[1]),
        range(self._vol_dim[2]),
        indexing='ij'
      )
      self.vox_coords = np.concatenate([
        xv.reshape(1,-1),
        yv.reshape(1,-1),
        zv.reshape(1,-1)
      ], axis=0).astype(int).T



  @staticmethod
  @njit(parallel=True)
  def vox2world(vol_origin, vox_coords, vox_size):
    """Convert voxel grid coordinates to world coordinates.
    """
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    for i in prange(vox_coords.shape[0]):
      for j in range(3):
        cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
    # cam_pts = (vox_coords.transpose()*vox_size + vol_origin.reshape(3, -1)).transpose().astype(np.float32)
    return cam_pts

  @staticmethod
  @njit(parallel=True)
  def cam2pix(cam_pts, intr, gpu_mode):
    """Convert camera coordinates to pixel coordinates.
    """
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    if gpu_mode:
      # widen camera view 3x
      cx, cy = intr[0, 2]*3, intr[1, 2]*3
    else:
      cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
      pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
      pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))

    return pix

  @staticmethod
  @njit(parallel=True)
  def valid_cam2pix(cam_pts, intr, mask_idx):
    """Convert camera coordinates to pixel coordinates.
    """
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    pix_index = np.zeros(cam_pts.shape[0])
    for i in prange(cam_pts.shape[0]):
      pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
      pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
      for j in range(mask_idx.shape[0]):
        if (pix[i, 0] == mask_idx[j, 1] and pix[i, 1] == mask_idx[j, 0]):
          pix_index[i] = 1
          break

    return pix, pix_index

  @staticmethod
  @njit(parallel=True)
  def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
    """Integrate the TSDF volume.
    """
    tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
    w_new = np.empty_like(w_old, dtype=np.float32)
    for i in prange(len(tsdf_vol)):
      w_new[i] = w_old[i] + obs_weight
      tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
    return tsdf_vol_int, w_new

  def integrate(self, color_im, depth_im, cam_intr, cam_pose,
                boxes,
                masks, num_masks, class_info,
                first_masking, frame_num,
                obs_weight=1.):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign for the current observation. A higher
        value
    """
    self.im_h, self.im_w = depth_im.shape
    self.cam_intr, self.cam_pose = cam_intr, cam_pose
    self.color_im = color_im.copy()
    self.masked_img = color_im.copy()

    self.good_matches = []

    # Fold RGB color image into a single channel image
    color_im = color_im.astype(np.float32)
    color_im = np.floor(color_im[...,2]*self._color_const + color_im[...,1]*256 + color_im[...,0])

    self.class_info = class_info
    comp_ID = []
    if num_masks > 0:
      self.masks_test = masks.squeeze(-1).to(torch.device("cpu")).detach().numpy().astype(np.float32)
      self.mask_data = -np.ones(self.masks_test[0].shape, dtype="int32").reshape(-1)
      self.seen_class = {}
      for i in range(num_masks):
        class_value = self.class_info[i]
        class_name = class_value.split(':')[0]
        class_index = int(class_value.split(':')[1].split('_')[0])
        class_score = int(float(class_value.split(':')[1].split('_')[1])*100)-1   # set value to 0 ~ 99
        self.class_score = class_score

        # add clustering method to depth --> get more precision masks
        if (self.is_depth_clustered):
          self.depth_val = np.zeros(self.masks_test[i].reshape(-1).shape)
          self.depth_val[self.masks_test[i].reshape(-1).nonzero()] = depth_im[self.masks_test[i].nonzero()]
          self.check_depth = self.depth_val[self.depth_val.nonzero()].reshape(-1, 1)

          if (self.cluster_technique == 'kmeans'):
            self.cluster = KMeans(n_clusters=3, random_state=0).fit(self.check_depth)

          if (self.cluster_technique == 'meanshift'):
            bandwidth = estimate_bandwidth(self.check_depth, quantile=0.2, n_samples=500)
            self.cluster = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(self.check_depth)

          if (self.cluster_technique == 'ransac'):
            median = np.median(self.check_depth, axis=0)
            diff = np.sum((self.check_depth - median) ** 2, axis=-1)
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)

            modified_z_score = 0.6745 * diff / med_abs_deviation
            thresh = 3.5

            inliers = modified_z_score <= thresh

            self.clustered_index = np.zeros(self.masks_test[0].reshape(-1).shape).astype('int')
            self.clustered_index[self.depth_val.nonzero()] = inliers.astype('int')
            self.valid_pts_index = self.clustered_index.astype('bool')

          if (self.cluster_technique == 'kmeans' or self.cluster_technique == 'meanshift'):
            self.clustered_index = -np.ones(self.masks_test[0].reshape(-1).shape).astype('int')
            self.clustered_index[self.depth_val.nonzero()] = self.cluster.labels_
            self.center_label = np.argsort(self.cluster.cluster_centers_, axis=0)[0][0]  # get nearest label
            self.valid_pts_index = self.clustered_index == self.center_label


        # get color hists and mask info for 3D reconstruction
        if (first_masking):
          self.ID_2D += 1
          if (self.is_depth_clustered):
            color_pixs = self.color_im[self.valid_pts_index.reshape(self.masks_test[0].shape)]
            self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)
            self.mask_data[self.valid_pts_index] = 10000 * class_score + \
                                                   100 * class_index + \
                                                   self.ID_2D

            self.make_node_data(self.ID_2D, class_name, boxes[i, :], self.color_hist, is_new=True)

          else:
            color_pixs = self.color_im[self.masks_test[i].nonzero()]
            self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)
            self.mask_data[self.masks_test[i].reshape(-1).nonzero()] = 10000 * class_score + \
                                                                       100 * class_index + \
                                                                       self.ID_2D
            self.make_node_data(self.ID_2D, class_name, boxes[i, :], self.color_hist, is_new=True)
          comp_ID += [self.ID_2D]
        else:
          update_id = max(self.unique_ID_3D) + i + 1
          # print('update_id :{}, update_class : {}'.format(update_id, class_name))
          if (self.is_depth_clustered):
            color_pixs = self.color_im[self.valid_pts_index.reshape(self.masks_test[0].shape)]
            self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)

            self.make_node_data(update_id, class_name, boxes[i, :], self.color_hist, is_new=True)


            self.mask_data[self.valid_pts_index] = 10000 * class_score + \
                                                   100 * class_index + \
                                                   update_id
          else:
            color_pixs = self.color_im[self.masks_test[i].nonzero()]
            self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)
            self.make_node_data(update_id, class_name, boxes[i, :], self.color_hist, is_new=True)
            self.mask_data[self.masks_test[i].reshape(-1).nonzero()] = 10000 * class_score + \
                                                                       100 * class_index + \
                                                                       update_id

          comp_ID += [update_id]

    if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
      update_masked_img, update_tsdf = 1, 1
      self.cuda_integrate_param(obs_weight, num_masks, update_masked_img, update_tsdf, first_masking, color_im, depth_im)

      self.cuda_to_cpu()
      self.draw_updated_3d_pts()

      # remove dropped idx during same node detection
      pop_list = []
      for (key, val) in self.node_data.items():
        if (not int(key) in self.unique_ID_3D):
          pop_list += [key]
      for pop_ in pop_list:
        self.node_data.pop(pop_)

      # Make relation nodes data
      if (self.node_data):
        self.make_rel_data(self.node_data.keys())
        self.draw_scene_graph(frame_num)

      self.cpu_to_cuda()


    else:  # CPU mode: integrate voxel volume (vectorized implementation)
      # Convert voxel grid coordinates to pixel coordinates
      cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
      self.cam_pts = cam_pts
      cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
      pix_z = cam_pts[:, 2]
      pix = self.cam2pix(cam_pts, cam_intr, self.gpu_mode)
      self.pix = pix
      pix_x, pix_y = pix[:, 0], pix[:, 1]
      self.pix_x, self.pix_y = pix_x, pix_y

      # Eliminate pixels outside view frustum
      valid_pix = np.logical_and(pix_x >= 0,
                  np.logical_and(pix_x < self.im_w,
                  np.logical_and(pix_y >= 0,
                  np.logical_and(pix_y <self.im_h,
                  pix_z > 0))))
      self.valid_pix = valid_pix
      depth_val = np.zeros(pix_x.shape)
      depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

      # Integrate TSDF
      depth_diff = depth_val - pix_z
      valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
      self.valid_pts = valid_pts
      dist = np.minimum(1, depth_diff / self._trunc_margin)
      valid_vox_x = self.vox_coords[valid_pts, 0]
      valid_vox_y = self.vox_coords[valid_pts, 1]
      valid_vox_z = self.vox_coords[valid_pts, 2]
      self.valid_vox_x, self.valid_vox_y, self.valid_vox_z = valid_vox_x, valid_vox_y, valid_vox_z

      w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      valid_dist = dist[valid_pts]
      tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
      self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
      self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

      # Integrate color
      old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      old_b = np.floor(old_color / self._color_const)
      old_g = np.floor((old_color-old_b*self._color_const)/256)
      old_r = old_color - old_b*self._color_const - old_g*256
      new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
      self.new_color = new_color

      new_b = np.floor(new_color / self._color_const)
      new_g = np.floor((new_color - new_b*self._color_const) /256)
      new_r = new_color - new_b*self._color_const - new_g*256
      new_b = np.minimum(255., np.round((w_old*old_b + obs_weight*new_b) / w_new))
      new_g = np.minimum(255., np.round((w_old*old_g + obs_weight*new_g) / w_new))
      new_r = np.minimum(255., np.round((w_old*old_r + obs_weight*new_r) / w_new))
      self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*self._color_const + new_g*256 + new_r

      if num_masks > 0:
        # Apply Yolact Mask to find object's segmentation
        self.valid_coords = np.vstack((self.valid_vox_x,self.valid_vox_y, self.valid_vox_z)).transpose()
        valid_cam_pts = self.vox2world(self._vol_origin, self.valid_coords, self._voxel_size)
        self.valid_cam_pts = rigid_transform(valid_cam_pts, np.linalg.inv(cam_pose))
        self.valid_pix_z = self.valid_cam_pts[:, 2]

        self.mask_centers = []
        self.class_label = []

        for i in range(num_masks):
          # class info
          class_value = self.class_info[i]
          class_name = class_value.split(':')[0]
          class_index = int(class_value.split(':')[1].split('_')[0])
          class_score = float(class_value.split(':')[1].split('_')[1])

          # mask info
          self.nonzero_mask_idx_ = self.masks_test[i].nonzero()
          self.nonzero_mask_idx = np.vstack((self.nonzero_mask_idx_[0], self.nonzero_mask_idx_[1])).transpose()
          self.valid_pix, self.pix_index = self.valid_cam2pix(self.valid_cam_pts, cam_intr, self.nonzero_mask_idx)
          self.pix_index = self.pix_index.astype(bool)

          pix_x, pix_y = self.valid_pix[:, 0], self.valid_pix[:, 1]
          self.depth_val = np.zeros(pix_x.shape)
          self.depth_val[self.pix_index] = depth_im[pix_y[self.pix_index], pix_x[self.pix_index]]
          self.depth_diff = self.valid_pix_z - self.depth_val   # origin
          # self.depth_diff = self.depth_val - self.valid_pix_z
          self.valid_pts_index = np.logical_and(self.depth_val > 0, self.depth_diff >= -self._trunc_margin)
          self.dist = np.minimum(1, self.depth_diff / self._trunc_margin)

          # add clustering method to depth --> get more precision masks
          self.check_depth = self.depth_val[self.valid_pts_index].reshape(-1, 1)
          self.kmeans = KMeans(n_clusters=3, random_state=0).fit(self.check_depth)
          self.clustered_index = -np.ones(self.valid_pts_index.shape).astype('int')
          self.clustered_index[self.valid_pts_index] = self.kmeans.labels_
          self.center_label = np.argsort(self.kmeans.cluster_centers_, axis=0)[0][0]  # get center label
          self.valid_pts_index = np.logical_and(self.valid_pts_index, self.clustered_index == self.center_label)

          # update changed mask img
          masked_pix_y, masked_pix_x = pix_y[self.valid_pts_index], pix_x[self.valid_pts_index]
          color_idx = (i * 5) % len(self.GCI.COLORS)
          color = self.GCI.COLORS[color_idx]
          color = np.array([color[2], color[1], color[0]], dtype='uint8')
          self.masked_img[masked_pix_y, masked_pix_x] = color
          # self.masked_img[masked_pix_y, masked_pix_x] = np.array([255, 0, 0], dtype='uint8')


          self.valid_vox = self.valid_coords[self.valid_pts_index, :]
          self.valid_vox_x = self.valid_coords[self.valid_pts_index, 0]
          self.valid_vox_y = self.valid_coords[self.valid_pts_index, 1]
          self.valid_vox_z = self.valid_coords[self.valid_pts_index, 2]

          # add mask color to 3D voxel space
          self.mask_color = self.class_colors[class_index]
          new_color = np.array([255*255*self.mask_color[0] + 255*self.mask_color[1] + self.mask_color[2]] * self.valid_vox_x.shape[0])
          self.new_color = new_color
          new_b = np.floor(new_color / self._color_const)
          new_g = np.floor((new_color - new_b * self._color_const) / 256)
          new_r = new_color - new_b * self._color_const - new_g * 256

          self._color_vol_cpu[self.valid_vox_x, self.valid_vox_y, self.valid_vox_z] = new_b * self._color_const + new_g * 256 + new_r

          self.real_valid_vox = self.valid_vox * self._voxel_size + self._vol_origin.reshape(-1, 3)
          self.curr_mean = np.mean(self.real_valid_vox, axis=0)
          self.mean_x, self.mean_y, self.mean_z = self.curr_mean[0], self.curr_mean[1], self.curr_mean[2]
          self.curr_var = np.var(self.real_valid_vox, axis=0)

          # self.mean_x, self.mean_y, self.mean_z = np.mean(self.valid_vox_x * self._voxel_size + self._vol_origin[0]),\
          #                                         np.mean(self.valid_vox_y * self._voxel_size + self._vol_origin[1]),\
          #                                         np.mean(self.valid_vox_z * self._voxel_size + self._vol_origin[2])

          self.pt_num = self.valid_vox_x.shape[0]
          self.verts, faces, norms, colors = self.get_mesh()


          self.mask_centers += [[self.mean_x, self.mean_y, self.mean_z]]
          self.class_label += [class_name]
          color_pixs = self.color_im[self.nonzero_mask_idx_]
          self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)


          ''' Make scene graph node '''
          # First for testing, make scene graph for each images
          if (bool(self.node_data)):
            # make consecutive node data
            node_score, max_score_index = self.SND.node_update(self.node_data, self.curr_mean,
                                                              self.curr_var, self.pt_num, class_name, class_score, self.color_hist)
            # Same Node detection
            # if same node
            if (node_score > self.SND.node_th):
              box_id = str(list(self.node_data.keys())[max_score_index])
              prev_mean = self.node_data[box_id]['mean']
              prev_var = self.node_data[box_id]['var']
              prev_pt_num = self.node_data[box_id]['pt_num']
              new_pts = self.valid_vox
              total_pt_num = self.pt_num + prev_pt_num
              # update mean, variance, pt_num
              updated_mean = (prev_mean * prev_pt_num + self.curr_mean * self.pt_num) / (prev_pt_num + self.pt_num)
              updated_var_x_2 = (prev_var + np.power(prev_mean, 2)) * prev_pt_num + np.sum(np.power(new_pts,2), axis=0)
              updated_var = updated_var_x_2 / total_pt_num - np.power(updated_mean, 2)

              self.node_data[box_id]['class'] = class_name
              self.node_data[box_id]['score'] = class_score
              self.node_data[box_id]['mean'] = updated_mean
              self.node_data[box_id]['var'] = updated_var
              self.node_data[box_id]['pt_num'] = total_pt_num
              self.node_data[box_id]['color_hist'] = self.color_hist    ## update?
              self.node_data[box_id]['detection_cnt'] += 1
              x1, y1, x2, y2 = boxes[i, :]
              box_im = cv2.cvtColor(self.color_im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
              thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + class_name + str(box_id) +
                                            '_' + str(self.node_data[str(box_id)]['detection_cnt']) + '.png')
              cv2.imwrite(thumbnail_path, box_im)
              self.node_data[str(box_id)]['thumbnail'] += [thumbnail_path]

            # else different node
            else:
              box_id = str(len(self.node_data))
              print("************************************new node was created : {} {}\n".format(box_id, class_name))
              self.node_data[str(box_id)] = {}
              self.node_data[str(box_id)]['class'] = class_name
              self.node_data[str(box_id)]['score'] = class_score
              self.node_data[str(box_id)]['mean'] = self.curr_mean
              self.node_data[str(box_id)]['var'] = self.curr_var
              self.node_data[str(box_id)]['pt_num'] = self.pt_num
              self.node_data[str(box_id)]['detection_cnt'] = 1
              x1, y1, x2, y2 = boxes[i, :]
              box_im = cv2.cvtColor(self.color_im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
              thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + class_name + str(box_id) +
                                            '_' + str(self.node_data[str(box_id)]['detection_cnt']) + '.png')
              cv2.imwrite(thumbnail_path, box_im)
              self.node_data[str(box_id)]['thumbnail'] = [thumbnail_path]
              self.node_data[str(box_id)]['color_hist'] = self.color_hist

          else:
            # make first node data
            box_id = i
            self.node_data[str(box_id)] = {}
            self.node_data[str(box_id)]['class'] = class_name
            self.node_data[str(box_id)]['score'] = class_score
            self.node_data[str(box_id)]['mean'] = self.curr_mean
            self.node_data[str(box_id)]['var'] = self.curr_var
            self.node_data[str(box_id)]['pt_num'] = self.pt_num
            self.node_data[str(box_id)]['detection_cnt'] = 1
            x1, y1, x2, y2 = boxes[i, :]
            box_im = cv2.cvtColor(self.color_im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + class_name + str(box_id) +
                                          '_' + str(self.node_data[str(box_id)]['detection_cnt'])+'.png')
            cv2.imwrite(thumbnail_path, box_im)
            self.node_data[str(box_id)]['thumbnail'] = [thumbnail_path]
            self.node_data[str(box_id)]['color_hist'] = self.color_hist


        # find camera's 3d position in tsdf_vol
        self.T = np.linalg.inv(cam_pose)
        # self.T = cam_pose
        self.R = self.T[:3, :3]
        self.t = self.T[:3, 3]
        self.cam_pose = np.matmul(-np.linalg.inv(self.R), self.t)
        # self.cam_pose = self.cam_pose * self._voxel_size + self._vol_origin
        self.mask_centers += [self.cam_pose]
        self.class_label += ['camera']

        # Make relation nodes data
        for i, (key_sub, sub) in enumerate(self.node_data.items()):
          for j, (key_obj, obj) in enumerate(self.node_data.items()):
            if (j > i):
              self.rel_data[sub['class'] + '_' + key_sub + '/' +
                            obj['class'] + '_' + key_obj] = ['some_rel']

        # Draw scene graph
        sg = Digraph('structs', format='pdf')
        detect_th = 1
        self.tomato_rgb = [236, 93, 87]
        self.blue_rgb = [81, 167, 250]
        self.pale_rgb = [112, 191, 64]
        self.tomato_hex = webcolors.rgb_to_hex(self.tomato_rgb)
        self.blue_hex = webcolors.rgb_to_hex(self.blue_rgb)
        self.pale_hex = webcolors.rgb_to_hex(self.pale_rgb)

        for node_idx in self.node_data.keys():
          idx = str(node_idx)
          node = self.node_data[idx]
          obj_cls = node['class']
          detect_num  = node['detection_cnt']
          if (detect_num >= detect_th):
            sg.node(obj_cls+'_'+idx, shape='box', style='filled, rounded', label=obj_cls + '_' + idx,
                    margin='0.11, 0.0001', width='0.11', height='0', fillcolor=self.tomato_hex,
                    fontcolor = 'black')
            sg.node('attribute_pose_'+idx, shape='box', style='filled, rounded',
                    label = '('+str(round(node['mean'][0], 2))+','+
                                str(round(node['mean'][1], 2))+','+
                                str(round(node['mean'][2], 2))+')',
                    margin='0.11,0.0001', width='0.11', height='0',
                    fillcolor=self.blue_hex, fontcolor='black'
                    )
            sg.node('attribute_color_'+idx, shape='box', style='filled, rounded',
                    label = str(node['color_hist'][0][1]),
                    margin='0.11, 0.0001', width='0.11', height='0',
                    fillcolor=node['color_hist'][0][0], fontcolor='black' )
            sg.node('thumbnail_'+idx, shape='box', label='.',
                    image=node['thumbnail'][int(len(node['thumbnail'])/2)])
            # For one shot image
            # sg.node('thumbnail_' + idx, shape='box', label='.',
            #         image=node['thumbnail'][0])
            sg.edge(obj_cls+'_'+idx, 'attribute_pose_'+idx)
            sg.edge(obj_cls+'_'+idx, 'attribute_color_'+idx)
            sg.edge(obj_cls+'_'+idx, 'thumbnail_'+idx)

        for i, (key, value) in enumerate(self.rel_data.items()):
            sub = key.split('/')[0]
            obj = key.split('/')[1]
            sub_id = sub.split('_')[1]
            obj_id = obj.split('_')[1]
            # Draw scene graph relation if objects detected more than detect_th
            if ((self.node_data[str(sub_id)]['detection_cnt'] >= detect_th) and (self.node_data[str(obj_id)]['detection_cnt'] >= detect_th)):
                if (value):
                    # check value has some info
                    rel = ''
                    for v in range(len(value)):
                        if (v != len(value)-1):
                            rel = rel + value[v] + ','
                        else:
                            rel += value[v]
                    sg.node('rel'+str(i), shape='box', style= 'filled, rounded', fillcolor= self.pale_hex,
                           fontcolor= 'black', margin = '0.11, 0.0001', width = '0.11', height='0',
                           label= rel)
                    sg.edge(sub, 'rel'+str(i))
                    sg.edge('rel' + str(i), obj)
        # sg.render(os.path.join(save_path, 'scene_graph'+str(f_idx)), view=True)
        sg.format = 'png'
        #sg.size = "480,640"
        sg.render(os.path.join(self.scene_graph_path, 'scene_graph' + str(self.f_idx)), view=False)
        self.f_idx += 1

  def integrate_remain(self, color_im, depth_im, cam_intr, cam_pose,
                      boxes,
                      masks, masks_color, num_masks, class_info,
                      prev_color_im, prev_masks, prev_num_dets_to_consider, prev_text_str,
                      obs_weight=1.):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign for the current observation. A higher
        value
    """
    self.im_h, self.im_w = depth_im.shape
    self.cam_intr, self.cam_pose = cam_intr, cam_pose
    self.color_im = color_im.copy()
    self.masked_img = color_im.copy()
    self.class_info = class_info
    if num_masks > 0:
      self.masks_test = masks.squeeze(-1).to(torch.device("cpu")).detach().numpy().astype(np.float32)

    self.prev_color_im = prev_color_im

    # Fold RGB color image into a single channel image
    color_im = color_im.astype(np.float32)
    color_im = np.floor(color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[..., 0])

    if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
      update_masked_img, update_tsdf = 0, 1
      self.cuda_integrate_param(obs_weight, num_masks, update_masked_img, update_tsdf, color_im, depth_im)

      # self.cuda_to_cpu()
      # self.wide_view_img, _, _, _ = self.widen_camera_view(self._color_vol_cpu.copy(), 'color')
      # self.wide_class_mask, \
      # self.wide_view_idx, \
      # self.wide_view_label,\
      # self.wide_view_score= self.widen_camera_view(self._class_vol_cpu.copy(), 'class')
      # self.cpu_to_cuda()

      comp_ID = []
      if num_masks > 0:
        self.mask_data = -np.ones(self.masks_test[0].shape, dtype="int32").reshape(-1)
        self.seen_class = {}
        for i in range(num_masks):
          class_value = self.class_info[i]
          class_name = class_value.split(':')[0]
          class_index = int(class_value.split(':')[1].split('_')[0])
          class_score = int(float(class_value.split(':')[1].split('_')[1]) * 100) - 1  # set value to 0 ~ 99
          self.class_score = class_score

          # add clustering method to depth --> get more precision masks
          if (self.is_depth_clustered and class_name in self.cluster_object):
            self.depth_val = np.zeros(self.masks_test[i].reshape(-1).shape)
            self.depth_val[self.masks_test[i].reshape(-1).nonzero()] = depth_im[self.masks_test[i].nonzero()]
            self.check_depth = self.depth_val[self.depth_val.nonzero()].reshape(-1, 1)

            if (self.cluster_technique == 'kmeans'):
              self.cluster = KMeans(n_clusters=3, random_state=0).fit(self.check_depth)
            if (self.cluster_technique == 'meanshift'):
              bandwidth = estimate_bandwidth(self.check_depth, quantile=0.2, n_samples=500)
              self.cluster = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(self.check_depth)

            self.clustered_index = -np.ones(self.masks_test[0].reshape(-1).shape).astype('int')
            self.clustered_index[self.depth_val.nonzero()] = self.cluster.labels_
            self.center_label = np.argsort(self.cluster.cluster_centers_, axis=0)[0][0]  # get nearest label
            self.valid_pts_index = self.clustered_index == self.center_label

          # get color hists and mask info for 3D reconstruction
          if (self.is_depth_clustered and class_name in self.cluster_object):
            color_pixs = self.color_im[self.valid_pts_index.reshape(self.masks_test[0].shape)]
            self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)
            self.mask_data[self.valid_pts_index] = 10000 * class_score + \
                                                   100 * class_index + \
                                                   self.ID_2D
          else:
            color_pixs = self.color_im[self.masks_test[i].nonzero()]
            self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)
            self.mask_data[self.masks_test[i].reshape(-1).nonzero()] = 10000 * class_score + \
                                                                       100 * class_index + \
                                                                       self.ID_2D


          # # Same Node detection!
          # if (1.0 >= overlapped_ratio >= 0.15):
          #   # update to previous idx
          #   update_id = Counter(self.prev_idxs).most_common()[0][0]
          #   update_labels = Counter(self.prev_labels).most_common()[0][0]
          #   update_score = self.prev_score[0]
          #   if (update_labels != class_index):
          #     # check overlapped areas
          #     self.ID_2D += 1
          #     update_id = self.ID_2D
          #     update_labels = class_index
          #     update_score = class_score
          #
          #     self.make_node_data(self.ID_2D, self.cfg.dataset.class_names[update_labels],
          #                         update_score, boxes[i, :], self.color_hist, is_new=True)
          #   else:
          #     self.make_node_data(update_id, self.cfg.dataset.class_names[update_labels],
          #                         update_score, boxes[i, :], self.color_hist, is_new=False)
          # else:
          #   # add new ids
          #   self.ID_2D += 1
          #   update_id = self.ID_2D
          #   update_labels = class_index
          #   update_score = class_score
          #
          #   self.make_node_data(self.ID_2D, self.cfg.dataset.class_names[update_labels],
          #                       update_score, boxes[i, :], self.color_hist, is_new=True)

          # # update mask info for 3D reconstruction
          # if (self.is_depth_clustered and class_name in self.cluster_object):
          #   self.mask_data[self.valid_pts_index] = 10000 * update_score + \
          #                                          100 * update_labels + \
          #                                          update_id
          # else:
          #   self.mask_data[self.masks_test[i].reshape(-1).nonzero()] = 10000 * update_score + \
          #                                                              100 * update_labels + \
          #                                                              update_id
          #
          # comp_ID += [update_id]


        # update mask values
        update_masked_img, update_tsdf = 1, 0
        self.cuda_integrate_param(obs_weight, num_masks, update_masked_img, update_tsdf, color_im, depth_im)

      self.cuda_to_cpu()
      self.prev_mask_data = self.mask_data.copy()

      # Draw 3D points and mask info in 3D vispy system
      self.draw_updated_3d_pts()


      # # remove dropped idx during same node detection
      # pop_list = []
      # for (key, val) in self.node_data.items():
      #   if (not 'mean' in val.keys()):
      #     pop_list += [key]
      # for pop_ in pop_list:
      #   self.node_data.pop(pop_)
      #
      # # Make relation nodes data
      # self.make_rel_data(comp_ID)
      #
      # # Draw scene graph
      # self.draw_scene_graph()
      self.cpu_to_cuda()

    else:  # CPU mode: integrate voxel volume (vectorized implementation)
      # Convert voxel grid coordinates to pixel coordinates
      cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
      self.cam_pts = cam_pts
      cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
      pix_z = cam_pts[:, 2]
      pix = self.cam2pix(cam_pts, cam_intr, self.gpu_mode)
      self.pix = pix
      pix_x, pix_y = pix[:, 0], pix[:, 1]
      self.pix_x, self.pix_y = pix_x, pix_y

      # Eliminate pixels outside view frustum
      valid_pix = np.logical_and(pix_x >= 0,
                                 np.logical_and(pix_x < self.im_w,
                                                np.logical_and(pix_y >= 0,
                                                               np.logical_and(pix_y < self.im_h,
                                                                              pix_z > 0))))
      self.valid_pix = valid_pix
      depth_val = np.zeros(pix_x.shape)
      depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

      # Integrate TSDF
      depth_diff = depth_val - pix_z
      valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
      self.valid_pts = valid_pts
      dist = np.minimum(1, depth_diff / self._trunc_margin)
      valid_vox_x = self.vox_coords[valid_pts, 0]
      valid_vox_y = self.vox_coords[valid_pts, 1]
      valid_vox_z = self.vox_coords[valid_pts, 2]
      self.valid_vox_x, self.valid_vox_y, self.valid_vox_z = valid_vox_x, valid_vox_y, valid_vox_z

      w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      valid_dist = dist[valid_pts]
      tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
      self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
      self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

      # Integrate color
      old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      old_b = np.floor(old_color / self._color_const)
      old_g = np.floor((old_color - old_b * self._color_const) / 256)
      old_r = old_color - old_b * self._color_const - old_g * 256
      new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
      self.new_color = new_color

      new_b = np.floor(new_color / self._color_const)
      new_g = np.floor((new_color - new_b * self._color_const) / 256)
      new_r = new_color - new_b * self._color_const - new_g * 256
      new_b = np.minimum(255., np.round((w_old * old_b + obs_weight * new_b) / w_new))
      new_g = np.minimum(255., np.round((w_old * old_g + obs_weight * new_g) / w_new))
      new_r = np.minimum(255., np.round((w_old * old_r + obs_weight * new_r) / w_new))
      self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b * self._color_const + new_g * 256 + new_r

      if num_masks > 0:
        # Apply Yolact Mask to find object's segmentation
        self.valid_coords = np.vstack((self.valid_vox_x, self.valid_vox_y, self.valid_vox_z)).transpose()
        valid_cam_pts = self.vox2world(self._vol_origin, self.valid_coords, self._voxel_size)
        self.valid_cam_pts = rigid_transform(valid_cam_pts, np.linalg.inv(cam_pose))
        self.valid_pix_z = self.valid_cam_pts[:, 2]

        self.mask_centers = []
        self.class_label = []

        for i in range(num_masks):
          # class info
          class_value = self.class_info[i]
          class_name = class_value.split(':')[0]
          class_index = int(class_value.split(':')[1].split('_')[0])
          class_score = float(class_value.split(':')[1].split('_')[1])

          # mask info
          self.nonzero_mask_idx_ = self.masks_test[i].nonzero()
          self.nonzero_mask_idx = np.vstack((self.nonzero_mask_idx_[0], self.nonzero_mask_idx_[1])).transpose()
          self.valid_pix, self.pix_index = self.valid_cam2pix(self.valid_cam_pts, cam_intr, self.nonzero_mask_idx)
          self.pix_index = self.pix_index.astype(bool)

          pix_x, pix_y = self.valid_pix[:, 0], self.valid_pix[:, 1]
          self.depth_val = np.zeros(pix_x.shape)
          self.depth_val[self.pix_index] = depth_im[pix_y[self.pix_index], pix_x[self.pix_index]]
          self.depth_diff = self.valid_pix_z - self.depth_val  # origin
          # self.depth_diff = self.depth_val - self.valid_pix_z
          self.valid_pts_index = np.logical_and(self.depth_val > 0, self.depth_diff >= -self._trunc_margin)
          self.dist = np.minimum(1, self.depth_diff / self._trunc_margin)

          # add clustering method to depth --> get more precision masks
          self.check_depth = self.depth_val[self.valid_pts_index].reshape(-1, 1)
          self.kmeans = KMeans(n_clusters=3, random_state=0).fit(self.check_depth)
          self.clustered_index = -np.ones(self.valid_pts_index.shape).astype('int')
          self.clustered_index[self.valid_pts_index] = self.kmeans.labels_
          self.center_label = np.argsort(self.kmeans.cluster_centers_, axis=0)[0][0]  # get center label
          self.valid_pts_index = np.logical_and(self.valid_pts_index, self.clustered_index == self.center_label)

          # update changed mask img
          masked_pix_y, masked_pix_x = pix_y[self.valid_pts_index], pix_x[self.valid_pts_index]
          color_idx = (i * 5) % len(self.GCI.COLORS)
          color = self.GCI.COLORS[color_idx]
          color = np.array([color[2], color[1], color[0]], dtype='uint8')
          self.masked_img[masked_pix_y, masked_pix_x] = color
          # self.masked_img[masked_pix_y, masked_pix_x] = np.array([255, 0, 0], dtype='uint8')

          self.valid_vox = self.valid_coords[self.valid_pts_index, :]
          self.valid_vox_x = self.valid_coords[self.valid_pts_index, 0]
          self.valid_vox_y = self.valid_coords[self.valid_pts_index, 1]
          self.valid_vox_z = self.valid_coords[self.valid_pts_index, 2]

          # add mask color to 3D voxel space
          self.mask_color = self.class_colors[class_index]
          new_color = np.array(
            [255 * 255 * self.mask_color[0] + 255 * self.mask_color[1] + self.mask_color[2]] * self.valid_vox_x.shape[
              0])
          self.new_color = new_color
          new_b = np.floor(new_color / self._color_const)
          new_g = np.floor((new_color - new_b * self._color_const) / 256)
          new_r = new_color - new_b * self._color_const - new_g * 256

          self._color_vol_cpu[
            self.valid_vox_x, self.valid_vox_y, self.valid_vox_z] = new_b * self._color_const + new_g * 256 + new_r

          self.real_valid_vox = self.valid_vox * self._voxel_size + self._vol_origin.reshape(-1, 3)
          self.curr_mean = np.mean(self.real_valid_vox, axis=0)
          self.mean_x, self.mean_y, self.mean_z = self.curr_mean[0], self.curr_mean[1], self.curr_mean[2]
          self.curr_var = np.var(self.real_valid_vox, axis=0)

          # self.mean_x, self.mean_y, self.mean_z = np.mean(self.valid_vox_x * self._voxel_size + self._vol_origin[0]),\
          #                                         np.mean(self.valid_vox_y * self._voxel_size + self._vol_origin[1]),\
          #                                         np.mean(self.valid_vox_z * self._voxel_size + self._vol_origin[2])

          self.pt_num = self.valid_vox_x.shape[0]
          self.verts, faces, norms, colors = self.get_mesh()

          self.mask_centers += [[self.mean_x, self.mean_y, self.mean_z]]
          self.class_label += [class_name]
          color_pixs = self.color_im[self.nonzero_mask_idx_]
          self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)

          ''' Make scene graph node '''
          # First for testing, make scene graph for each images
          if (bool(self.node_data)):
            # make consecutive node data
            node_score, max_score_index = self.SND.node_update(self.node_data, self.curr_mean,
                                                               self.curr_var, self.pt_num, class_name, class_score,
                                                               self.color_hist)
            # Same Node detection
            # if same node
            if (node_score > self.SND.node_th):
              box_id = str(list(self.node_data.keys())[max_score_index])
              prev_mean = self.node_data[box_id]['mean']
              prev_var = self.node_data[box_id]['var']
              prev_pt_num = self.node_data[box_id]['pt_num']
              new_pts = self.valid_vox
              total_pt_num = self.pt_num + prev_pt_num
              # update mean, variance, pt_num
              updated_mean = (prev_mean * prev_pt_num + self.curr_mean * self.pt_num) / (prev_pt_num + self.pt_num)
              updated_var_x_2 = (prev_var + np.power(prev_mean, 2)) * prev_pt_num + np.sum(np.power(new_pts, 2), axis=0)
              updated_var = updated_var_x_2 / total_pt_num - np.power(updated_mean, 2)

              self.node_data[box_id]['class'] = class_name
              self.node_data[box_id]['score'] = class_score
              self.node_data[box_id]['mean'] = updated_mean
              self.node_data[box_id]['var'] = updated_var
              self.node_data[box_id]['pt_num'] = total_pt_num
              self.node_data[box_id]['color_hist'] = self.color_hist  ## update?
              self.node_data[box_id]['detection_cnt'] += 1
              x1, y1, x2, y2 = boxes[i, :]
              box_im = cv2.cvtColor(self.color_im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
              thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + class_name + str(box_id) +
                                            '_' + str(self.node_data[str(box_id)]['detection_cnt']) + '.png')
              cv2.imwrite(thumbnail_path, box_im)
              self.node_data[str(box_id)]['thumbnail'] += [thumbnail_path]

            # else different node
            else:
              box_id = str(len(self.node_data))
              print("************************************new node was created : {} {}\n".format(box_id, class_name))
              self.node_data[str(box_id)] = {}
              self.node_data[str(box_id)]['class'] = class_name
              self.node_data[str(box_id)]['score'] = class_score
              self.node_data[str(box_id)]['mean'] = self.curr_mean
              self.node_data[str(box_id)]['var'] = self.curr_var
              self.node_data[str(box_id)]['pt_num'] = self.pt_num
              self.node_data[str(box_id)]['detection_cnt'] = 1
              x1, y1, x2, y2 = boxes[i, :]
              box_im = cv2.cvtColor(self.color_im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
              thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + class_name + str(box_id) +
                                            '_' + str(self.node_data[str(box_id)]['detection_cnt']) + '.png')
              cv2.imwrite(thumbnail_path, box_im)
              self.node_data[str(box_id)]['thumbnail'] = [thumbnail_path]
              self.node_data[str(box_id)]['color_hist'] = self.color_hist

          else:
            # make first node data
            box_id = i
            self.node_data[str(box_id)] = {}
            self.node_data[str(box_id)]['class'] = class_name
            self.node_data[str(box_id)]['score'] = class_score
            self.node_data[str(box_id)]['mean'] = self.curr_mean
            self.node_data[str(box_id)]['var'] = self.curr_var
            self.node_data[str(box_id)]['pt_num'] = self.pt_num
            self.node_data[str(box_id)]['detection_cnt'] = 1
            x1, y1, x2, y2 = boxes[i, :]
            box_im = cv2.cvtColor(self.color_im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + class_name + str(box_id) +
                                          '_' + str(self.node_data[str(box_id)]['detection_cnt']) + '.png')
            cv2.imwrite(thumbnail_path, box_im)
            self.node_data[str(box_id)]['thumbnail'] = [thumbnail_path]
            self.node_data[str(box_id)]['color_hist'] = self.color_hist

        # find camera's 3d position in tsdf_vol
        self.T = np.linalg.inv(cam_pose)
        # self.T = cam_pose
        self.R = self.T[:3, :3]
        self.t = self.T[:3, 3]
        self.cam_pose = np.matmul(-np.linalg.inv(self.R), self.t)
        # self.cam_pose = self.cam_pose * self._voxel_size + self._vol_origin
        self.mask_centers += [self.cam_pose]
        self.class_label += ['camera']

        # Make relation nodes data
        for i, (key_sub, sub) in enumerate(self.node_data.items()):
          for j, (key_obj, obj) in enumerate(self.node_data.items()):
            if (j > i):
              self.rel_data[sub['class'] + '_' + key_sub + '/' +
                            obj['class'] + '_' + key_obj] = ['some_rel']

        # Draw scene graph
        sg = Digraph('structs', format='pdf')
        detect_th = 1
        self.tomato_rgb = [236, 93, 87]
        self.blue_rgb = [81, 167, 250]
        self.pale_rgb = [112, 191, 64]
        self.tomato_hex = webcolors.rgb_to_hex(self.tomato_rgb)
        self.blue_hex = webcolors.rgb_to_hex(self.blue_rgb)
        self.pale_hex = webcolors.rgb_to_hex(self.pale_rgb)

        for node_idx in self.node_data.keys():
          idx = str(node_idx)
          node = self.node_data[idx]
          obj_cls = node['class']
          detect_num = node['detection_cnt']
          if (detect_num >= detect_th):
            sg.node(obj_cls + '_' + idx, shape='box', style='filled, rounded', label=obj_cls + '_' + idx,
                    margin='0.11, 0.0001', width='0.11', height='0', fillcolor=self.tomato_hex,
                    fontcolor='black')
            sg.node('attribute_pose_' + idx, shape='box', style='filled, rounded',
                    label='(' + str(round(node['mean'][0], 2)) + ',' +
                          str(round(node['mean'][1], 2)) + ',' +
                          str(round(node['mean'][2], 2)) + ')',
                    margin='0.11,0.0001', width='0.11', height='0',
                    fillcolor=self.blue_hex, fontcolor='black'
                    )
            sg.node('attribute_color_' + idx, shape='box', style='filled, rounded',
                    label=str(node['color_hist'][0][1]),
                    margin='0.11, 0.0001', width='0.11', height='0',
                    fillcolor=node['color_hist'][0][0], fontcolor='black')
            sg.node('thumbnail_' + idx, shape='box', label='.',
                    image=node['thumbnail'][int(len(node['thumbnail']) / 2)])
            # For one shot image
            # sg.node('thumbnail_' + idx, shape='box', label='.',
            #         image=node['thumbnail'][0])
            sg.edge(obj_cls + '_' + idx, 'attribute_pose_' + idx)
            sg.edge(obj_cls + '_' + idx, 'attribute_color_' + idx)
            sg.edge(obj_cls + '_' + idx, 'thumbnail_' + idx)

        for i, (key, value) in enumerate(self.rel_data.items()):
          sub = key.split('/')[0]
          obj = key.split('/')[1]
          sub_id = sub.split('_')[1]
          obj_id = obj.split('_')[1]
          # Draw scene graph relation if objects detected more than detect_th
          if ((self.node_data[str(sub_id)]['detection_cnt'] >= detect_th) and (
                  self.node_data[str(obj_id)]['detection_cnt'] >= detect_th)):
            if (value):
              # check value has some info
              rel = ''
              for v in range(len(value)):
                if (v != len(value) - 1):
                  rel = rel + value[v] + ','
                else:
                  rel += value[v]
              sg.node('rel' + str(i), shape='box', style='filled, rounded', fillcolor=self.pale_hex,
                      fontcolor='black', margin='0.11, 0.0001', width='0.11', height='0',
                      label=rel)
              sg.edge(sub, 'rel' + str(i))
              sg.edge('rel' + str(i), obj)
        # sg.render(os.path.join(save_path, 'scene_graph'+str(f_idx)), view=True)
        sg.format = 'png'
        # sg.size = "480,640"
        sg.render(os.path.join(self.scene_graph_path, 'scene_graph' + str(self.f_idx)), view=False)
        self.f_idx += 1


  def cuda_to_cpu(self):
    cuda.memcpy_dtoh(self._class_vol_cpu, self._class_vol_gpu)
    cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
    self._class_vol_gpu.free()
    self._color_vol_gpu.free()

  def cpu_to_cuda(self):
    self._class_vol_gpu = cuda.mem_alloc(self._class_vol_cpu.nbytes)
    cuda.memcpy_htod(self._class_vol_gpu, self._class_vol_cpu)
    self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
    cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

  def cuda_integrate_param(self, obs_weight, num_masks, update_masked_img, update_tsdf, fisrt_masking, color_im, depth_im):
    for gpu_loop_idx in range(self._n_gpu_loops):
      self._cuda_integrate(self._tsdf_vol_gpu,
                           self._weight_vol_gpu,
                           self._color_vol_gpu,
                           self._class_vol_gpu,
                           cuda.InOut(self.mask_data),
                           cuda.InOut(self.class_colors.reshape(-1).astype(np.float32)),
                           cuda.InOut(self._vol_dim.astype(np.float32)),
                           cuda.InOut(self._vol_origin.astype(np.float32)),
                           cuda.InOut(self.cam_intr.reshape(-1).astype(np.float32)),
                           cuda.InOut(self.cam_pose.reshape(-1).astype(np.float32)),
                           cuda.InOut(np.asarray([
                             gpu_loop_idx,
                             self._voxel_size,
                             self.im_h,
                             self.im_w,
                             self._trunc_margin,
                             obs_weight,
                             num_masks,
                             self._trunc_margin_mask,
                             update_masked_img,
                             update_tsdf,
                             fisrt_masking
                           ], np.float32)),
                           cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                           cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                           block=(self._max_gpu_threads_per_block, 1, 1),
                           grid=(
                             int(self._max_gpu_grid_dim[0]),
                             int(self._max_gpu_grid_dim[1]),
                             int(self._max_gpu_grid_dim[2]),
                           )
                           )

  def widen_camera_view(self, voxel, type):
    ''' Widden camera view x3'''
    # Convert voxel grid coordinates to current camera coordinates
    # self.vox_coords = np.array(self._color_vol_cpu.nonzero()).transpose()
    im_w, im_h = self.im_w, self.im_h

    if (type == 'color'):
      vox_coords = np.array(voxel.nonzero()).transpose()
    elif (type == 'class'):
      vox_coords = np.array((voxel+1).nonzero()).transpose()
    cam_pts = self.vox2world(self._vol_origin, vox_coords, self._voxel_size)
    cam_pts = rigid_transform(cam_pts, np.linalg.inv(self.cam_pose))
    self.points = cam_pts
    pix = self.cam2pix(cam_pts, self.cam_intr, self.gpu_mode)
    pix_x, pix_y = pix[:, 0], pix[:, 1]
    pix_z = cam_pts[:, 2]

    # Eliminate pixels outside view frustum
    valid_pix = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < im_w * 3,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < im_h * 3,
                pix_z >0
                ))))

    if (type == 'color'):
      color_val = voxel[voxel.nonzero()].flatten()
      rgb_sum = color_val[valid_pix]
      colors_b = np.floor(rgb_sum / self._color_const)
      colors_g = np.floor((rgb_sum - colors_b * self._color_const) / 256)
      colors_r = rgb_sum - colors_b * self._color_const - colors_g * 256
      colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
      colors = colors.astype(np.uint8)
      self.wide_colors = colors
      wide_view_img = np.zeros((3 * im_h, 3 * im_w, 3), dtype='uint8')
      # wide_view_img = np.zeros((im_h, im_w, 3), dtype='uint8')
      wide_view_img[pix_y[valid_pix], pix_x[valid_pix]] = colors

      return wide_view_img, None, None, None

    elif (type == 'class'):
      # class_vol is composed [detect_cnt][cls_score][cls_label][3D_instance]
      class_val = voxel[(voxel+1).nonzero()].flatten()
      class_view = class_val[valid_pix]
      class_vol_ids = (class_view).astype('int') - ((class_view / 100).astype('int') * 100)
      class_vol_label = (class_view/100).astype('int') - (class_view / 10000).astype('int') * 100
      class_vol_score = (class_view/10000).astype('int') - (class_view / 1000000).astype('int') * 100
      self.class_vol_ids = class_vol_ids
      self.class_vol_label = class_vol_label
      self.class_pix_y, self.class_pix_x, self.class_valid_pix = pix_y, pix_x, valid_pix

      freq_id = Counter(class_vol_ids).most_common()
      self.class_freq_id = freq_id

      wide_view_img = np.zeros((3 * im_h, 3 * im_w, 3), dtype='uint8')
      wide_view_idx = np.zeros((3 * im_h, 3 * im_w), dtype='uint8')
      wide_view_label = np.zeros((3 * im_h, 3 * im_w), dtype='uint8')
      wide_view_score = np.zeros((3 * im_h, 3 * im_w), dtype='uint8')

      # Bounding Box type
      is_bounding_box_type = True
      if (is_bounding_box_type):
        for (idx, num) in freq_id:
          idx_loc = np.where(idx == class_vol_ids)
          freq_labels = Counter(class_vol_label[idx_loc]).most_common()[0][0]

          min_y, max_y = min(pix_y[valid_pix][idx_loc]), max(pix_y[valid_pix][idx_loc])
          min_x, max_x = min(pix_x[valid_pix][idx_loc]), max(pix_x[valid_pix][idx_loc])
          self.value_test = [min_y, max_y, min_x, max_x]
          margin = 15
          min_y, min_x = min_y - margin, min_x - margin
          max_y, max_x = max_y + margin, max_x + margin

          colors = self.class_colors[idx]
          score = Counter(class_vol_score[idx_loc]).most_common()[0][0]

          wide_view_img[min_y:max_y+1, min_x:max_x+1] = colors
          wide_view_idx[min_y:max_y+1, min_x:max_x+1] = idx
          wide_view_label[min_y:max_y+1, min_x:max_x+1] = freq_labels
          wide_view_score[min_y:max_y+1, min_x:max_x+1] = score
      else:
        # Mask type
        colors = self.class_colors[class_vol_ids]
        wide_view_img[pix_y[valid_pix], pix_x[valid_pix]] = colors
        wide_view_idx[pix_y[valid_pix], pix_x[valid_pix]] = class_vol_ids
        wide_view_label[pix_y[valid_pix], pix_x[valid_pix]] = class_vol_label
        wide_view_score[pix_y[valid_pix], pix_x[valid_pix]] = class_vol_score

      return wide_view_img, wide_view_idx, wide_view_label, wide_view_score

  def draw_updated_3d_pts(self):
    # class_vol_cpu : [detect_cnt][cls_score][cls_label][3D_instance]
    nonzero_class_vol = self._class_vol_cpu[(self._class_vol_cpu + 1).nonzero()].copy()
    ID_3D = nonzero_class_vol.astype('int') - (nonzero_class_vol / 100).astype('int') * 100
    unique_ID_3D = list(set(ID_3D))
    self.unique_ID_3D = unique_ID_3D

    if self.debug_same_node_detector:
      self.mask_centers = []
      self.cam_frustum = []
      self.class_label = []
      self.bbox_3ds = {}
      class_vol_ids = (self._class_vol_cpu).astype('int') - ((self._class_vol_cpu / 100).astype('int') * 100)
      detect_cnt = (self._class_vol_cpu / 1000000).astype('int')
      for idx in unique_ID_3D:
        ID_index = np.where(np.logical_and(class_vol_ids == idx, detect_cnt >= 1))
        # ID_index = np.where(class_vol_ids == idx)
        self.ID_index_array = np.array(ID_index)
        if (len(ID_index[0]) != 0):
          ID_3D_pose = self.ID_index_array.transpose() * self._voxel_size + self._vol_origin.reshape(-1, 3)
          ID_3D_mean = np.mean(ID_3D_pose, axis=0)
          ID_class = int(self._class_vol_cpu[ID_index][0] / 100) - int(self._class_vol_cpu[ID_index][0] / 10000) * 100
          ID_class = self.cfg.dataset.class_names[ID_class]
          self.mask_centers += [list(ID_3D_mean)]
          self.class_label += [ID_class + str(idx)]

          min_x, min_y, min_z = np.min(ID_3D_pose, axis=0)
          max_x, max_y, max_z = np.max(ID_3D_pose, axis=0)
          self.bbox_3ds[str(idx)] = [[min_x, min_y, min_z], [max_x, min_y, min_z],
                                          [max_x, max_y, min_z], [min_x, max_y, min_z],
                                          [min_x, min_y, max_z], [max_x, min_y, max_z],
                                          [max_x, max_y, max_z], [min_x, max_y, max_z]
                                          ]

          # # for debugging the individual voxel class points
          # self.ID_3D_pose = ID_3D_pose
          # self.mask_centers += ID_3D_pose.tolist()
          # self.class_label += [ID_class + str(idx)] * ID_3D_pose.shape[0]

          # make scene graph node
          self.node_data[str(idx)]['mean'] = list(ID_3D_mean)

      # find camera's 3d position in tsdf_vol
      self.T = np.linalg.inv(self.cam_pose)
      self.R = self.T[:3, :3]
      self.t = self.T[:3, 3]
      self.camera_3D_pose = np.matmul(-np.linalg.inv(self.R), self.t)
      # self.mask_centers += [self.camera_3D_pose]
      # self.class_label += ['camera']

      self.cam_frustum += [self.camera_3D_pose]
      r1 = self.R[0] * self._voxel_size
      r2 = self.R[1] * self._voxel_size
      r3 = self.R[2] * self._voxel_size
      self.r1_ = r1 / np.sqrt(sum(r1 * r1))
      self.r2_ = r2 / np.sqrt(sum(r2 * r2))
      self.r3_ = r3 / np.sqrt(sum(r3 * r3))

      a1 = -r1 -r2 + r3 + self.camera_3D_pose
      a2 = -r1 + r2 + r3 + self.camera_3D_pose
      a3 = r1 + r2 + r3 + self.camera_3D_pose
      a4 = r1 - r2 + r3 + self.camera_3D_pose
      self.cam_frustum_plane = np.array([a1, a2, a3, a4])
      self.cam_frustum += self.cam_frustum_plane.tolist()
      self.cam_connect = np.array([[0,1], [0,2], [0,3], [0,4],[1,2],[2,3],[3,4],[4,1]], dtype=np.int32)
      self.cam_centers = [self.camera_3D_pose]
      self.cam_label = ['camera']

  def make_node_data(self, obj_id, class_name, boxes, color_hist, is_new=True):
    if is_new:
      self.node_data[str(obj_id)] = {}
      self.node_data[str(obj_id)]['class'] = class_name
      self.node_data[str(obj_id)]['detection_cnt'] = 0
      x1, y1, x2, y2 = boxes
      box_im = cv2.cvtColor(self.color_im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
      thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + str(obj_id) +
                                    '_' + str(self.node_data[str(obj_id)]['detection_cnt']) + '.png')
      cv2.imwrite(thumbnail_path, box_im)
      self.node_data[str(obj_id)]['color_hist'] = color_hist

    else:
      self.node_data[str(obj_id)]['class'] = class_name
      self.node_data[str(obj_id)]['detection_cnt'] += 1
      x1, y1, x2, y2 = boxes
      box_im = cv2.cvtColor(self.color_im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
      thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + str(obj_id) +
                                    '_' + str(self.node_data[str(obj_id)]['detection_cnt']) + '.png')
      cv2.imwrite(thumbnail_path, box_im)
      self.node_data[str(obj_id)]['color_hist'] = color_hist

  def make_rel_data(self, comp_ID):
    cam_pose = self.camera_3D_pose
    self.rel_data = {}
    for i, sub_idx in enumerate(comp_ID):
      for j, obj_idx in enumerate(comp_ID):
        rel = []
        th = self._voxel_size * 3
        if (sub_idx > obj_idx):
          if (str(sub_idx) in self.node_data.keys() and str(obj_idx) in self.node_data.keys()):
            sub = self.node_data[str(sub_idx)]
            obj = self.node_data[str(obj_idx)]
            sub_pose = sub['mean']
            obj_pose = obj['mean']
            sub_cam = sub_pose - cam_pose
            obj_cam = obj_pose - cam_pose
            s1 = np.dot(sub_cam, self.r1_)
            s2 = np.dot(sub_cam, self.r2_)
            s3 = np.dot(sub_cam, self.r3_)

            o1 = np.dot(obj_cam, self.r1_)
            o2 = np.dot(obj_cam, self.r2_)
            o3 = np.dot(obj_cam, self.r3_)

            if (s1 - o1 > th):
              rel += ['right']
            elif (s1 - o1 < -th):
              rel += ['left']
            if (s2 - o2 > th):
              rel += ['up']
            elif (s2 - o2 < -th):
              rel += ['down']
            if (s3 - o3 > th):
              rel += ['behind']
            elif (s3 - o3 < -th):
              rel += ['forward']

            self.rel_data[sub['class'] + '_' + str(sub_idx) + '/' +
                          obj['class'] + '_' + str(obj_idx)] = rel

  def draw_scene_graph(self, frame_num):
    # Draw scene graph
    sg = Digraph('structs', format='pdf')
    detect_th = 0
    tomato_rgb = [236, 93, 87]
    blue_rgb = [81, 167, 250]
    pale_rgb = [112, 191, 64]
    tomato_hex = webcolors.rgb_to_hex(tomato_rgb)
    blue_hex = webcolors.rgb_to_hex(blue_rgb)
    pale_hex = webcolors.rgb_to_hex(pale_rgb)

    for node_idx in self.node_data.keys():
      idx = str(node_idx)
      node = self.node_data[idx]
      obj_cls = node['class']
      detect_num = node['detection_cnt']
      if (detect_num >= detect_th):
        sg.node(obj_cls + '_' + idx, shape='box', style='filled, rounded', label=obj_cls + '_' + idx,
                margin='0.11, 0.0001', width='0.11', height='0', fillcolor=tomato_hex,
                fontcolor='black')
        sg.node('attribute_pose_' + idx, shape='box', style='filled, rounded',
                label='(' + str(round(node['mean'][0], 2)) + ',' +
                      str(round(node['mean'][1], 2)) + ',' +
                      str(round(node['mean'][2], 2)) + ')',
                margin='0.11,0.0001', width='0.11', height='0',
                fillcolor=blue_hex, fontcolor='black'
                )
        sg.node('attribute_color_' + idx, shape='box', style='filled, rounded',
                label=str(node['color_hist'][0][1]),
                margin='0.11, 0.0001', width='0.11', height='0',
                fillcolor=node['color_hist'][0][0], fontcolor='black')
        thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + str(idx) +
                                    '_' + str(int(self.node_data[str(idx)]['detection_cnt']/2)) + '.png')
        sg.node('thumbnail_'+idx, shape='box', label='.', image=thumbnail_path)
        # For one shot image
        # sg.node('thumbnail_' + idx, shape='box', label='.',
        #         image=node['thumbnail'][0])
        sg.edge(obj_cls + '_' + idx, 'attribute_pose_' + idx)
        sg.edge(obj_cls + '_' + idx, 'attribute_color_' + idx)
        sg.edge(obj_cls + '_' + idx, 'thumbnail_' + idx)

    for i, (key, value) in enumerate(self.rel_data.items()):
      sub = key.split('/')[0]
      obj = key.split('/')[1]
      sub_id = sub.split('_')[1]
      obj_id = obj.split('_')[1]
      # Draw scene graph relation if objects detected more than detect_th
      if ((self.node_data[str(sub_id)]['detection_cnt'] >= detect_th) and (
              self.node_data[str(obj_id)]['detection_cnt'] >= detect_th)):
        if (value):
          # check value has some info
          rel = ''
          for v in range(len(value)):
            if (v != len(value) - 1):
              rel = rel + value[v] + ','
            else:
              rel += value[v]
          sg.node('rel' + str(i), shape='box', style='filled, rounded', fillcolor=pale_hex,
                  fontcolor='black', margin='0.11, 0.0001', width='0.11', height='0',
                  label=rel)
          sg.edge(sub, 'rel' + str(i))
          sg.edge('rel' + str(i), obj)
    # sg.render(os.path.join(save_path, 'scene_graph'+str(f_idx)), view=True)
    sg.format = 'png'
    # sg.size = "480,640"
    sg.render(os.path.join(self.scene_graph_path, 'scene_graph' + str(frame_num)), view=False)
    self.f_idx += 1


  def get_volume(self):
    if self.gpu_mode:
      cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
      cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
    return self._tsdf_vol_cpu, self._color_vol_cpu

  def get_point_cloud(self):
    """Extract a point cloud from the voxel volume.
    """
    tsdf_vol, color_vol = self.get_volume()


    # Marching cubes
    verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + self._vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._color_const)
    colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
    colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    pc = np.hstack([verts, colors])
    return pc

  def get_mesh(self):
    """Compute a mesh from the voxel volume using marching cubes.
    """
    tsdf_vol, color_vol = self.get_volume()

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size+self._vol_origin  # voxel grid coordinates to world coordinates

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
    colors_b = np.floor(rgb_vals/self._color_const)
    colors_g = np.floor((rgb_vals-colors_b*self._color_const)/256)
    colors_r = rgb_vals-colors_b*self._color_const-colors_g*256
    colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
    colors = colors.astype(np.uint8)
    return verts, faces, norms, colors


def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      colors[i,0], colors[i,1], colors[i,2],
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()


def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      xyz[i, 0], xyz[i, 1], xyz[i, 2],
      rgb[i, 0], rgb[i, 1], rgb[i, 2],
    ))
