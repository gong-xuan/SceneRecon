#https://github.com/ScanNet/ScanNet
#python2 required

import os, sys

from SensorData import SensorData


def process_scene(filename, output_path, export_depth_images=1, export_color_images=1, export_poses=1, export_intrinsics=1):
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  #save path
  depth_path = os.path.join(output_path, 'depth')
  color_path = os.path.join(output_path, 'color')
  pose_path = os.path.join(output_path, 'pose')
  intr_path = os.path.join(output_path, 'intrinsic')
  # if os.path.exists(depth_path) and os.path.exists(color_path) and os.path.exists(pose_path) and os.path.exists(intr_path):
  #   return

  # load the data
  sys.stdout.write('loading %s...' % filename)
  sd = SensorData(filename)
  sys.stdout.write('loaded!\n')
  
  if export_depth_images: #and not os.path.exists(depth_path):
    sd.export_depth_images(depth_path)
  if export_color_images and not os.path.exists(color_path):
    sd.export_color_images(color_path)
  if export_poses and not os.path.exists(pose_path):
    sd.export_poses(pose_path)
  if export_intrinsics and not os.path.exists(intr_path):
    sd.export_intrinsics(intr_path)
  return

if __name__ == '__main__':
    #/data_local/xuangong/scannet/scans/scene0000_00/scene0000_00.sens    
    #/data_local/xuangong/scannet/scans/scene0000_00/
    rootpath = '/data_local/xuangong/scannet/scans'
    # rootpath = '/data_local/xuangong/scannet/scans_test'

    scenes = os.listdir(rootpath)
    total_length = len(scenes)
    print(total_length)
    for n, scene in enumerate(scenes):
      scene_path = os.path.join(rootpath, scene)
      filename = os.path.join(scene_path, scene+'.sens')
      # import pdb; pdb.set_trace()
      if os.path.exists(filename): 
        process_scene(filename, scene_path, export_depth_images=1, export_color_images=1, export_poses=1, export_intrinsics=1)
      print("Finish...", n, "/", total_length)