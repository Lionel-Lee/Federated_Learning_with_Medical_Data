import trimesh
import os
import numpy as np
import random
import h5py
import sklearn
from feature_extraction import extract_feature_labels
import feature_extraction
import sys
# from feature_extraction import extract_feature_labels
SAMPLE_SIZE=10000
# VAL_SRC = '/home/ubuntu/raw_data/val'
# TRAIN_SRC = '/home/ubuntu/raw_data/train'
# VAL_DIST = '/home/ubuntu/seg_val.data'
# TRAIN_DIST = '/home/ubuntu/seg_train.data'

# TRAIN_SRC = '/Users/wmz/Downloads/Data1'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
# TRAIN_SRC =os.path.join(BASE_DIR, 'data')
# TRAIN_DIST = os.path.join(BASE_DIR, 'part_seg/hdf5_data1')

#get_ipython().system('rm -rf /home/ubuntu/ar_train.data')
#get_ipython().system('rm -rf /home/ubuntu/ar_val.data')
g_valid_labels = [0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
g_label_mapping = dict(zip(g_valid_labels, range(len(g_valid_labels))))

# In[3]:

def get_scale():
    # 80% -> 130%
    return np.random.rand() * 0.3 + 1.0 if np.random.rand() < 0.5 else 1.0 / (np.random.rand() * 0.2 + 1.0)

def get_trans_matrix():
    return trimesh.transformations.translation_matrix(5*(np.random.rand(3)-0.5))

def get_all_rot_matrix():
    matrices = []
    radians = 0.1745  # 10 degree in radian
    half_radians = radians / 2.0
    for d in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        for angle in [-radians, -half_radians, half_radians, radians]:
            matrices.append(trimesh.transformations.rotation_matrix(angle=angle, direction=d))
        matrices.append(trimesh.transformations.rotation_matrix(angle=0, direction=d))
    return matrices

def get_mesh_by_id(mesh, inds):
    new_mesh_vertices = mesh.vertices[mesh.faces[inds]].reshape(-1, 3)
    new_mesh_faces = np.arange(mesh.faces[inds].shape[0]*3).reshape(-1, 3)
    return trimesh.Trimesh(new_mesh_vertices, new_mesh_faces)

def process_folder(folder, dump_fname, name, num_augmentation=0):
    all_rot_mtx = get_all_rot_matrix()
    counter = 1
    if not os.path.exists(dump_fname):
        os.makedirs(dump_fname)
    all_file=open(os.path.join(TRAIN_DIST, name+'_train_hdf5_file_list.txt'),'w')
    #test_file=open(os.path.join(TRAIN_DIST, 'test_hdf5_file_list.txt'),'w')
    val_file=open(os.path.join(TRAIN_DIST, name+'_val_hdf5_file_list.txt'),'w')
    feas = []
    labs = []
    cats = []
    for i in os.listdir(folder):
        base_path = os.path.join(folder, i)
        if not os.path.isdir(base_path):
            continue
        for j in os.listdir(base_path):
            if j == 'l.txt':
                c = [1]
                stl_path = 'l_aligned.stl'
            elif j == 'u.txt':
                c = [0]
                stl_path = 'u_aligned.stl'
            else:
                continue

            if not os.path.exists(os.path.join(base_path, stl_path)):
                    continue

            m = trimesh.load(os.path.join(base_path, stl_path))
            ll = np.loadtxt(os.path.join(base_path, j)).reshape(-1).astype(np.int)

            # Remove all faces not with the 33 labels
            label_mask = np.isin(ll, g_valid_labels)
            m = get_mesh_by_id(m, np.arange(len(ll))[label_mask])
            ll = ll[label_mask]

            # label mapping
            ll = np.array([g_label_mapping[x] for x in ll],dtype=int)
            m.fix_normals()
            m.apply_translation(-m.centroid)
            f, l = extract_feature_labels(m, ll, SAMPLE_SIZE)
            #fi=h5py.File(dump_fname+'/Output'+str(counter//32)+'.h5','w')
            #all_file.write(dump_fname+'/Output'+str(counter)+'.h5'+'\n')
            #test_file.write(dump_fname+'/Output'+str(counter)+'.h5'+'\n')
            if c == [1]:
                for lab in range(len(l)):
                    if l[lab] == 0:
                        l[lab] = 33
            feas.append(f)
            labs.append(l)
            cats.append(c)
            if (counter%160 ==0):
                fi=h5py.File(dump_fname+'/Output_'+name+str(counter//32-1)+'.h5','w')
                fi.create_dataset('data',data=feas)
                fi.create_dataset('label',data=labs)
                fi.create_dataset('cat',data=cats)
                fi.close()
                val_file.write(dump_fname+'/Output_'+name+str(counter//32-1)+'.h5'+'\n')
                feas = []
                labs = []
                cats = []
            elif (counter % 32 ==0):
                fi=h5py.File(dump_fname+'/Output_'+name+str(counter//32-1)+'.h5','w')
                fi.create_dataset('data',data=feas)
                fi.create_dataset('label',data=labs)
                fi.create_dataset('cat',data=cats)
                fi.close()
                all_file.write(dump_fname+'/Output_'+name+str(counter//32-1)+'.h5'+'\n')
                feas = []
                labs = []
                cats = []
            counter+=1


            for _ in range(num_augmentation):
                mcpy = m.copy()
                if np.random.rand() < 0.5:
                    mcpy.apply_transform(random.sample(all_rot_mtx, 1)[0])
                    mcpy.apply_scale(get_scale())
                    mcpy.apply_transform(get_trans_matrix())

                f, l = extract_feature_labels(mcpy, ll, SAMPLE_SIZE)
                #all_file.write(dump_fname+'/Output'+str(counter)+'.h5'+'\n')
                #test_file.write(dump_fname+'/Output'+str(counter)+'.h5'+'\n')
                if c == [1]:
                    for lab in range(len(l)):
                        if l[lab] == 0:
                            l[lab] = 33
                feas.append(f)
                labs.append(l)
                cats.append(c)
                if (counter%160 ==0):
                    fi=h5py.File(dump_fname+'/Output_'+name+str(counter//32-1)+'.h5','w')
                    fi.create_dataset('data',data=feas)
                    fi.create_dataset('label',data=labs)
                    fi.create_dataset('cat',data=cats)
                    fi.close()
                    val_file.write(dump_fname+'/Output_'+name+str(counter//32-1)+'.h5'+'\n')
                    feas = []
                    labs = []
                    cats = []
                elif (counter % 32 ==0):
                    fi=h5py.File(dump_fname+'/Output_'+name+str(counter//32-1)+'.h5','w')
                    fi.create_dataset('data',data=feas)
                    fi.create_dataset('label',data=labs)
                    fi.create_dataset('cat',data=cats)
                    fi.close()
                    all_file.write(dump_fname+'/Output_'+name+str(counter//32-1)+'.h5'+'\n')
                    feas = []
                    labs = []
                    cats = []
                counter+=1
    all_file.close()
    #test_file.close()
    val_file.close()


# In[ ]:

# process_folder(VAL_SRC, VAL_DIST, 0)
clients_name_list = ['A','B','C','D','E']
for i in range(len(clients_name_list)):
    TRAIN_SRC =os.path.join(BASE_DIR, 'data/'+clients_name_list[i])
    TRAIN_DIST = os.path.join(BASE_DIR, 'part_seg/hdf5_data')
    process_folder(TRAIN_SRC, TRAIN_DIST, clients_name_list[i], 15)