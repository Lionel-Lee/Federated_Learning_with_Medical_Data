import trimesh
import numpy as np

def translate_delta_in_x(mesh, delta):
    mesh.apply_translation([delta, 0, 0])

def translate_delta_in_y(mesh, delta):
    mesh.apply_translation([0, delta, 0])

def translate_in_x(mesh):
    delta_x = np.random.uniform(-8, 8)
    mesh.apply_translation([delta_x, 0, 0])

def translate_in_y(mesh):
    delta_y = np.random.uniform(-8, 8)
    mesh.apply_translation([0, delta_y, 0])

def translate_in_xy(mesh):
    translate_in_x(mesh)
    translate_in_y(mesh)

def translate_delta(mesh, axis, delta):
    translation = [0, 0, 0]
    translation[axis] = delta
    mesh.apply_translation(translation)

def rotate_delta(mesh, axis, delta):
    d = 1 if delta > 0 else -1
    direction = [0, 0, 0]
    direction[axis] = d
    rotate_z = trimesh.transformations.rotation_matrix(angle=np.abs(delta), direction=direction)
    mesh.apply_transform(rotate_z)

def rotate_in_z(mesh):
    # 10 in degree = 0.174533 radians
    angle = np.random.uniform(-0.174533, 0.174533)
    d = 1 if angle > 0 else -1
    rotate_z = trimesh.transformations.rotation_matrix(angle=np.abs(angle), direction=(0,0,d))
    mesh.apply_transform(rotate_z)

def random_rotate(mesh, axis):
    delta = np.random.uniform(-0.174533*2, 0.174533*2)
    d = 1 if delta > 0 else -1
    direction = [0, 0, 0]
    direction[axis] = d
    rotate_m = trimesh.transformations.rotation_matrix(angle=np.abs(delta), direction=direction)
    mesh.apply_transform(rotate_m)

def generate_random_delta(l, u):
    return np.random.randint(l, u)*np.random.choice([-1,1])

def get_triangle_struct(mesh):
    # each face has three vertices v1, v2, v3, and a center c
    # get (v1-c), (v2-c), (v3-c)
    return (mesh.triangles - np.expand_dims(mesh.triangles_center, axis=1)).reshape(-1, 9)

def extract_feature_labels(mesh, label, sample_size=3000):
    points = mesh.triangles_center
    structs = get_triangle_struct(mesh)
    fnormals = mesh.face_normals
    label = np.reshape(label, (-1))
    assert points.shape[0] == len(label), 'mesh and label does not match'

    features = np.hstack((
        points, # center, 3
        structs, # triangle structure 9
        fnormals, # normal, 3
    ))

    if sample_size > points.shape[0]:
        # do augmentation
        centi = (mesh.triangles + np.expand_dims(mesh.triangles_center, axis=1)) / 2.0
        features1 = np.hstack((
            centi[:, 0, :], # center, 3
            structs, # triangle structure 9
            fnormals, # normal, 3
        ))
        features2 = np.hstack((
            centi[:, 1, :], # center, 3
            structs, # triangle structure 9
            fnormals, # normal, 3
        ))
        features3 = np.hstack((
            centi[:, 2, :], # center, 3
            structs, # triangle structure 9
            fnormals, # normal, 3
        ))

        label = np.tile(label, 4)
        features = np.vstack((features, features1, features2, features3))
    sampled_indices = np.random.choice(np.arange(len(label)), replace=False, size=sample_size)
    return features[sampled_indices], label[sampled_indices]