import numpy as np
import openmesh as om
import os

import sys
sys.path.append('../')
from KnittingEngine import *
import MeshUtility as mu


def field(mesh):
    field = mesh.points()[:, 0]
    iso = mu.pyisocurve.isocurve(
            mesh.points(),
            mesh.face_vertex_indices(),
            field,
        )
    _, edges, edge_ratios, curves = iso.extract(0.0, 1.e-5)
    mesh, source = mu.split_mesh(mesh.points(), mesh.face_vertex_indices(),
            edges[curves[0]], edge_ratios[curves[0]])

    field = mu.pygeodesic.distance_field(mesh.points(),
                                         mesh.face_vertex_indices(),
                                         source, 0.05)
    np.save(folder+'field.npy', field)
    mu.colormap_vertex_color(folder+'field.off', mesh.points(),
                             mesh.face_vertex_indices(),
                             field/field.max())

    field = geodesic_field_reorient(mesh, field, source)
    np.save(folder+'field_reorient.npy', field)
    mu.colormap_vertex_color(folder+'field_reorient.off', mesh.points(),
                             mesh.face_vertex_indices(),
                             field/field.max())


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        fb = int(sys.argv[1])
    else:
        fb = 0

    patches = ['front', 'back']
    patch = patches[fb]  # 0 or 1

    folder = '../output/mannequin_%s/' % patch
    os.makedirs(folder, exist_ok=True)

    mesh = om.read_trimesh('../input/mannequin_%s.obj' % patch)
    field(mesh)

    fn_mesh = folder+'field_reorient.off'
    fn_field = folder+'field_reorient.npy'
    col_width, row_height = 0.3, 0.2
    engine = KnittingEngine(row_height, col_width)
    engine.load_mesh(fn_mesh)
    engine.load_field(fn_field)
    engine.run(folder)
