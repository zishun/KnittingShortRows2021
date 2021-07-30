import numpy as np
import time
import openmesh as om
import pickle
from tqdm import trange
import os

import sys
sys.path.append('../')
from KnittingEngine import *
import MeshUtility as mu


def isocurve(r, v, step):
    x = -np.cos(v/r)*r
    r0 = np.sqrt(r*r-x*x)
    length = np.pi*r0
    n = int(np.ceil(length/step))+1
    pts = np.empty((n, 3))
    theta = np.arange(n)/(n-1)*np.pi
    pts[:, 0] = x
    pts[:, 1] = -r0*np.cos(theta)
    pts[:, 2] = r0*np.sin(theta)
    return pts, [np.arange(n)]


def extract_column_curves(engine, r):
    m = int(np.floor(np.pi*r/2/engine.col_width))
    values = np.arange(-m, m+1)*engine.col_width + np.pi*r/2
    num = m*2+1
    engine.col_curves = []
    for i in trange(num):
        v = values[i]
        pts, isocurve_indices = isocurve(r, v, engine.row_height/2)  # use a small step size

        same_value = []
        for piece in range(len(isocurve_indices)):
            col_curve = ColumnCurve(pts, isocurve_indices[piece], v, i)
            col_curve.resample(engine.row_height)
            same_value.append(col_curve)

        engine.col_curves.append(same_value)


def main():
    start = time.time()
    d = 15
    col_width, row_height = 0.3, 0.2
    engine = KnittingEngine(row_height, col_width)
    print('------------------------------------------------')

    # step 1
    print(time.ctime(), '- Extract column curves')
    extract_column_curves(engine, d/2)
    print(time.ctime(), '- Done')

    engine.collect_pts()
    mu.write_obj_lines(folder+'/curves_col.obj',
                             engine.pts, engine.col_edges)
    print('------------------------------------------------')

    with open(folder+'col_curves.pkl', 'wb') as f:
        pickle.dump(engine.col_curves, f)

    # step 2
    print(time.ctime(), '- Generate row edges')

    # NOTE: Use 100.0 or 0.0 and find out the difference.
    engine.fps_weight = 100.0
    engine.generate_rows()
    print(time.ctime(), '- Done')
    mu.write_obj_lines(folder+'/curves_row.obj',
                             engine.pts, engine.row_edges)
    print('------------------------------------------------')

    # step 3
    print(time.ctime(), '- Trace rows')
    engine.generate_2d_knitting_map()
    print(time.ctime(), '- Done')
    print('------------------------------------------------')
    engine.color_by_row(engine.knitting_mesh, engine.row_col_idx)
    om.write_mesh(folder+'/knitting_mesh_raw.obj', 
                  engine.knitting_mesh,
                  face_color=True)

    km = KnittingMesh().set(
            engine.knitting_mesh.points(), 
            engine.knitting_mesh.face_vertex_indices(),
            engine.row_col_idx,
            engine.pts_col,
            engine.pts_next)
    km.save(folder+'/knitting_mesh.npz')

    print(time.ctime(), '- One stroke adjustment')
    km = engine.map_one_stroke(km)
    print(time.ctime(), '- Done')
    print('------------------------------------------------')
    km = engine.map_boundary_tri2quad(km)
    km = engine.map_remove_unreferenced_vertices(km)
    km.save(folder+'/knitting_mesh_repair.npz')

    mesh = om.PolyMesh(km.v, km.f)
    engine.color_by_row(mesh, km.f_ij)
    om.write_mesh(folder+'/onestroke.obj', mesh,
                  face_color=True)

    print('Total time: %.2fs' % (time.time()-start))
    print('------------------------------------------------')


if __name__ == '__main__':
    folder = '../output/hemisphere/'
    os.makedirs(folder, exist_ok=True)

    main()
