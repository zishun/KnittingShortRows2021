import numpy as np
import networkx as nx
import openmesh as om
from scipy.spatial import cKDTree
import os

import sys
sys.path.append('../')
from KnittingEngine import *
import MeshUtility as mu


class ValidGeodesicField:

    def __init__(self, folder=None):
        self.folder = folder

    def run(self, mesh, src_pts):
        self.cnt = 0
        field = self.compute_field(mesh, src_pts)
        while True:
            x = self.find_local_maxima(mesh, field)
            if x < 0:
                break
            print('%d Found local maxima %d' % (self.cnt, x))
            mesh = self.cut_to_boundary(mesh, x, method='graph')
            self.cnt += 1
            field = self.compute_field(mesh, src_pts)
        return mesh, field

    def run2(self, mesh, src_pts):
        self.cnt = 0
        field = self.compute_field(mesh, src_pts)
        while True:
            indices, values = self.find_all_local_maxima(mesh, field)
            if len(indices) == 0:
                break
            x = indices[np.argmin(np.array(values))]
            print('%d Found local maxima %d' % (self.cnt, x))
            mesh = self.cut_to_boundary(mesh, x, method='mmp')
            self.cnt += 1
            field = self.compute_field(mesh, src_pts)
        return mesh, field

    def compute_field(self, mesh, src_pts):
        tree = cKDTree(mesh.points())
        idx = tree.query(src_pts)[1]
        field = mu.pygeodesic.distance_field(mesh.points(),
                                             mesh.face_vertex_indices(),
                                             idx, 0.05)
        if self.folder is not None:
            mu.colormap_vertex_color(self.folder+'field_%d.off' % self.cnt,
                                     mesh.points(),
                                     mesh.face_vertex_indices(),
                                     field/field.max())
        return field

    def find_local_maxima(self, mesh, field):
        vv = mesh.vv_indices()
        field = np.concatenate((field, np.array([np.finfo('f8').min])))
        for vh in mesh.vertices():
            if mesh.is_boundary(vh):
                continue
            i = vh.idx()
            m = np.max(field[vv[i]])
            if field[i] > m:
                return i
        return -1

    def find_all_local_maxima(self, mesh, field):
        vv = mesh.vv_indices()
        field = np.concatenate((field, np.array([np.finfo('f8').min])))
        idx = []
        values = []
        for vh in mesh.vertices():
            if mesh.is_boundary(vh):
                continue
            i = vh.idx()
            m = np.max(field[vv[i]])
            if field[i] > m:
                idx.append(i)
                values.append(field[i])
        return idx, values

    def cut_to_boundary(self, mesh, x, method='mmp'):
        if method == 'graph':
            return self.cut_to_boundary_graph(mesh, x)
        elif method == 'mmp':
            return self.cut_to_boundary_mmp(mesh, x)
        else:
            assert False, 'Unknown method'
            
    def cut_to_boundary_graph(self, mesh, x):
        G = nx.empty_graph(mesh.n_vertices())
        edges = mesh.ev_indices()
        pts = mesh.points()
        edge_vecs = pts[edges[:, 0]] - pts[edges[:, 1]]
        lengths = np.linalg.norm(edge_vecs, axis=1)
        G.add_weighted_edges_from([(e[0], e[1], l) for e, l in zip(edges, lengths)])

        dist = nx.shortest_path_length(G, source=x, weight='weight')
        boundary = []
        for v in mesh.vertices():
            if mesh.is_boundary(v):
                boundary.append(v.idx())
        boundary = np.array(boundary, 'i4')
        dist_to_boundary = [dist[x] for x in boundary]
        idx = np.argmin(dist_to_boundary)
        v = boundary[idx]
        p = nx.shortest_path(G, source=x, target=v, weight='weight')
        print('path to boundary', p)
        mesh, _ = mu.cut_along_curve(mesh.points(), mesh.face_vertex_indices(), p)
        return mesh

    def cut_to_boundary_mmp(self, mesh, x):
        dist = mu.pygeodesic.distance_field(mesh.points(),
                                            mesh.face_vertex_indices(),
                                            np.array([x], 'i4'), -1)
        boundary = []
        for v in mesh.vertices():
            if mesh.is_boundary(v):
                boundary.append(v.idx())
        boundary = np.array(boundary, 'i4')
        dist_to_boundary = [dist[x] for x in boundary]
        idx = np.argmin(dist_to_boundary)
        v = boundary[idx]
        
        path_edge, path_ratio = mu.pygeodesic.find_path(mesh.points(),
            mesh.face_vertex_indices(), x, v)
        mesh, p = mu.split_mesh(mesh.points(), mesh.face_vertex_indices(),
            path_edge, path_ratio)
        
        print('path to boundary', p)
        mesh, _ = mu.cut_along_curve(mesh.points(), mesh.face_vertex_indices(), p)
        return mesh


def find_source(mesh):
    print('Find source curve')
    G = nx.Graph()
    pts = mesh.points()
    mask = np.abs(pts[:, 0]) < 1.e-6
    edges = []
    for e in mesh.ev_indices():
        if mask[e[0]] and mask[e[1]]:
            edges.append(e)
    G.add_edges_from(edges)

    cnt = 0
    for c in nx.connected_components(G):
        sG = G.subgraph(c)
        p = nx.eulerian_path(sG)
        verts = [v for e in p for v in e]
        verts = verts[::2] + [verts[-1]]

        c = np.array(verts, 'i4')
        m = om.TriMesh(pts[c])
        om.write_mesh(folder+'source_%d.obj' % cnt, m)
        np.save(folder+'source_%d.npy' % cnt, c)
        cnt += 1


def disk_topology(mesh):
    print('Cut mesh to a disk')
    p = np.load(folder+'source_0.npy')
    mesh, _ = mu.cut_along_curve(mesh.points(), mesh.face_vertex_indices(), p)
    om.write_mesh(folder+'skullcap_disk.obj', mesh)


def cutting_closed_isocurves():
    mesh = om.read_trimesh(folder+'skullcap_disk.obj')
    src_pts = om.read_trimesh(folder+'source_1.obj').points()[::-1]
    vgf = ValidGeodesicField(folder)
    mesh, field = vgf.run2(mesh, src_pts)

    # reorient
    tree = cKDTree(mesh.points())
    idx = tree.query(src_pts)[1]
    field = geodesic_field_reorient(mesh, field, idx)
    np.save(folder+'field_reorient.npy', field)
    mu.colormap_vertex_color(folder+'field_reorient.off', mesh.points(),
                             mesh.face_vertex_indices(),
                             field/field.max())


if __name__ == '__main__':
    folder = '../output/skullcap/'
    os.makedirs(folder, exist_ok=True)
    mesh = om.read_trimesh('../input/skullcap.obj')
    find_source(mesh)
    disk_topology(mesh)
    cutting_closed_isocurves()

    fn_mesh = folder+'field_reorient.off'
    fn_field = folder+'field_reorient.npy'
    col_width, row_height = 0.3, 0.2
    engine = KnittingEngine(row_height, col_width)
    # NOTE: Use 100.0 (default value) or 0.0 and find out the difference.
    # engine.fps_weight = 0.0
    engine.load_mesh(fn_mesh)
    engine.load_field(fn_field)
    engine.run(folder)
