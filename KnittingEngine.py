"""
 * GPL-3.0 License
 *
 * by Zishun Liu, June 6, 2021
"""


import numpy as np
import openmesh as om
from scipy.spatial.distance import cdist
import networkx as nx
from tqdm import trange
import pickle
import time
import meshutility as mu


__all__ = ['KnittingEngine', 'KnittingMesh', 'ColumnCurve', 'geodesic_field_reorient']

class KnittingEngine:

    def __init__(self, row_height=0.2, col_width=0.3):
        # parameters
        self.row_height = row_height
        self.col_width = col_width
        self.tol = self.col_width + self.row_height*2
        self.apices = np.zeros((0, 3))
        self.fps_weight = 100.0

    def load_mesh(self, mesh_fn):
        self.mesh = om.read_trimesh(mesh_fn)

    def load_field(self, field_fn):
        self.field = np.load(field_fn).ravel()

    def run(self, folder=None, has_col=False):
        start = time.time()
        print('------------------------------------------------')
        print(time.ctime(), '- Extract column curves')
        fn = folder+'curves_col.pkl'
        if not has_col:
            self.extract_column_curves()
            with open(fn, 'wb') as f:
                pickle.dump(self.col_curves, f)
        else:
            with open(fn, 'rb') as f:
                self.col_curves = pickle.load(f)
        print(time.ctime(), '- Done')

        self.collect_pts()
        mu.write_obj_lines(folder+'/curves_col.obj', self.pts, self.col_edges)
        print('------------------------------------------------')

        # step 2
        print(time.ctime(), '- Generate row edges')
        self.generate_rows()
        print(time.ctime(), '- Done')
        mu.write_obj_lines(folder+'/curves_row.obj', self.pts, self.row_edges)
        print('------------------------------------------------')

        # step 3
        print(time.ctime(), '- Trace rows')
        self.generate_2d_knitting_map()
        print(time.ctime(), '- Done')
        print('------------------------------------------------')
        self.color_by_row(self.knitting_mesh, self.row_col_idx)
        om.write_mesh(folder+'/knitting_mesh_raw.obj',
                      self.knitting_mesh,
                      face_color=True)

        km = KnittingMesh().set(
                self.knitting_mesh.points(),
                self.knitting_mesh.face_vertex_indices(),
                self.row_col_idx,
                self.pts_col,
                self.pts_next)
        km.save(folder+'/knitting_mesh_raw.npz')

        print(time.ctime(), '- One stroke adjustment')
        km = self.map_one_stroke(km)
        print(time.ctime(), '- Done')
        print('------------------------------------------------')
        km = self.map_boundary_tri2quad(km)
        km = self.map_remove_unreferenced_vertices(km)
        km.save(folder+'/knitting_mesh_repair.npz')

        mesh = om.PolyMesh(km.v, km.f)
        self.color_by_row(mesh, km.f_ij)
        om.write_mesh(folder+'/onestroke.obj', mesh,
                      face_color=True)

        print('Total time: %.2fs' % (time.time()-start))
        print('------------------------------------------------')

    def extract_column_curves(self, values=None):
        if values is None:
            num = int(np.floor(np.max(self.field)/self.col_width))
            values = np.arange(1, num+1)*self.col_width
        else:
            num = values.shape[0]

        self.col_curves = []
        isocurves = mu.pyisocurve.isocurve(
            self.mesh.points(),
            self.mesh.face_vertex_indices(),
            self.field,
        )
        eql_tol = 1.e-5
        for i in trange(num):
            v = values[i]
            pts, _, _, isocurve_indices = isocurves.extract(v, eql_tol)
            same_value = []  # everything of the same value
            for piece in range(len(isocurve_indices)):
                col_curve = ColumnCurve(pts, isocurve_indices[piece], v, i)
                col_curve.resample(self.row_height)
                same_value.append(col_curve)

            self.col_curves.append(same_value)

        return

    def collect_pts(self,):
        num_cols = len(self.col_curves)
        all_pts = []
        for i in range(num_cols):
            for x in self.col_curves[i]:
                all_pts.append(x.resampled)
        self.pts = np.vstack(tuple(all_pts))
        num_pts = self.pts.shape[0]
        self.pts_col = np.empty((num_pts,), dtype=np.int32)
        self.pts_next = np.arange(1, num_pts+1, dtype=np.int32)

        cnt = 0
        self.col_start = []
        self.col_edges = []
        for i in range(num_cols):
            cnt_on_piece = 0
            piece_start = []
            for x in self.col_curves[i]:
                n = x.resampled.shape[0]
                self.col_edges.append(np.arange(cnt+cnt_on_piece,
                                                cnt+cnt_on_piece+n))
                self.pts_next[cnt+cnt_on_piece+n-1] = -1
                piece_start.append(cnt+cnt_on_piece)
                cnt_on_piece += n
            self.col_start.append(piece_start)
            self.pts_col[cnt:cnt+cnt_on_piece] = i
            cnt += cnt_on_piece
        self.col_start.append([self.pts.shape[0]])
        return

    def generate_rows(self):
        self.knitting_mesh = om.PolyMesh(self.pts, np.array([]))
        self.row_edges = []  # only used for write_obj_lines
        np.random.seed(5)

        num_cols = len(self.col_curves)
        for i in trange(num_cols-1):
            edges = self.connect_col_curves_graph(i)
            self.row_edge_to_column_mesh(i, edges)
            self.row_edges.extend(edges)

        return

    def connect_col_curves_graph(self, col_id):
        offset1 = self.col_start[col_id][0]
        offset2 = self.col_start[col_id+1][0]
        piece_len_1 = [x.resampled.shape[0] for x in self.col_curves[col_id]]
        piece_len_2 = [x.resampled.shape[0] for x in self.col_curves[col_id+1]]
        piece_len_1.insert(0, 0)
        piece_len_2.insert(0, 0)
        pieces1 = np.array(piece_len_1).cumsum() + offset1
        pieces2 = np.array(piece_len_2).cumsum() + offset2

        edges = []

        for i in range(pieces1.size-1):
            for j in range(pieces2.size-1):
                pts1 = self.pts[pieces1[i]:pieces1[i+1]:2]
                pts2 = self.pts[pieces2[j]:pieces2[j+1]:2]
                pts11 = self.pts[pieces1[i]+1:pieces1[i+1]:2]
                pts21 = self.pts[pieces2[j]+1:pieces2[j+1]:2]
                e = self.connect_col_curves_graph_piece(pts1, pts2, pts11,
                                                         pts21, pieces1[i],
                                                         pieces2[j])
                edges.extend(e)
        return edges

    def connect_col_curves_graph_piece(self, pts1, pts2, pts11, pts21,
                                        offset1, offset2):
        if pts1.shape[0] < 2 or pts2.shape[0] < 2:
            return []

        def weight(w, d2apices):
            return w+self.fps_weight*self.col_width*np.exp(-d2apices*d2apices)

        dist = cdist(pts1, pts2)
        mask = (dist < self.tol)

        r, c = np.where(mask)
        if r.size == 0:
            return []

        dirc1 = np.empty_like(pts1)
        dirc1[:-1] = pts1[1:]-pts1[:-1]
        dirc1[-1] = pts1[-1]-pts1[-2]
        dirc2 = np.empty_like(pts2)
        dirc2[:-1] = pts2[1:]-pts2[:-1]
        dirc2[-1] = pts2[-1]-pts2[-2]
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not mask[i, j]:
                    continue
                if dirc1[i].dot(dirc2[j]) < 0:
                    mask[i, j] = False
        r, c = np.where(mask)
        if r.size == 0:
            return []

        bridges_as_nodes = mask.astype(np.int32)
        size = bridges_as_nodes.shape
        bridges_as_nodes = (bridges_as_nodes.ravel().cumsum()-1).reshape(size)
        bridges_as_nodes[~mask] = -1

        if np.where(mask)[0].size == 0:
            return []

        r, c = np.where(mask)
        vec = pts2[c[0]] - pts1[r[0]]
        if dirc1[r[0]].dot(vec) < 0.:
            c0 = np.argmin(dist[r[0], :])
            source = bridges_as_nodes[r[0], c0]
            if source == -1:
                idx = np.argsort(dist[r[0], :])
                for c0 in idx[1:]:
                    source = bridges_as_nodes[r[0], c0]
                    if source >= 0:
                        break
        else:
            r0 = np.argmin(dist[:, c[0]])
            source = bridges_as_nodes[r0, c[0]]
            if source == -1:
                idx = np.argsort(dist[:, c[0]])
                for r0 in idx[1:]:
                    source = bridges_as_nodes[r0, c[0]]
                    if source >= 0:
                        break
        vec = pts2[c[-1]] - pts1[r[-1]]
        if dirc1[r[-1]].dot(vec) > 0.:
            c0 = np.argmin(dist[r[-1], :])
            target = bridges_as_nodes[r[-1], c0]
            if target == -1:  # found an invalid bridge
                idx = np.argsort(dist[r[-1], :])
                for c0 in idx[1:]:
                    target = bridges_as_nodes[r[-1], c0]
                    if target >= 0:
                        break
        else:
            r0 = np.argmin(dist[:, c[-1]])
            target = bridges_as_nodes[r0, c[-1]]
            if target == -1:
                idx = np.argsort(dist[:, c[-1]])
                for r0 in idx[1:]:
                    target = bridges_as_nodes[r0, c[-1]]
                    if target >= 0:
                        break

        nodes = np.empty((bridges_as_nodes.max()+1, 2), dtype=np.int32)
        for i, j in zip(*np.where(bridges_as_nodes >= 0)):
            node = bridges_as_nodes[i, j]
            nodes[node, :] = [i, j]

        if 0 in nodes[source]-nodes[target]:
            return []

        G = nx.DiGraph()
        
        for i, j in zip(*np.where(bridges_as_nodes >= 0)):
            node = bridges_as_nodes[i, j]

            # type A
            if (i+2 < bridges_as_nodes.shape[0] and j+1 < bridges_as_nodes.shape[1]
                    and bridges_as_nodes[i+2, j+1] >= 0):
                w = dist[i+1, j+1] + dist[i+2, j+1] + \
                    np.linalg.norm(pts11[i, :]-pts21[j, :]) + \
                    np.linalg.norm(pts11[i+1, :]-pts2[j+1, :])
                if self.apices.shape[0] > 0:
                    diff = self.apices-pts2[j+1, :][np.newaxis, :]
                    d2apices = np.linalg.norm(diff, axis=1).min()
                    w = weight(w, d2apices)
                if bridges_as_nodes[i+2, j+1] == target:
                    w *= 10  # penalize this triangle
                G.add_edge(node, bridges_as_nodes[i+2, j+1], weight=w)

            # type B
            if (i+3 < bridges_as_nodes.shape[0] and j+1 < bridges_as_nodes.shape[1]
                    and bridges_as_nodes[i+3, j+1] >= 0):
                w = dist[i+1, j+1] + dist[i+2, j+1] + dist[i+3, j+1] + \
                    np.linalg.norm(pts11[i, :]-pts21[j, :]) + \
                    np.linalg.norm(pts11[i+1, :]-pts2[j+1, :]) + \
                    np.linalg.norm(pts11[i+2, :]-pts2[j+1, :])
                if self.apices.shape[0] > 0:
                    diff = self.apices-pts2[j+1, :][np.newaxis, :]
                    d2apices = np.linalg.norm(diff, axis=1).min()
                    w = weight(w, d2apices/2)
                if bridges_as_nodes[i+3, j+1] == target:
                    w *= 10  # penalize triangle on the boundary
                w *= 100  # increase penalty
                G.add_edge(node, bridges_as_nodes[i+3, j+1], weight=w)

            # type A2
            if (i+1 < bridges_as_nodes.shape[0] and j+2 < bridges_as_nodes.shape[1]
                    and bridges_as_nodes[i+1, j+2] >= 0):
                w = dist[i+1, j+2] + np.linalg.norm(pts11[i, :]-pts2[j+1, :])\
                 + np.linalg.norm(pts11[i, :]-pts21[j, :]) + \
                 np.linalg.norm(pts11[i, :]-pts21[j+1, :])
                if self.apices.shape[0] > 0:
                    diff = self.apices-pts11[i, :][np.newaxis, :]
                    d2apices = np.linalg.norm(diff, axis=1).min()
                    w = weight(w, d2apices)
                G.add_edge(node, bridges_as_nodes[i+1, j+2], weight=w)

            # type B2
            if (i+1 < bridges_as_nodes.shape[0] and j+3 < bridges_as_nodes.shape[1]
                    and bridges_as_nodes[i+1, j+3] >= 0):
                w = dist[i+1, j+3] + np.linalg.norm(pts11[i, :]-pts2[j+1, :]) \
                    + np.linalg.norm(pts11[i, :]-pts2[j+2, :]) \
                    + np.linalg.norm(pts11[i, :]-pts21[j, :]) + \
                    np.linalg.norm(pts11[i, :]-pts21[j+1, :]) + \
                    np.linalg.norm(pts11[i, :]-pts21[j+2, :])
                if self.apices.shape[0] > 0:
                    diff = self.apices-pts11[i, :][np.newaxis, :]
                    d2apices = np.linalg.norm(diff, axis=1).min()
                    w = weight(w, d2apices/2)
                w *= 100  # increase penalty
                G.add_edge(node, bridges_as_nodes[i+1, j+3], weight=w)

            # type C
            if (i+1 < bridges_as_nodes.shape[0] and j+1 < bridges_as_nodes.shape[1]
                    and bridges_as_nodes[i+1, j+1] >= 0):
                w = dist[i+1, j+1] + np.linalg.norm(pts11[i, :]-pts21[j, :])
                G.add_edge(node, bridges_as_nodes[i+1, j+1], weight=w)

        try:
            p = nx.shortest_path(G, source=source, target=target,
                                 weight='weight')
        except Exception as e:
            print('[Networkx Exception]', e)
            return []
        
        x = p[0]
        bridges = [[nodes[x, 0]*2+offset1, nodes[x, 1]*2+offset2]]
        pi, pj = nodes[x, :]
        apices = []
        for x in p[1:]:
            i, j = nodes[x, :]
            # type A
            if i-pi == 2 and j-pj == 1:
                bridges.append([i*2-3+offset1, j*2-1+offset2])
                bridges.append([i*2-2+offset1, j*2+offset2])
                bridges.append([i*2-1+offset1, j*2+offset2])
                bridges.append([i*2+offset1, j*2+offset2])
                pi, pj = i, j
                apices.append(pts2[j, :])
                continue
            # type B
            if i-pi == 3 and j-pj == 1:
                bridges.append([i*2-5+offset1, j*2-1+offset2])
                bridges.append([i*2-4+offset1, j*2+offset2])
                bridges.append([i*2-3+offset1, j*2+offset2])
                bridges.append([i*2-2+offset1, j*2+offset2])
                bridges.append([i*2-1+offset1, j*2+offset2])
                bridges.append([i*2+offset1, j*2+offset2])
                pi, pj = i, j
                apices.append(pts2[j, :])
                continue
            # type A2
            if i-pi == 1 and j-pj == 2:
                bridges.append([i*2-1+offset1, j*2-3+offset2])
                bridges.append([i*2-1+offset1, j*2-2+offset2])
                bridges.append([i*2-1+offset1, j*2-1+offset2])
                bridges.append([i*2+offset1, j*2+offset2])
                pi, pj = i, j
                apices.append(pts11[i-1, :])
                continue
            # type B2
            if i-pi == 1 and j-pj == 3:
                bridges.append([i*2-1+offset1, j*2-5+offset2])
                bridges.append([i*2-1+offset1, j*2-4+offset2])
                bridges.append([i*2-1+offset1, j*2-3+offset2])
                bridges.append([i*2-1+offset1, j*2-2+offset2])
                bridges.append([i*2-1+offset1, j*2-1+offset2])
                bridges.append([i*2+offset1, j*2+offset2])
                pi, pj = i, j
                apices.append(pts11[i-1, :])
                continue
            # type C
            if i-pi == 1 and j-pj == 1:
                bridges.append([i*2-1+offset1, j*2-1+offset2])
                bridges.append([i*2+offset1, j*2+offset2])
                pi, pj = i, j
                continue
            print('Unknown case in connect_col_curves_graph_piece!')

        if len(apices) > 0:
            self.apices = np.vstack((self.apices, np.array(apices)))

        return bridges

    def row_edge_to_column_mesh(self, col_id, edges):
        G = nx.Graph()  # graph of edges
        G.add_edges_from(edges)

        start = self.col_start[col_id][0]
        for piece in self.col_curves[col_id]:
            for i in range(piece.resampled.shape[0]):
                p0 = start + i
                p3 = self.pts_next[p0]
                if G.has_node(p0):
                    neighbors0 = [n for n in G[p0]]
                    neighbors0.sort()
                    for p1 in neighbors0:
                        p2 = self.pts_next[p1]
                        if p2 >= 0 and p3 >= 0 and G.has_edge(p2, p3):
                            self.add_face(self.knitting_mesh,
                                          [p0, p1, p2, p3])
                            # remove the diagonal edge if two triangles are
                            # merged as a quad
                            if G.has_edge(p0, p2):
                                G.remove_edge(p0, p2)
                            if G.has_edge(p1, p3):
                                G.remove_edge(p1, p3)
                            break
                        if p3 >= 0 and G.has_edge(p1, p3):
                            self.add_face(self.knitting_mesh, [p0, p1, p3])
                            break
                        if p2 >= 0 and G.has_edge(p0, p2):
                            self.add_face(self.knitting_mesh, [p0, p1, p2])
                            continue

            start += piece.resampled.shape[0]

        return

    # Mesh utilities

    def add_face(self, mesh, idx):
        verts = [mesh.vertex_handle(x) for x in idx]
        mesh.add_face(verts)

    def num_fv(self, mesh, fh):
        return sum(1 for _ in mesh.fv(fh))

    def find_halfedge(self, mesh, het):
        vh0 = mesh.vertex_handle(het[0])
        vh1 = mesh.vertex_handle(het[1])
        heh = mesh.find_halfedge(vh0, vh1)
        return heh

    def generate_2d_knitting_map(self):
        face_row_idx = self.trace_rows()
        self.sort_rows(face_row_idx)
        km = KnittingMesh().set(
            self.knitting_mesh.points(),
            self.knitting_mesh.face_vertex_indices(),
            self.row_col_idx,
            self.pts_col,
            self.pts_next)
        return km

    def trace_rows(self,):
        n_face = self.knitting_mesh.n_faces()
        face_visited = np.zeros((n_face,), '?')
        face_row_idx = -np.ones((n_face,), 'i')

        cnt_rows = 0
        unvisited = np.where(~face_visited)[0]
        while unvisited.shape[0] > 0:
            fh = self.knitting_mesh.face_handle(unvisited[0])
            face_visited[fh.idx()] = True
            face_row_idx[fh.idx()] = cnt_rows
            for heh in self.knitting_mesh.fh(fh):
                vh0 = self.knitting_mesh.from_vertex_handle(heh)
                vh1 = self.knitting_mesh.to_vertex_handle(heh)
                if self.pts_col[vh0.idx()] == self.pts_col[vh1.idx()]:
                    face_visited, face_row_idx = self.extend_row(
                        face_visited, heh, cnt_rows, face_row_idx)

            cnt_rows += 1
            unvisited = np.where(~face_visited)[0]
        return face_row_idx

    def sort_rows(self, face_row_idx):
        DG = nx.DiGraph()
        num_pts = self.pts_next.shape[0]
        for p0 in range(num_pts):
            p1 = self.pts_next[p0]
            if p1 < 0:
                continue
            p2 = self.pts_next[p1]
            if p2 < 0:
                continue
            e01 = self.find_halfedge(self.knitting_mesh, (p0, p1))
            if not e01.is_valid():
                continue
            e12 = self.find_halfedge(self.knitting_mesh, (p1, p2))
            if not e12.is_valid():
                continue
            e10 = self.knitting_mesh.opposite_halfedge_handle(e01)
            e21 = self.knitting_mesh.opposite_halfedge_handle(e12)
            faces01 = []
            if not self.knitting_mesh.is_boundary(e01):
                f = self.knitting_mesh.face_handle(e01)
                if self.num_fv(self.knitting_mesh, f) <= 4:
                    faces01.append(f)
            if not self.knitting_mesh.is_boundary(e10):
                f = self.knitting_mesh.face_handle(e10)
                if self.num_fv(self.knitting_mesh, f) <= 4:
                    faces01.append(f)
            faces12 = []
            if not self.knitting_mesh.is_boundary(e12):
                f = self.knitting_mesh.face_handle(e12)
                if self.num_fv(self.knitting_mesh, f) <= 4:
                    faces12.append(f)
            if not self.knitting_mesh.is_boundary(e21):
                f = self.knitting_mesh.face_handle(e21)
                if self.num_fv(self.knitting_mesh, f) <= 4:
                    faces12.append(f)

            for f01 in faces01:
                for f12 in faces12:
                    r0 = face_row_idx[f01.idx()]
                    r1 = face_row_idx[f12.idx()]
                    DG.add_edge(r0, r1)

        for fh in self.knitting_mesh.faces():
            row_idx = face_row_idx[fh.idx()]
            col_idx = [self.pts_col[vh.idx()] for vh in
                       self.knitting_mesh.fv(fh)]
            col_idx = min(col_idx)

        order = np.array(list(nx.topological_sort(DG)), 'i')
        self.collect_row_col_index(order, face_row_idx)

    def extend_row(self, face_visited, heh, cnt_rows, face_row_idx):
        oheh = self.knitting_mesh.opposite_halfedge_handle(heh)
        while (not self.knitting_mesh.is_boundary(oheh)):
            fh = self.knitting_mesh.face_handle(oheh)
            if self.num_fv(self.knitting_mesh, fh) != 4:
                face_visited[fh.idx()] = True
                face_row_idx[fh.idx()] = cnt_rows
                break
            face_visited[fh.idx()] = True
            face_row_idx[fh.idx()] = cnt_rows
            heh = self.knitting_mesh.next_halfedge_handle(oheh)
            heh = self.knitting_mesh.next_halfedge_handle(heh)
            oheh = self.knitting_mesh.opposite_halfedge_handle(heh)

        return face_visited, face_row_idx

    def collect_row_col_index(self, row_sort, face_row_idx):
        num_rows = row_sort.shape[0]
        row_sort_inv = np.empty_like(row_sort, dtype=np.int32)
        row_sort_inv[row_sort[:]] = np.arange(num_rows)

        self.row_col_idx = -np.ones((self.knitting_mesh.n_faces(), 2),
                                    dtype=np.int32)
        for fh in self.knitting_mesh.faces():
            row_id_unsort = face_row_idx[fh.idx()]
            row_idx = row_sort_inv[row_id_unsort]
            col_idx = [self.pts_col[vh.idx()] for vh in
                       self.knitting_mesh.fv(fh)]
            col_idx = min(col_idx)
            self.row_col_idx[fh.idx(), 0] = row_idx
            self.row_col_idx[fh.idx(), 1] = col_idx

    def map_boundary_tri2quad(self, km_in):
        map_f = self.generate_map_face(km_in.f_ij)
        num_r, _ = map_f.shape

        verts = km_in.v.copy()
        faces = km_in.f.copy()
        face_ij = km_in.f_ij.copy()
        vert_col = km_in.v_col.copy()
        vert_nxt = km_in.v_nxt.copy()
        mesh = om.PolyMesh(verts, faces)

        prev = self.next2prev(vert_nxt).tolist()

        verts_add = []
        vert_col_add = []
        vert_nxt_add = []  # also fix that of the exitings
        cnt = verts.shape[0]
        w = 0.6  # weight for inserted vertex, does not matter
        for r in range(num_r):
            found = np.where(map_f[r, :] >= 0)[0]
            left = found[0]
            right = found[-1]

            # part I
            f = faces[map_f[r, left], :].copy()
            if (f[3] < 0 and vert_col[f[1]] == vert_col[f[2]] and
                    mesh.is_boundary(mesh.vertex_handle(f[0]))):
                p = w*verts[f[0], :]+(1-w)*verts[f[1], :]
                p = w*(verts[f[0], :]-verts[f[2], :]) + \
                    w*(verts[f[1], :]-verts[f[2], :])+verts[f[2], :]
                verts_add.append(p)
                vert_col_add.append(vert_col[f[0]])
                if prev[f[0]] >= 0:
                    vert_nxt[prev[f[0]]] = cnt
                vert_nxt_add.append(f[0])
                faces[map_f[r, left], :] = [cnt, f[1], f[2], f[0]]
                for i in range(r):
                    for j in range(2):
                        fid = map_f[i, left-j]
                        if left-j < 0:
                            continue
                        if fid < 0:
                            continue
                        for k in range(4):
                            if faces[fid, k] == f[0]:
                                faces[fid, k] = cnt

                cnt += 1

            # part II
            f = faces[map_f[r, right], :].copy()
            if (f[3] < 0 and vert_col[f[0]] == vert_col[f[2]] and
                    mesh.is_boundary(mesh.vertex_handle(f[1]))):
                p = w*verts[f[1], :]+(1-w)*verts[f[0], :]
                p = w*(verts[f[1], :]-verts[f[2], :]) + \
                    w*(verts[f[0], :]-verts[f[2], :])+verts[f[2], :]
                verts_add.append(p)
                vert_col_add.append(vert_col[f[1]])
                if prev[f[1]] >= 0:
                    vert_nxt[prev[f[1]]] = cnt
                vert_nxt_add.append(f[1])
                faces[map_f[r, right], :] = [f[0], cnt, f[1], f[2]]
                for i in range(r):
                    for j in range(2):
                        if right+j >= map_f.shape[1]:
                            continue
                        fid = map_f[i, right+j]
                        if fid < 0:
                            continue
                        for k in range(4):
                            if faces[fid, k] == f[1]:
                                faces[fid, k] = cnt

                cnt += 1

        if len(verts_add) == 0:
            return km_in
        verts_a = np.vstack((verts, np.array(verts_add)))
        vert_col = np.concatenate((vert_col, np.array(vert_col_add,
                                                      dtype=np.int32)))
        vert_nxt = np.concatenate((vert_nxt, np.array(vert_nxt_add,
                                                      dtype=np.int32)))

        km = KnittingMesh().set(verts_a, faces, face_ij, vert_col, vert_nxt)
        return km

    def next2prev(self, nxt):
        prev = -1*np.ones_like(nxt)
        for i in range(nxt.shape[0]):
            if nxt[i] >= 0:
                prev[nxt[i]] = i
        return prev

    def map_one_stroke(self, km_in):
        map2d = self.generate_map_face(km_in.f_ij)
        map2d_v = self.generate_map_vertex(km_in.f, km_in.f_ij, km_in.v_col)
        num_r = map2d.shape[0]//2
        v_a = km_in.v.copy()
        faces_a = km_in.f.copy()
        face_ij_a = km_in.f_ij.copy()
        v_col_a = km_in.v_col.copy()
        v_nxt_a = km_in.v_nxt.copy()
        for i in trange(num_r-1):
            r = 2*i+1

            found = np.where(map2d[r, :] >= 0)[0]
            head0 = found[0]
            tail0 = found[-1]
            found = np.where(map2d[r+1, :] >= 0)[0]
            head1 = found[0]
            tail1 = found[-1]
            if head0 > tail1+1 or head1 > tail0+1:
                print('jumping %d' % (r))
                continue

            if head0 < head1:
                v_a, faces_a, face_ij_a, v_col_a, v_nxt_a = \
                self.add_halfrow(r+1, head0, head1,
                    map2d, map2d_v, v_a, faces_a, face_ij_a, v_col_a, v_nxt_a)
            elif head0 > head1:
                v_a, faces_a, face_ij_a, v_col_a, v_nxt_a = \
                self.add_halfrow(r, head1, head0,
                    map2d, map2d_v, v_a, faces_a, face_ij_a, v_col_a, v_nxt_a)
        km = KnittingMesh().set(v_a, faces_a, face_ij_a,
                                v_col_a, v_nxt_a)
        return km

    def add_halfrow(self, r, head0, head1,
                    map2d, map2d_v, v_a, faces_a, face_ij_a,
                    v_col_a, v_nxt_a):
        num_f = faces_a.shape[0]
        num_v = v_a.shape[0]
        num_add = head1-head0
        map2d[r, head0:head1] = np.arange(num_f, num_f+num_add)

        if r % 2 == 0:  # even
            map2d_v[r+1, head0:head1] = np.arange(num_v, num_v+num_add)
            ij_add = np.empty((num_add, 2), dtype=np.int32)
            ij_add[:, 0] = r
            ij_add[:, 1] = np.arange(head0, head1)
            v_col_add = np.arange(head0, head1)
            v_nxt_a[map2d_v[r, head0:head1]] = map2d_v[r+1, head0:head1]
            v_nxt_add = np.full((num_add,), -1, dtype=np.int32)
            idx0 = map2d_v[r-1, head0:head1]
            idx1 = v_nxt_a[idx0]
            v_add = 2*v_a[idx1] - v_a[idx0]
            faces_add = np.empty((num_add, 4), dtype=np.int32)
            for i in range(num_add):
                 faces_add[i] = [map2d_v[r, head0+i], map2d_v[r, head0+i+1],
                 map2d_v[r+1, head0+i+1], map2d_v[r+1, head0+i]]
        else:  # odd
            map2d_v[r, head0:head1] = np.arange(num_v, num_v+num_add)
            ij_add = np.empty((num_add, 2), dtype=np.int32)
            ij_add[:, 0] = r
            ij_add[:, 1] = np.arange(head0, head1)
            v_col_add = np.arange(head0, head1)
            v_nxt_add = map2d_v[r+1, head0:head1]
            idx0 = map2d_v[r+1, head0:head1]
            idx1 = v_nxt_a[idx0]
            v_add = 2*v_a[idx0] - v_a[idx1]
            faces_add = np.empty((num_add, 4), dtype=np.int32)
            for i in range(num_add):
                 faces_add[i] = [map2d_v[r, head0+i], map2d_v[r, head0+i+1],
                 map2d_v[r+1, head0+i+1], map2d_v[r+1, head0+i]]

        v_a = np.vstack((v_a, v_add))
        faces_a = np.vstack((faces_a, faces_add))
        face_ij_a = np.vstack((face_ij_a, ij_add))
        v_col_a = np.concatenate((v_col_a, v_col_add))
        v_nxt_a = np.concatenate((v_nxt_a, v_nxt_add))
        return v_a, faces_a, face_ij_a, v_col_a, v_nxt_a
        
    def map_remove_unreferenced_vertices(self, km_in):
        num_v = km_in.v.shape[0]
        used_idx = np.unique(km_in.f.ravel())
        used_idx = used_idx[used_idx >= 0]
        used = np.zeros((num_v,), dtype=np.int32)
        used[used_idx] = 1
        compress_cum = np.cumsum(used)
        compress_cum[used == 0] = 0
        compress = compress_cum-1

        verts = km_in.v[used.astype('?'), :]
        faces = km_in.f
        for i in range(faces.shape[0]):
            for j in range(faces.shape[1]):
                if faces[i, j] >= 0:
                    faces[i, j] = compress[faces[i, j]]

        face_ij = km_in.f_ij
        face_ij[:, 1] -= face_ij[:, 1].min()

        vert_wale = km_in.v_col[used.astype('?')]
        vert_wale -= vert_wale.min()

        vert_next = km_in.v_nxt[used.astype('?')]
        for i in range(vert_next.shape[0]):
            if vert_next[i] >= 0:
                vert_next[i] = compress[vert_next[i]]

        km = KnittingMesh().set(verts, faces, face_ij, vert_wale, vert_next)

        return km

    def color_by_row(self, mesh, face_ij, end=0):
        num_rows = np.max(face_ij[:, 0])+1
        np.random.seed(5)
        colors = np.random.random((num_rows, 4)) * 0.7
        colors[1::2, :3] = colors[::2, :3] + 0.3
        colors[:, 3] = 1

        mesh.request_face_colors()
        for fh in mesh.faces():
            row_ind = face_ij[fh.idx(), 0]
            if end == -1:
                row_ind = (row_ind+1)//2
            elif end == 1:
                row_ind = (row_ind)//2
            mesh.set_color(fh, colors[row_ind, :])

    def generate_map_face(self, ij):
        num_r, num_c = np.max(ij, axis=0)
        num_r += 1
        num_c += 1
        map2d = -np.ones((num_r, num_c), dtype=np.int32)
        for f in range(ij.shape[0]):
            i, j = ij[f, :]
            if i >= 0 and j >= 0:
                map2d[i, j] = f
            else:
                print('Warning! found unknown face in generate_map_face', f)

        return map2d

    def generate_map_vertex(self, faces, ij, v_col):
        num_r, num_c = np.max(ij, axis=0)
        num_r += 2
        num_c += 2
        map2d = -np.ones((num_r, num_c), dtype=np.int32)
        for fid in range(ij.shape[0]):
            i, j = ij[fid, :]
            if i < 0 or j < 0:
                print('Warning! found unknown face in generate_map_vertex',
                      fid)
                continue
            f = faces[fid, :]
            if f[-1] >= 0:  # quad
                map2d[i, j] = f[0]
                map2d[i, j+1] = f[1]
                map2d[i+1, j+1] = f[2]
                map2d[i+1, j] = f[3]
            else:  # tri
                map2d[i, j] = f[0]
                map2d[i, j+1] = f[1]
                if v_col[f[2]] == v_col[f[0]]:
                    map2d[i+1, j+1] = f[1]
                    map2d[i+1, j] = f[2]
                else:
                    map2d[i+1, j+1] = f[2]
                    map2d[i+1, j] = f[0]

        return map2d


class KnittingMesh:
    def __init__(self,):
        pass

    def set(self, verts, faces, face_ij, vert_col, vert_nxt):
        self.v = verts
        self.f = faces
        self.f_ij = face_ij
        self.v_col = vert_col
        self.v_nxt = vert_nxt
        return self

    def load(self, fn):
        data = np.load(fn)
        self.set(data['verts'], data['faces'], data['face_ij'],
                 data['vert_col'], data['vert_nxt'])
        return self

    def save(self, fn):
        np.savez(fn,
                 verts=self.v,
                 faces=self.f,
                 face_ij=self.f_ij,
                 vert_col=self.v_col,
                 vert_nxt=self.v_nxt)


class ColumnCurve:
    def __init__(self, pts, isocurve_indices, value, ind):
        self.value = value
        self.ind = ind
        self.pts = pts
        self.indices = isocurve_indices

    def resample(self, step):
        pl = PolyLine(self.pts, self.indices)
        # sample on curve
        length = pl.length()
        num = int(np.round(length/(step*2)))
        self.resampled = pl.reparametrization_num(num*2)


class PolyLine:
    def __init__(self, pts, indices):
        self.pts = pts[indices,:]
        self.num = self.pts.shape[0]-1
        self.segment_lengths = None

    def length(self,):
        if self.segment_lengths is None:
            self.segment_lengths = np.empty((self.num,))
            for i in range(self.num):
                self.segment_lengths[i] = np.linalg.norm(self.pts[i,:]-self.pts[i+1,:])
        return self.segment_lengths.sum()

    def march_at_most_one_segment(self, idx, ratio, step):
        current_segment_left = self.segment_lengths[idx]*(1-ratio)
        if current_segment_left >= step:
            ratio_march = ratio + step/self.segment_lengths[idx]
            p_march = self.pts[idx] + ratio_march*(self.pts[idx+1]-self.pts[idx])
            return True, p_march, idx, ratio_march, 0.0
        return False, self.pts[idx+1,:], idx+1, 0.0, step-current_segment_left

    def march_step(self, idx, ratio, step):
        reached = False
        step_left = step
        idx_march = idx
        ratio_march = ratio
        while not reached:
            reached, p_march, idx_march, ratio_march, step_left = self.march_at_most_one_segment(idx_march, ratio_march, step_left)
        return p_march, idx_march, ratio_march
        
    def reparametrization(self, step):
        length = self.length()
        num = int(np.round(length/step))
        return self.reparametrization_num(num)
        
    def reparametrization_num(self, num):
        length = self.length()
        if num <= 0:
            num = 1
        step = length/num

        pts_new = [self.pts[0,:]]
        idx = 0
        ratio = 0.0
        for _ in range(num-1):
            p, idx, ratio = self.march_step(idx, ratio, step)
            pts_new.append(p)
        pts_new.append(self.pts[-1,:])

        return np.array(pts_new)


def geodesic_field_reorient(mesh, field, split_index):
    num_v = mesh.n_vertices()
    v_on_split = np.zeros((num_v,), '?')
    v_on_split[split_index] = True

    G = nx.Graph()
    for eh in mesh.edges():
        heh0 = mesh.halfedge_handle(eh, 0)
        heh1 = mesh.halfedge_handle(eh, 1)
        vh0 = mesh.from_vertex_handle(heh0)
        vh1 = mesh.to_vertex_handle(heh0)
        if v_on_split[vh0.idx()] and v_on_split[vh1.idx()]:
            continue
        if mesh.is_boundary(heh0) or mesh.is_boundary(heh1):
            continue
        else:
            fh0 = mesh.face_handle(heh0)
            fh1 = mesh.face_handle(heh1)
            G.add_edge(fh0.idx(), fh1.idx())

    num_comp = nx.algorithms.components.number_connected_components(G)
    if num_comp != 2:
        print('Caution! Found #%d components' % (num_comp))
    comp0 = sorted(nx.connected_components(G), key=len)[0]
    faces = mesh.face_vertex_indices()
    assert(faces.shape[1] == 3)
    flip_f = faces[np.array(list(comp0), dtype=np.int32)]
    flip = flip_f.ravel()
    field[flip] *= -1

    # check again
    for i in range(len(split_index)-1):
        v0 = split_index[i]
        v1 = split_index[i+1]
        vh0 = mesh.vertex_handle(v0)
        vh1 = mesh.vertex_handle(v1)
        heh = mesh.find_halfedge(vh0, vh1)
        if not mesh.is_boundary(heh):
            heh1 = mesh.next_halfedge_handle(heh)
            v2 = mesh.to_vertex_handle(heh1).idx()
            if field[v2] < 0.:
                field -= np.min(field)
            else:
                field = np.max(field) - field
            break

    return field


if __name__ == '__main__':
    pass
