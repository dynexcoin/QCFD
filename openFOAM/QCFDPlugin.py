from __future__ import print_function
from scipy.sparse import lil_matrix, csr_matrix, save_npz
import os
import re
import numpy as np
import gzip
import shutil
import tarfile
import zipfile
import struct
from collections import namedtuple

# some methods are forked from OpenFoamParser library

def calc_phase_surface_area(mesh, phi, face_area=None, omg=1.5):
    """
    calculate phase surface area for VOF
    :param mesh: FoamMesh object
    :param phi: vof data, numpy array
    :param face_area: face area, scalar or list or numpy array
    :param omg: power index
    :return: phase surface area
    """
    if face_area is not None:
        try:
            if len(face_area) == 1:
                face_area = [face_area[0]] * mesh.num_face
        except TypeError:
            face_area = [face_area] * mesh.num_face
    else:
        if mesh.face_areas is None:
            face_area = [1.] * mesh.num_face
        else:
            face_area = mesh.face_areas
    return sum([face_area[n]*abs(phi[mesh.owner[n]] - phi[mesh.neighbour[n]])**omg for n in range(mesh.num_inner_face)])


def parse_field_all(fn):
    """
    parse internal field, extract data to numpy.array
    :param fn: file name
    :return: numpy array of internal field and boundary
    """
    if not os.path.exists(fn):
        print("Can not open file " + fn)
        return None
    with open(fn, "rb") as f:
        content = f.readlines()
        return parse_internal_field_content(content), parse_boundary_content(content)


def parse_internal_field(fn):
    """
    parse internal field, extract data to numpy.array
    :param fn: file name
    :return: numpy array of internal field
    """
    if not os.path.exists(fn):
        print("Can not open file " + fn)
        return None
    with open(fn, "rb") as f:
        content = f.readlines()
        return parse_internal_field_content(content)


def parse_internal_field_content(content):
    """
    parse internal field from content
    :param content: contents of lines
    :return: numpy array of internal field
    """
    is_binary = is_binary_format(content)
    for ln, lc in enumerate(content):
        if lc.startswith(b'internalField'):
            if b'nonuniform' in lc:
                return parse_data_nonuniform(content, ln, len(content), is_binary)
            elif b'uniform' in lc:
                return parse_data_uniform(content[ln])
            break
    return None


def parse_boundary_field(fn):
    """
    parse internal field, extract data to numpy.array
    :param fn: file name
    :return: numpy array of boundary field
    """
    if not os.path.exists(fn):
        print("Can not open file " + fn)
        return None
    with open(fn, "rb") as f:
        content = f.readlines()
        return parse_boundary_content(content)


def parse_boundary_content(content):
    """
    parse each boundary from boundaryField
    :param content:
    :return:
    """
    data = {}
    is_binary = is_binary_format(content)
    bd = split_boundary_content(content)
    for boundary, (n1, n2) in bd.items():
        pd = {}
        n = n1
        while True:
            lc = content[n]
            if b'nonuniform' in lc:
                v = parse_data_nonuniform(content, n, n2, is_binary)
                pd[lc.split()[0]] = v
                if not is_binary:
                    n += len(v) + 4
                else:
                    n += 3
                continue
            elif b'uniform' in lc:
                pd[lc.split()[0]] = parse_data_uniform(content[n])
            n += 1
            if n > n2:
                break
        data[boundary] = pd
    return data


def parse_data_uniform(line):
    """
    parse uniform data from a line
    :param line: a line include uniform data, eg. "value           uniform (0 0 0);"
    :return: data
    """
    if b'(' in line:
        return np.array([float(x) for x in line.split(b'(')[1].split(b')')[0].split()])
    return float(line.split(b'uniform')[1].split(b';')[0])


def parse_data_nonuniform(content, n, n2, is_binary):
    """
    parse nonuniform data from lines
    :param content: data content
    :param n: line number
    :param n2: last line number
    :param is_binary: binary format or not
    :return: data
    """
    num = int(content[n + 1])
    if not is_binary:
        if b'scalar' in content[n]:
            data = np.array([float(x) for x in content[n + 3:n + 3 + num]])
        else:
            data = np.array([ln[1:-2].split() for ln in content[n + 3:n + 3 + num]], dtype=float)
    else:
        nn = 1
        if b'vector' in content[n]:
            nn = 3
        elif b'symmTensor' in content[n]:
            nn = 6
        elif b'tensor' in content[n]:
            nn = 9
        buf = b''.join(content[n+2:n2+1])
        vv = np.array(struct.unpack('{}d'.format(num*nn),
                                    buf[struct.calcsize('c'):num*nn*struct.calcsize('d')+struct.calcsize('c')]))
        if nn > 1:
            data = vv.reshape((num, nn))
        else:
            data = vv
    return data


def split_boundary_content(content):
    """
    split each boundary from boundaryField
    :param content:
    :return: boundary and its content range
    """
    bd = {}
    n = 0
    in_boundary_field = False
    in_patch_field = False
    current_patch = ''
    while True:
        lc = content[n]
        if lc.startswith(b'boundaryField'):
            in_boundary_field = True
            if content[n+1].startswith(b'{'):
                n += 2
                continue
            elif content[n+1].strip() == b'' and content[n+2].startswith(b'{'):
                n += 3
                continue
            else:
                print('no { after boundaryField')
                break
        if in_boundary_field:
            if lc.rstrip() == b'}':
                break
            if in_patch_field:
                if lc.strip() == b'}':
                    bd[current_patch][1] = n-1
                    in_patch_field = False
                    current_patch = ''
                n += 1
                continue
            if lc.strip() == b'':
                n += 1
                continue
            current_patch = lc.strip()
            if content[n+1].strip() == b'{':
                n += 2
            elif content[n+1].strip() == b'' and content[n+2].strip() == b'{':
                n += 3
            else:
                print('no { after boundary patch')
                break
            in_patch_field = True
            bd[current_patch] = [n,n]
            continue
        n += 1
        if n > len(content):
            if in_boundary_field:
                print('error, boundaryField not end with }')
            break

    return bd


def is_binary_format(content, maxline=20):
    """
    parse file header to judge the format is binary or not
    :param content: file content in line list
    :param maxline: maximum lines to parse
    :return: binary format or not
    """
    for lc in content[:maxline]:
        if b'format' in lc:
            if b'binary' in lc:
                return True
            return False
    return False

Boundary = namedtuple('Boundary', 'type, num, start, id')

def is_integer(s):
    try:
        x = int(s)
        return True
    except ValueError:
        return False
    

class FoamMesh(object):
    """ FoamMesh class """
    def __init__(self, path):
        self.path = os.path.join(path, "constant/polyMesh/")
        self._parse_mesh_data(self.path)
        self.num_point = len(self.points)
        self.num_face = len(self.owner)
        self.num_inner_face = len(self.neighbour)
        self.num_cell = max(max(self.owner), max(self.neighbour)) + 1
        self._set_boundary_faces()
        self._construct_cells()
        self.cell_centres = None
        self.cell_volumes = None
        self.face_areas = None

    def read_cell_centres(self, fn):
        """
        read cell centres coordinates from data file,
        the file can be got by `postProcess -func 'writeCellCentres' -time 0'
        :param fn: cell centres file name, eg. '0/C'
        :return: None
        """
        self.cell_centres = parse_internal_field(fn)

    def read_cell_volumes(self, fn):
        """
        read cell volumes from data file,
        the file can be got by `postProcess -func 'writeCellVolumes' -time 0'
        :param fn: cell centres file name, eg. '0/C'
        :return: None
        """
        self.cell_volumes = parse_internal_field(fn)

    def read_face_areas(self, fn):
        """
        read face areas from data file,
        :param fn: cell centres file name, eg. '0/C'
        :return: None
        """
        self.face_areas = parse_internal_field(fn)

    def cell_neighbour_cells(self, i):
        """
        return neighbour cells of cell i
        :param i: cell index
        :return: neighbour cell list
        """
        return self.cell_neighbour[i]

    def is_cell_on_boundary(self, i, bd=None):
        """
        check if cell i is on boundary bd
        :param i: cell index, 0<=i<num_cell
        :param bd: boundary name, byte str
        :return: True or False
        """
        if i < 0 or i >= self.num_cell:
            return False
        if bd is not None:
            try:
                bid = self.boundary[bd].id
            except KeyError:
                return False
        for n in self.cell_neighbour[i]:
            if bd is None and n < 0:
                return True
            elif bd and n == bid:
                return True
        return False

    def is_face_on_boundary(self, i, bd=None):
        """
        check if face i is on boundary bd
        :param i: face index, 0<=i<num_face
        :param bd: boundary name, byte str
        :return: True or False
        """
        if i < 0 or i >= self.num_face:
            return False
        if bd is None:
            if self.neighbour[i] < 0:
                return True
            return False
        try:
            bid = self.boundary[bd].id
        except KeyError:
            return False
        if self.neighbour[i] == bid:
            return True
        return False

    def boundary_cells(self, bd):
        """
        return cell id list on boundary bd
        :param bd: boundary name, byte str
        :return: cell id generator
        """
        try:
            b = self.boundary[bd]
            return (self.owner[f] for f in range(b.start, b.start+b.num))
        except KeyError:
            return ()

    def _set_boundary_faces(self):
        """
        set faces' boundary id which on boundary
        :return: none
        """
        self.neighbour.extend([-10]*(self.num_face - self.num_inner_face))
        for b in self.boundary.values():
            self.neighbour[b.start:b.start+b.num] = [b.id]*b.num

    def _construct_cells(self):
        """
        construct cell faces, cell neighbours
        :return: none
        """
        cell_num = max(max(self.owner), max(self.neighbour)) + 1
        self.cell_faces = [[] for i in range(cell_num)]
        self.cell_neighbour = [[] for i in range(cell_num)]
        for i, n in enumerate(self.owner):
            self.cell_faces[n].append(i)
        for i, n in enumerate(self.neighbour):
            if n >= 0:
                self.cell_faces[n].append(i)
                self.cell_neighbour[n].append(self.owner[i])
            self.cell_neighbour[self.owner[i]].append(n)

    def _parse_mesh_data(self, path):
        """
        parse mesh data from mesh files
        :param path: path of mesh files
        :return: none
        """
        self.boundary = self.parse_mesh_file(os.path.join(path, 'boundary'), self.parse_boundary_content)
        self.points = self.parse_mesh_file(os.path.join(path, 'points'), self.parse_points_content)
        self.faces = self.parse_mesh_file(os.path.join(path, 'faces'), self.parse_faces_content)
        self.owner = self.parse_mesh_file(os.path.join(path, 'owner'), self.parse_owner_neighbour_content)
        self.neighbour = self.parse_mesh_file(os.path.join(path, 'neighbour'), self.parse_owner_neighbour_content)

    @classmethod
    def parse_mesh_file(cls, fn, parser):
        """
        parse mesh file
        :param fn: boundary file name
        :param parser: parser of the mesh
        :return: mesh data
        """
        try:
            with open(fn, "rb") as f:
                content = f.readlines()
                return parser(content, is_binary_format(content))
        except FileNotFoundError:
            print('file not found: %s'%fn)
            return None

    @classmethod
    def parse_points_content(cls, content, is_binary, skip=10):
        """
        parse points from content
        :param content: file contents
        :param is_binary: binary format or not
        :param skip: skip lines
        :return: points coordinates as numpy.array
        """
        n = skip
        while n < len(content):
            lc = content[n]
            if is_integer(lc):
                num = int(lc)
                if not is_binary:
                    data = np.array([ln[1:-2].split() for ln in content[n + 2:n + 2 + num]], dtype=float)
                else:
                    buf = b''.join(content[n+1:])
                    disp = struct.calcsize('c')
                    vv = np.array(struct.unpack('{}d'.format(num*3),
                                                buf[disp:num*3*struct.calcsize('d') + disp]))
                    data = vv.reshape((num, 3))
                return data
            n += 1
        return None


    @classmethod
    def parse_owner_neighbour_content(cls, content, is_binary, skip=10):
        """
        parse owner or neighbour from content
        :param content: file contents
        :param is_binary: binary format or not
        :param skip: skip lines
        :return: indexes as list
        """
        n = skip
        while n < len(content):
            lc = content[n]
            if is_integer(lc):
                num = int(lc)
                if not is_binary:
                    data = [int(ln) for ln in content[n + 2:n + 2 + num]]
                else:
                    buf = b''.join(content[n+1:])
                    disp = struct.calcsize('c')
                    data = struct.unpack('{}i'.format(num),
                                         buf[disp:num*struct.calcsize('i') + disp])
                return list(data)
            n += 1
        return None

    @classmethod
    def parse_faces_content(cls, content, is_binary, skip=10):
        """
        parse faces from content
        :param content: file contents
        :param is_binary: binary format or not
        :param skip: skip lines
        :return: faces as list
        """
        n = skip
        num = -1
        while n < len(content):
            if is_integer(content[n]):
                num = int(content[n])
                break
            n += 1
        if num < 0:
            return None
        if not is_binary:
            return [[int(s) for s in re.findall(b"\d+", ln)[1:]] for ln in content[n + 2:n + 2 + num]]
        n2 = n + 1
        while n2 < len(content):
            if is_integer(content[n2]):
                size = int(content[n2])
                if size > num:  # size = (num-1)*4
                    break
            n2 += 1
        buf = b''.join(content[n+1:n2])
        disp = struct.calcsize('c')
        idx = struct.unpack('{}i'.format(num), buf[disp:disp+num*struct.calcsize('i')])
        buf = b''.join(content[n2+1:])
        pp = struct.unpack('{}i'.format(idx[-1]), buf[disp:disp+idx[-1]*struct.calcsize('i')])
        data = [pp[idx[i]:idx[i+1]] for i in range(num - 1)]
        return data

    @classmethod
    def parse_boundary_content(cls, content, is_binary=None, skip=10):
        """
        parse boundary from content
        :param content: file contents
        :param is_binary: binary format or not, not used
        :param skip: skip lines
        :return: boundary dict
        """
        bd = {}
        num_boundary = 0
        n = skip
        bid = 0
        in_boundary_field = False
        in_patch_field = False
        current_patch = b''
        current_type = b''
        current_nFaces = 0
        current_start = 0
        while True:
            if n > len(content):
                if in_boundary_field:
                    print('error, boundaryField not end with )')
                break
            lc = content[n]
            if not in_boundary_field:
                if is_integer(lc.strip()):
                    num_boundary = int(lc.strip())
                    in_boundary_field = True
                    if content[n + 1].startswith(b'('):
                        n += 2
                        continue
                    elif content[n + 1].strip() == b'' and content[n + 2].startswith(b'('):
                        n += 3
                        continue
                    else:
                        print('no ( after boundary number')
                        break
            if in_boundary_field:
                if lc.startswith(b')'):
                    break
                if in_patch_field:
                    if lc.strip() == b'}':
                        in_patch_field = False
                        bd[current_patch] = Boundary(current_type, current_nFaces, current_start, -10-bid)
                        bid += 1
                        current_patch = b''
                    elif b'nFaces' in lc:
                        current_nFaces = int(lc.split()[1][:-1])
                    elif b'startFace' in lc:
                        current_start = int(lc.split()[1][:-1])
                    elif b'type' in lc:
                        current_type = lc.split()[1][:-1]
                else:
                    if lc.strip() == b'':
                        n += 1
                        continue
                    current_patch = lc.strip()
                    if content[n + 1].strip() == b'{':
                        n += 2
                    elif content[n + 1].strip() == b'' and content[n + 2].strip() == b'{':
                        n += 3
                    else:
                        print('no { after boundary patch')
                        break
                    in_patch_field = True
                    continue
            n += 1

        return bd

class QCFDPlugin:
    def __init__(self, casePath):
        self.casePath = casePath
        self.mesh = None
        self.fieldNames = None
        self.ValidateFilesFolders()
        

    def ValidateFilesFolders(self):
        print("[QCFDPlugin]: Detecting Files...")

        reqDirs = ['system', 'constant', '0']
        for dir_name in reqDirs:
            dir_path = os.path.join(self.casePath, dir_name)
            if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Required directory '{dir_name}' not found in the case path.")

        if not os.path.exists(os.path.join(self.casePath, 'case.foam')):
            raise FileNotFoundError("case.foam file not found in the case path.")

        self.DecompFiles()
        self.ValidatePolyMesh()
        self.ValidateInitCond()
        self.DetectSolver()
        print("[QCFDPlugin]: Loading 3D mesh...")
        self.LoadMesh()
    
    def DecompFiles(self):
        extensions = ['.gz', '.tar', '.zip']
        for root, dirs, files in os.walk(self.casePath):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    print("[QCFDPlugin]: Found a compressed file... Decompressing...")
                    filePath = os.path.join(root, file)
                    if file.endswith('.gz'):
                        self.decompress_gz(filePath)
                    elif file.endswith('.tar'):
                        self.decompress_tar(filePath)
                    elif file.endswith('.zip'):
                        self.decompress_zip(filePath)

    def decompress_gz(self, filePath):
        with gzip.open(filePath, 'rb') as f_in:
            with open(filePath[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(filePath)

    def decompress_tar(self, filePath):
        with tarfile.open(filePath, 'r') as tar:
            tar.extractall(path=os.path.dirname(filePath))
        os.remove(filePath)

    def decompress_zip(self, filePath):
        with zipfile.ZipFile(filePath, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(filePath))
        os.remove(filePath)

    def ValidatePolyMesh(self):
        polyMeshPth = os.path.join(self.casePath, 'constant', 'polyMesh')
        if not os.path.exists(polyMeshPth) or not os.path.isdir(polyMeshPth):
            raise FileNotFoundError("PolyMesh folder not found.")
        print("[QCFDPlugin]: Found PolyMesh 3D... Analyzing...")

    def ValidateInitCond(self):
        initPth = os.path.join(self.casePath, '0')
        self.fieldNames = [file for file in os.listdir(initPth) if os.path.isfile(os.path.join(initPth, file))]
        if not self.fieldNames:
            raise FileNotFoundError("No field names found in the initial state.")
        print(f"[QCFDPlugin]: Found timelapse 0 ... {', '.join(self.fieldNames)} at the initial state found...")


    def LoadMesh(self):
        meshPth = os.path.join(self.casePath, "")
        self.mesh = FoamMesh(meshPth)
        faceAreasPth = os.path.join(meshPth, 'faceAreas')
        if os.path.exists(faceAreasPth):
            print("[QCFDPlugin]: Found faceAreas... [LOADING]")
            self.mesh.read_face_areas(faceAreasPth)
        else:
            print("[QCFDPlugin]: faceAreas NOT FOUND! Calculating them manually...")
            self.mesh.face_areas = [self.CalcFaceArea(self.mesh.faces[i], self.mesh.points) 
                                    for i in range(self.mesh.num_face)]
        cellCentresPth = os.path.join(meshPth, 'cellCentres')
        if os.path.exists(cellCentresPth):
            print("[QCFDPlugin]: Found cellCentres... [LOADING]")
            self.mesh.read_cell_centres(cellCentresPth)
        else:
            print("[QCFDPlugin]: cellCentres NOT FOUND! Calculating them manually...")
            self.CalcCellCent()

    def ExtractFD(self, fieldName, timeStep):
        fieldPth = os.path.join(self.casePath, str(timeStep), fieldName)
        fieldData = parse_internal_field(fieldPth)
        return fieldData

    def ExtractBD(self, fieldName, timeStep):
        boundary_path = os.path.join(self.casePath, str(timeStep), fieldName)
        boundary_data = parse_boundary_field(boundary_path)
        return boundary_data
    
    def ExtractATS(self, fieldName):
        time_steps = self._get_time_steps()
        fieldData_all_steps = {}
        for step in time_steps:
            fieldData_all_steps[step] = self.ExtractFD(fieldName, step)
        return fieldData_all_steps

    def _get_time_steps(self):
        return [d for d in os.listdir(self.casePath) if os.path.isdir(os.path.join(self.casePath, d)) and d.isdigit()]
    
    def ExtractSimPara(self):
        params = {
            "endTime": self._read_control_dict("endTime"),
        }
        return params

    def _read_control_dict(self, parameter):
        control_dict_path = os.path.join(self.casePath, "system/controlDict")
        with open(control_dict_path, 'r') as file:
            content = file.read()

        if parameter == "endTime":
            match = re.search(r'endTime\s+(\d+(\.\d+)?)', content)
            if match:
                return float(match.group(1))
        
        return None
    
    def DetectSolver(self):
        solver_path = os.path.join(self.casePath, "system/controlDict")
        with open(solver_path, 'r') as file:
            for line in file:
                if "application" in line:
                    solver = line.split()[1].strip(';').strip('"')
                    if "SimpleFoam" in solver:
                        print("[QCFDPlugin]: Found main solver --- SimpleFoam --- [COMPATIBLE]")
                        return "SimpleFoam"
                    else:
                        print(f"[QCFDPlugin]: Found main solver --- {solver} --- [INCOMPATIBLE]")
                        raise ValueError("SimpleFoam is only compatible at the moment with QCFDPlugin.")
        return None

    def fvSchemesReader(self):
        fvSchemes = {}
        fvSchemes_path = os.path.join(self.casePath, "system/fvSchemes")
        with open(fvSchemes_path, 'r') as file:
            current_section = None
            for line in file:
                line = line.strip()
                if line.startswith("//") or line == "":
                    continue
                if line.endswith("{"):
                    current_section = line.split()[0]
                    fvSchemes[current_section] = {}
                elif current_section and line.endswith("}"):
                    current_section = None
                elif current_section and ";" in line:
                    parts = line.split(";")[0].split(maxsplit=1)
                    if len(parts) == 2:
                        key, value = parts
                        fvSchemes[current_section][key] = value
        return fvSchemes
    
    def CalcFaceArea(self, face_points, points):
        if len(face_points) < 3:
            return 0.0  
        p1, p2, p3 = [points[face_points[i]] for i in range(3)]
        vec1 = p2 - p1
        vec2 = p3 - p1
        area = np.linalg.norm(np.cross(vec1, vec2)) / 2
        return area
    
    def CalcFaceNorm(self, face_points, points):
        if len(face_points) < 3:
            return np.array([0.0, 0.0, 0.0])
        p1 = points[face_points[0]]
        p2 = points[face_points[1]]
        p3 = points[face_points[2]]
        vec1 = p2 - p1
        vec2 = p3 - p1
        normal = np.cross(vec1, vec2)
        norm = np.linalg.norm(normal)
        if norm != 0:
            normal = normal / norm
        return normal
    
    def CalcFaceCent(self, face_points, points):
        centroid = np.mean([points[pt_index] for pt_index in face_points], axis=0)
        return centroid
    
    def CalcCellCent(self):
        """
        Calculate the centroid of each cell in the mesh.
        """
        cell_centres = []
        for cell_faces in self.mesh.cell_faces:
            face_centroids = [self.CalcFaceCent(self.mesh.faces[face_index], self.mesh.points) for face_index in cell_faces]
            cell_centre = np.mean(face_centroids, axis=0)
            cell_centres.append(cell_centre)
        self.mesh.cell_centres = np.array(cell_centres)
    
    def GetFaceVelocity(self, velocityData, face_index):
        owner_cell = self.mesh.owner[face_index]
        neighbor_cell = self.mesh.neighbour[face_index] if face_index < self.mesh.num_inner_face else -1
        owner_velocity = self.ValidateFaceVel(velocityData, owner_cell)
        neighbor_velocity = self.ValidateFaceVel(velocityData, neighbor_cell)
        if neighbor_cell >= 0:
            face_velocity = (owner_velocity + neighbor_velocity) / 2
        else:
            face_velocity = owner_velocity
    
        return face_velocity
    
    def ValidateFaceVel(self, velocityData, cell_index):
        if 0 <= cell_index < len(velocityData) and velocityData[cell_index].ndim == 1 and velocityData[cell_index].size == 3:
            return velocityData[cell_index]
        else:
            return np.array([0.0, 0.0, 0.0]) 


    def CalcFlux(self, fieldName, timeStep):
        print("[QCFDPlugin]: Calculating fluxes... This may take a few minutes")
        velocityData = self.ExtractFD(fieldName, timeStep)
        fluxes = np.zeros(self.mesh.num_face) 
        for face_index in range(self.mesh.num_face):
            face_area = self.mesh.face_areas[face_index]
            face_normal = self.CalcFaceNorm(self.mesh.faces[face_index], self.mesh.points)
            if face_normal.ndim != 1 or face_normal.size != 3:
                raise ValueError("Face normal must be a 1D array of length 3")
            owner_cell = self.mesh.owner[face_index]
            neighbor_cell = self.mesh.neighbour[face_index] if face_index < self.mesh.num_inner_face else None
            face_velocity = self.GetFaceVelocity(velocityData, face_index)
            if face_velocity.ndim != 1 or face_velocity.size != 3:
                raise ValueError("Face velocity must be a 1D array of length 3")
            face_flux = np.dot(face_velocity, face_normal) * face_area
            if not np.isscalar(face_flux):
                raise ValueError("Face flux must be a scalar value")
            fluxes[owner_cell] += face_flux
            if neighbor_cell is not None:
                fluxes[neighbor_cell] -= face_flux
        print("[QCFDPlugin]: Calculating fluxes... [DONE]")
        return fluxes

    def export(self, timeStep, fieldName):
        number_of_cells = len(self.mesh.cell_faces)
        print(f"[QCFDPlugin]: Found {number_of_cells} Cells... Initializing flux Calculation")
        fluxes = self.CalcFlux(fieldName, timeStep)
        A, b = self.LinearSys(fieldName, timeStep, fluxes)
        print("[QCFDPlugin]: DynexQCFD will take it from here :)")
        return A, b    

    def LinearSys(self, fieldName, timeStep, fluxes):
        print("[QCFDPlugin]: Preparing the linear system...")
        print("[QCFDPlugin]: Loading fvSchemes...")
        fvSchemes = self.fvSchemesReader()
        #print("fvSchemes: ", fvSchemes)
        number_of_cells = self.mesh.num_cell
        if number_of_cells > 10000:
            print("[QCFDPlugin]: Using sparse matrix for large-scale simulation.")
            A = lil_matrix((number_of_cells, number_of_cells))
        else:
            A = np.zeros((number_of_cells, number_of_cells))
        b = np.zeros(number_of_cells)
        for cell_index in range(number_of_cells):
            A[cell_index, cell_index] = -np.sum(np.abs(fluxes[cell_index]))
            b[cell_index] = self.CalcSourceTerm(cell_index, fieldName, timeStep)  
        div_scheme = fvSchemes['divSchemes']['div(phi,U)']
        if div_scheme == 'bounded Gauss linearUpwindV grad(U)':
            for cell_index in range(number_of_cells):
                for face_index in self.mesh.cell_faces[cell_index]:
                    if face_index < self.mesh.num_inner_face:  # Check if it's an internal face
                        neighbor_cell_index = self.mesh.neighbour[face_index]
                        face_flux = fluxes[face_index]  # Access flux for the current face
                        # Apply the bounded Gauss linearUpwindV grad(U) scheme
                        A[cell_index, neighbor_cell_index] -= face_flux
                        A[cell_index, cell_index] += face_flux
        elif div_scheme == 'Gauss linear':
                for face_index in self.mesh.cell_faces[cell_index]:
                    if face_index < self.mesh.num_inner_face:  # Check if it's an internal face
                        neighbor_cell_index = self.mesh.neighbour[face_index]
                        face_flux = fluxes[face_index]  # Access flux for the current face
                        A[cell_index, neighbor_cell_index] -= 0.5 * face_flux
                        A[cell_index, cell_index] += 0.5 * face_flux

        self.BoundaryCond(A, b, fieldName, timeStep)
        if number_of_cells > 10000:
            A = A.tocsr()
            b = csr_matrix(b).transpose()
            save_npz('A.npz', A)
            save_npz('b.npz', b)

        print("[QCFDPlugin]: Linear system prepared.")
        return A, b
    
    def BoundaryCond(self, A, b, fieldName, timeStep):
        print("[QCFDPlugin]: Applying boundary conditions...")
        boundary_data = self.ExtractBD(fieldName, timeStep)
        for boundary_name, data in boundary_data.items():
            boundary_type = data.get('type')
            boundary_value = data.get('value')
            if boundary_type and boundary_value is not None:
                for face_index in self.mesh.boundary_cells(boundary_name):
                    cell_index = self.mesh.owner[face_index]
                    if boundary_type == 'Dirichlet':
                        A[cell_index, :] = 0
                        A[cell_index, cell_index] = 1
                        b[cell_index] = boundary_value
                    elif boundary_type == 'Neumann':
                        normal_vector = self.mesh.face_normals[face_index]
                        for d in range(len(normal_vector)):
                            A[cell_index, cell_index] += normal_vector[d]
                            b[cell_index] += normal_vector[d] * boundary_value
                    else:
                        continue
            else:
                continue
        print("[QCFDPlugin]: Boundary conditions applied.")
    
    def CalcSourceTerm(self, cell_index, fieldName, timeStep):
        if fieldName == "U":
            velocityData = self.ExtractFD("U", timeStep)
            pressureData = self.ExtractFD("p", timeStep)
            if isinstance(velocityData, np.ndarray) and velocityData.ndim > 1:
                fvSchemes = self.fvSchemesReader()
                divSchemes = fvSchemes['divSchemes'].get('div(phi,U)', 'default')
                pressureGrad = np.gradient(pressureData, self.mesh.cell_centres, axis=0)
                sourceTerm = -pressureGrad
                if "omega" in self.fieldNames and "nut" in self.fieldNames:
                    print("[QCFDPlugin: Calculate the source term using the SST k-omega turbulence model...]")
                    omegaData = self.ExtractFD("omega", timeStep)
                    nutData = self.ExtractFD("nut", timeStep)
                    velocityGrad = np.array([np.gradient(velocityData[:, d], self.mesh.cell_centres, axis=0) for d in range(3)]).T
                    if divSchemes == 'bounded Gauss linearUpwindV grad(U)':
                        sourceTerm -= omegaData * np.sum(velocityGrad, axis=1) + nutData * np.sum(velocityGrad, axis=1)
                return sourceTerm[cell_index] if cell_index < len(sourceTerm) else 0.0
            else:
                return 0.0
        else:
            return 0.0

        

