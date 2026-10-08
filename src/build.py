"""Four-mirror overlay rig, prototype v1.
World axes: x = right, y = forward (toward subject), z = up. Floor bottom at z=0.
All sizes in mm. Change the numbers in PARAMETERS and re-run to regenerate the STL files.
"""
import math, numpy as np, trimesh
from manifold3d import Manifold, CrossSection

# ---------------- PARAMETERS ----------------
SEAT_D      = 52.6   # pocket that the metal 49mm filter frame (glass removed) glues into
SEAT_DEPTH  = 4.0
COLLAR_HOLE = 45.0   # clear opening for the lens
AXIS_Z      = 44.5   # lens center height above floor bottom
MIRROR_T    = 2.0    # mirror glass thickness
V_LEN, V_H  = 46.0, 58.0   # V mirrors: 46 mm side runs along the 45 degree face, 58 mm tall
O_LEN, O_H  = 75.0, 75.0   # outer mirrors 75 x 75
XO, YC      = 62.0, 26.0   # center of right outer mirror's reflecting face
FLOOR_T     = 4.0
BASE_T      = 3.0    # carrier base thickness
M3          = 3.4    # clearance hole
NUT_F       = 5.8    # M3 nut across flats (with clearance)
NUT_DEPTH   = 2.6
SEG         = 96

S2 = math.sqrt(2)

def ccw(pts):
    a = sum(pts[i][0]*pts[(i+1)%len(pts)][1]-pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))
    return pts if a > 0 else pts[::-1]

def box(x0, y0, z0, x1, y1, z1):
    return Manifold.cube([x1-x0, y1-y0, z1-z0]).translate([x0, y0, z0])

def cyl_z(x, y, z0, h, d, seg=SEG):
    return Manifold.cylinder(h, d/2, d/2, seg).translate([x, y, z0])

def poly_xy(pts, z0, z1):
    return CrossSection([ccw(pts)]).extrude(z1-z0).translate([0, 0, z0])

def poly_xz(pts, y0, y1):
    """2D (x,z) polygon extruded along y from y0 to y1."""
    m = CrossSection([ccw(pts)]).extrude(y1-y0)          # (p,q,r)
    m = m.rotate([90, 0, 0])                         # -> (p,-r,q)
    return m.translate([0, y1, 0])

def hex_nut_pocket(x, y, depth):
    r = NUT_F/math.sqrt(3)
    return Manifold.cylinder(depth, r, r, 6).translate([x, y, -0.01])

# ---------- carrier (right side), local coords: u along mirror, yl = in front of backing wall ----------
PIVOT = (-22.0, -9.0)
CLAMP = (24.0, -9.0)
SLOT_ANG = 4.5   # +/- degrees of convergence adjustment

def carrier_local():
    half = 37.0
    base = box(-half, -18, 0, half, 3, BASE_T)
    wall = box(-half, -4, 0, half, 0, BASE_T + 70)
    m = base + wall
    rib_pts = [(-4, BASE_T), (-16, BASE_T), (-4, 60)]
    for u in (-34.0, 0.0, 34.0):
        r = CrossSection([ccw(rib_pts)]).extrude(3).rotate([90, 0, 0]).rotate([0, 0, 90])  # p->y, q->z, extrude->x
        m = m + r.translate([u-1.5, 0, 0])
    m = m - cyl_z(PIVOT[0], PIVOT[1], -1, BASE_T+2, M3)
    pts = []
    for a in (-SLOT_ANG, SLOT_ANG):
        R = CLAMP[0]-PIVOT[0]
        pts.append(cyl_z(PIVOT[0]+R*math.cos(math.radians(a)), PIVOT[1]+R*math.sin(math.radians(a)), -1, BASE_T+2, M3))
    m = m - Manifold.batch_hull(pts)
    return m

E_V = np.array([1, -1])/S2
O_R = np.array([XO, YC]) + MIRROR_T*E_V   # backing face center

def to_world_right(p):
    c = s = 1/S2
    u, yl = p
    return (O_R[0] + c*u - s*yl, O_R[1] + s*u + c*yl)

def carrier_world_right():
    return carrier_local().rotate([0, 0, 45]).translate([O_R[0], O_R[1], FLOOR_T])

# ---------------- body: floor + back plate + V block ----------------
def body():
    floor = box(-102, -16, 0, 102, 58, FLOOR_T)
    plate = box(-33, -5, 0, 33, 0, 90)
    # V block: region y >= |x| + 6 ; mirror faces at 45 degrees
    ap = 6.0
    vblock = poly_xy([(0, ap), (33.5, ap+33.5), (-33.5, ap+33.5)], FLOOR_T-0.01, 72)
    ledge_ap = ap - 1.5*S2
    ledge = poly_xy([(0, ledge_ap), (ap+33.5-ledge_ap, ap+33.5), (-(ap+33.5-ledge_ap), ap+33.5)],
                    FLOOR_T-0.01, AXIS_Z - V_H/2)
    m = floor + plate + vblock + ledge
    # teardrop lens opening (prints without support in the vertical wall)
    r = 24.0
    pts = [(r*math.cos(math.radians(a)), AXIS_Z + r*math.sin(math.radians(a))) for a in np.linspace(-225, 45, 80)]
    pts.append((0, AXIS_Z + r*S2))
    m = m - poly_xz(pts, -6, 1)
    # collar bolt holes on a 31 mm radius, diagonals
    for sx in (-1, 1):
        for sz in (-1, 1):
            hx, hz = sx*31/S2, AXIS_Z + sz*31/S2
            hole = [(hx + 1.7*math.cos(t), hz + 1.7*math.sin(t)) for t in np.linspace(0, 2*math.pi, 32, endpoint=False)]
            m = m - poly_xz(hole, -6, 1)
    # pivot + clamp holes with nut pockets underneath, both sides
    for side in (1, -1):
        for p in (PIVOT, CLAMP):
            wx, wy = to_world_right(p)
            wx *= side
            m = m - cyl_z(wx, wy, -1, FLOOR_T+2, M3) - hex_nut_pocket(wx, wy, NUT_DEPTH)
    return m

# ---------------- lens collar ----------------
def collar():
    T = 7.0
    m = cyl_z(0, 0, 0, T, 74)
    m = m - cyl_z(0, 0, -1, T+2, COLLAR_HOLE)
    m = m - cyl_z(0, 0, T-SEAT_DEPTH, SEAT_DEPTH+1, SEAT_D)   # seat pocket on top face
    for k in range(4):
        a0 = 45 + 90*k
        pts = [cyl_z(31*math.cos(math.radians(a0+d)), 31*math.sin(math.radians(a0+d)), -1, T+2, M3) for d in (-8, 8)]
        m = m - Manifold.batch_hull(pts)
    return m

def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.array(mesh.vert_properties)[:, :3], faces=np.array(mesh.tri_verts), process=True)

def flat(m):
    b = m.bounding_box()
    return m.translate([-b[0], -b[1], -b[2]])

if __name__ == "__main__":
    import os, json
    here = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(here, "..", "stl"); asm_dir = os.path.join(here, "build"); os.makedirs(out, exist_ok=True); os.makedirs(asm_dir, exist_ok=True)
    cr = carrier_local()
    parts = {
        "1_body": body(),
        "2_outer_mirror_carrier_RIGHT": cr,
        "3_outer_mirror_carrier_LEFT": cr.mirror([1, 0, 0]),
        "4_lens_collar": collar(),
    }
    report = {}
    for name, m in parts.items():
        t = to_trimesh(flat(m))
        t.export(f"{out}/{name}.stl")
        report[name] = dict(watertight=bool(t.is_watertight), volume_cm3=round(t.volume/1000, 1),
                            size_mm=[round(v, 1) for v in t.extents], genus=m.genus())
    print(json.dumps(report, indent=1))
    # assembled preview meshes
    asm = {
        "body": body(),
        "carR": carrier_world_right(),
        "carL": carrier_world_right().mirror([1, 0, 0]),
        "collar": collar().rotate([90, 0, 0]).translate([0, -5, AXIS_Z]),
    }
    print("collision body/carR:", (asm["body"] ^ asm["carR"]).volume(), " body/collar:", (asm["body"] ^ asm["collar"]).volume(),
          " carR/collar:", (asm["carR"] ^ asm["collar"]).volume())
    for k, m in asm.items():
        to_trimesh(m).export(os.path.join(asm_dir, f"asm_{k}.stl"))
