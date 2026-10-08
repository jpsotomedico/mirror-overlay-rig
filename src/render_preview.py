import trimesh, numpy as np, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
from PIL import Image, ImageDraw, ImageFont
S2=math.sqrt(2); AZ=44.5; ya=6-2*S2
parts=[]
for k,c in {'body':(170,170,170),'carR':(205,140,95),'carL':(205,140,95),'collar':(95,130,205)}.items():
    m=trimesh.load(os.path.join(HERE,'build',f'asm_{k}.stl')); parts.append((m.vertices[m.faces],c))
def slab(p0,p1,z0,z1,t=0.6):
    d=np.array([p1[0]-p0[0],p1[1]-p0[1]]); n=np.array([-d[1],d[0]])/np.linalg.norm(d)*t
    v=[]; 
    for (x,y) in (p0,p1):
        for z in (z0,z1): v.append((x+n[0],y+n[1],z))
    b=trimesh.creation.box(); 
    q=[(p0[0],p0[1],z0),(p1[0],p1[1],z0),(p1[0],p1[1],z1),(p0[0],p0[1],z1)]
    return np.array([[q[0],q[1],q[2]],[q[0],q[2],q[3]]])
mir=[]
for s in (1,-1):
    mir.append(slab((0.0,ya),(s*46/S2,ya+46/S2),AZ-29,AZ+29))
    mir.append(slab((s*(62-37.5/S2),26-37.5/S2),(s*(62+37.5/S2),26+37.5/S2),AZ-37.5,AZ+37.5))
parts.append((np.concatenate(mir),(120,200,240)))
def render(az,el,W=760,H=520,scale=2.9):
    a,e=math.radians(az),math.radians(el)
    R1=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
    R2=np.array([[1,0,0],[0,math.cos(e),-math.sin(e)],[0,math.sin(e),math.cos(e)]])
    R=R2@R1
    img=np.full((H,W,3),255,np.uint8); zb=np.full((H,W),-1e9)
    L=np.array([0.35,0.45,0.82]); L/=np.linalg.norm(L)
    for tris,col in parts:
        P=(tris-np.array([0,20,45]))@R.T   # view: x right, y up(after el), z toward viewer
        for t,t0 in zip(P,tris):
            n=np.cross(t0[1]-t0[0],t0[2]-t0[0]); nn=np.linalg.norm(n)
            if nn==0: continue
            sh=0.45+0.55*abs(np.dot(n/nn,R.T@L))
            xs=W/2+t[:,0]*scale; ys=H/2-t[:,2]*scale; zs=t[:,1]
            x0,x1=int(max(0,xs.min())),int(min(W-1,xs.max())+1); y0,y1=int(max(0,ys.min())),int(min(H-1,ys.max())+1)
            if x1<=x0 or y1<=y0: continue
            gx,gy=np.meshgrid(np.arange(x0,x1)+.5,np.arange(y0,y1)+.5)
            d=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
            if abs(d)<1e-9: continue
            w0=((ys[1]-ys[2])*(gx-xs[2])+(xs[2]-xs[1])*(gy-ys[2]))/d
            w1=((ys[2]-ys[0])*(gx-xs[2])+(xs[0]-xs[2])*(gy-ys[2]))/d
            w2=1-w0-w1; m=(w0>=-1e-6)&(w1>=-1e-6)&(w2>=-1e-6)
            z=-(w0*zs[0]+w1*zs[1]+w2*zs[2])
            sub=zb[y0:y1,x0:x1]; upd=m&(z>sub); sub[upd]=z[upd]
            img[y0:y1,x0:x1][upd]=(np.array(col)*sh).astype(np.uint8)
    return img
# camera behind at -y. view from front-left above: rotate so viewer looks from +y side
a=render(205,22); b=render(0,62,scale=2.9)
out=Image.new('RGB',(1540,560),'white'); out.paste(Image.fromarray(a),(0,40)); out.paste(Image.fromarray(b),(780,40))
d=ImageDraw.Draw(out)
try: f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
except: f=None
d.text((20,8),'From the subject side. Gray: body. Orange: carriers. Blue: mirrors.',fill='black',font=f)
d.text((800,8),'From above and behind (lens collar in blue, nearest you)',fill='black',font=f)
out.save(os.path.join(HERE,'..','images','preview.png'))
