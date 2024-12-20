# import netgen 
from netgen.occ import *
from netgen.csg import *
from netgen.geom2d import SplineGeometry
## problem 1
# create a box
d = 23.5
l = 90
w = 3.0
box = Box( (0,0,0),(l/2,d/2,w/2) )
# Name the faces
box.faces.Max(Y).name="top"
box.faces.Min(Y).name = "bottom"
box.faces.Max(X).name = "back"
box.faces.Min(X).name = "front"
box.faces.Max(Z).name = "right"
box.faces.Min(Z).name = "left"

L = 90
d = 23.5
L3 = 3.0
box.WriteStep("geo_3D_bonded.stp")


# Problem 2
w = 3.0
l = 15

rect = SplineGeometry()
pnts = [(0,0), (l,0), (l,w), (0,w)]
p1,p2,p3,p4 = [rect.AppendPoint(*pnt) for pnt in pnts]
curves = [[["line",p1,p2],"bottom"],
        [["line",p2,p3],"right"],
        [["line",p3,p4],"top"],
        [["line",p4,p1],"left"]]
[rect.Append(c,bc=bc, leftdomain=1, rightdomain=0) for c,bc in curves]

# save the geometry
import pickle   
with open("geo_2D_bonded.pkl", "wb") as f:
    pickle.dump(rect, f)

