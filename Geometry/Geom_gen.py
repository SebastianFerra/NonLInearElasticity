# import netgen 
from netgen.occ import *
from netgen.csg import *

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

left  = Plane (Pnt(0,0,0), Vec(-1,0,0) ).bc('left')
right = Plane (Pnt(0.5*L,0,0), Vec( 1,0,0) )
bot = Plane (Pnt(0,0,0), Vec(0,-1,0) ).bc('bot')
top  = Plane (Pnt(0,0.5*d,0), Vec(0, 1,0) )
back   = Plane (Pnt(0,0,0), Vec(0,0,-1) ).bc('back')
front   = Plane (Pnt(0,0,0.5*L3), Vec(0,0, 1) )

brick = left * right * front * back * bot * top

geo = CSGeometry()
geo.Add (brick)

box.WriteStep("geo.stp")

