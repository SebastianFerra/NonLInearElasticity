# import netgen 
from netgen.occ import *

## problem 1
# create a box
d = 1.5
l = 45
w = 11.75
box = Box( (0,0,0),(l/2,d/2,w/2) )
# Name the faces
box.faces.Max(Y).name="top"
box.faces.Min(Y).name = "bottom"
box.faces.Max(X).name = "back"
box.faces.Min(X).name = "front"
box.faces.Max(Z).name = "right"
box.faces.Min(Z).name = "left"

box.WriteStep("box.stp")

