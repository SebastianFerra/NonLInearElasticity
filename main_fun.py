from ngsolve import *
from netgen.geom2d import SplineGeometry
from netgen.occ import *

import problems
import numpy as np
import params
import pickle


## get problem parameters and geometry
problem = problems.problem1

phi0 = problem[0]['phi0']
chi = problem[0]['chi']
G = problem[0]['G']
geom = problem[1]
BC = problem[2]
h = 1
ord = 2
N = params.N
KBTV = params.KBTV

## Generate mesh and geometry
def mesher(geom, h):
    geo = OCCGeometry(geom)
    mesh = Mesh(geo.GenerateMesh(maxh=h))
    return mesh

mesh = mesher(geom, h)

def F(u):
    return Id(3) + Grad(u)

def Gel_energy_functional(F):
    gamma = G/KBTV
    J = Det(F)
    phi = phi0/J
    H = (J - phi0)*log(1-phi)  + phi0 * chi*(1-phi) + phi0/1000*log(phi)
    return 0.5*gamma* Trace(F.trans*F ) + H

def Gel_energy_EDP(F): ## |F|^2 + H => gamma F:Gradv + H'*J'
    # ddet(A(t))/dt = det(A(t))*trace(A^-1(t)*Grad (v))
    gamma = G/KBTV
    J = Det(F)
    phi = phi0/J
    dv = Grad(v)
    invF = Inv(F)
    H_prime = -phi/N + log(1-phi) + phi + chi*phi**2
    edp = gamma * InnerProduct(F,dv) + H_prime * J * Trace(invF*dv)
    return edp

## Generate spaces and forms
fes = VectorH1(mesh, order=ord, dirichletx = BC["x"], dirichlety = BC["y"], dirichletz = BC["z"])
u = fes.TrialFunction()
v = fes.TestFunction()
BF = BilinearForm(fes)
F = Id(3) + Grad(u)

## Assemble forms
def Assemble_Bilinear_Form(BF, F, form):
    if form == "Functional":
        BF += Variation(Gel_energy_functional(F).Compile()*dx)
        return BF
    elif form == "EDP":
        BF += Gel_energy_EDP(F) * dx
        return BF

form = "Functional"
BF = Assemble_Bilinear_Form(BF, F, form)

def Solver_freeswell(BF, gfu, tol=1e-8, maxiter=250, damp = 0.5):
    """
    Solves the problem
    """
    res = gfu.vec.CreateVector()
    w = gfu.vec.CreateVector()
    history = GridFunction(fes, multidim = 0)
    # here we may need to add another loop 
    for iter in range(maxiter):
        # Prints before the iteration: number of it, residual, energy
        print("Energy: ", BF.Energy(gfu.vec), "Residual: ", sqrt(abs(InnerProduct(res,res))), "Iteration: ", iter)
        BF.Apply(gfu.vec, res)
        BF.AssembleLinearization(gfu.vec)
        inv = BF.mat.Inverse(freedofs = fes.FreeDofs())        
        w.data = damp * inv * res
        gfu.vec.data -= w
        history.AddMultiDimComponent(gfu.vec)
        if sqrt(abs(InnerProduct(w,res))) < tol:
            print("Converged")
            break
    return gfu, history

gfu = GridFunction(fes)
gfu.vec[:] = 0
gfu, history = Solver_freeswell(BF, gfu)

# pickle the results, history and mesh for later use
pickle.dump(history, open(f"Sol_Problem{problem[-1]}/history_{form}.p", "wb"))
pickle.dump(gfu, open(f"Sol_Problem{problem[-1]}/gfu_{form}.p", "wb"))
pickle.dump(mesh, open(f"Sol_Problem{problem[-1]}/mesh.p", "wb"))


        

