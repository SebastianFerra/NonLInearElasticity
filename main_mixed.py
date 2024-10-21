 

from ngsolve import *
from netgen.geom2d import SplineGeometry

from netgen.occ import *
import netgen.meshing as ngm
# from ngsPETSc import NonLinearSolver
# from mpi4py.MPI import COMM_WORLD
import problems
import numpy as np
import params
import pickle
from time import time

## get problem parameters and geometry
problem = problems.problem1

phi0 = problem[0]['phi0']
chi = problem[0]['chi']
G = problem[0]['G']
geom = problem[1]
BC = problem[2]
h = 2
ord = 2
N = params.N
KBTV = params.KBTV
form = "EDP" # EDP //functional

## Generate mesh and geometry ### add parallel stuff
def mesher(geom, h):
    geo = OCCGeometry(geom)
    mesh = Mesh(geo.GenerateMesh(maxh=h))
    return mesh
mesh = mesher(geom, h)


def F(u):
    return Id(3) + Grad(u)


def Norm(vec):
    return InnerProduct(vec, vec)**0.5

div_2D = lambda A: CoefficientFunction(A.Diff(x)+A.Diff(y)+A.Diff(z))
def div_custom(A):
    return CoefficientFunction((div_2D(A[0,:]), div_2D(A[1,:]), div_2D  (A[2,:])))

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

def Gel_energy_mixed(F,v,P,T):
    gamma = G/KBTV
    J = Det(F)
    phi = phi0/J
    invF = Inv(F)
    tens_eq = Trace((gamma*F - (phi0 / N) * invF + log(1 - phi) * J * invF + phi0 * invF + (chi/J) * (phi0**2) * invF - P).trans * T)
    print("int")
    div_eq = InnerProduct(div_custom(P),v) 
    
    return tens_eq + div_eq


## Generate spaces and forms
fesU = VectorH1(mesh, order=ord, dirichletx = BC["x"], dirichlety = BC["y"], dirichletz = BC["z"])
fesTensorP = MatrixValued(HDiv(mesh, order=ord-1))
fes = fesU*fesTensorP
u,P = fes.TrialFunction()
P = P[:,:,0]
v,T = fes.TestFunction()
T = T[:,:,0]
BF = BilinearForm(fes)
F = Id(3) + Grad(u)

## Assemble forms
def Assemble_Bilinear_Form(BF, F,v=None,P=None,T=None, form = "Mixed"):
    
    if form == "EDP":
        BF += Gel_energy_EDP(F).Compile() * dx
        return BF
    elif form == "Mixed":
        
        BF += Gel_energy_mixed(F,v,P,T).Compile() * dx
        
        return BF

BF = Assemble_Bilinear_Form(BF, F,v,P,T)

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
t1 =  time()
gfu, history = Solver_freeswell(BF, gfu)
print("Time on ngsolve newton:", abs(t1-time()))

#gfu = GridFunction(fes)
# #gfu.vec[:] = 0
# t1 = time() 
# gfu = petsc_solver(fes,BF, gfu)
# print("Time on snes newton:", abs(t1-time()))

# pickle the results, history and mesh for later use
#pickle.dump(history, open(f"Sol_Problem{problem[-1]}/history_{form}.p", "wb"))
# pickle.dump(gfu, open(f"Sol_Problem{problem[-1]}/gfu_{form}.p", "wb"))
# pickle.dump(mesh, open(f"Sol_Problem{problem[-1]}/mesh.p", "wb"))
vtk = VTKOutput(ma=mesh, coefs=[gfu], names=["u"], filename=f"freeswekk_EDP_{h}", subdivision=0)
vtk.Do() 

