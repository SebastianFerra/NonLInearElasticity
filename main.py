from ngsolve import *
from netgen.geom2d import SplineGeometry
from netgen.occ import *

import problems
import numpy as np
import params
import pickle
import alive_progress

class Gel():
    """
    Only properties for a specific gel, not problem specific
    """
    def __init__(self, gel):
        self.phi0 = gel['phi0']
        self.chi = gel['chi']
        self.G = gel['G']
        
        

class Geomerty():
    def __init__(self, geom_path):
        self.geometry = self.get_geom(geom_path)

    def get_geom(self, geom_path):
        return OCCGeometry(geom_path)   

    def mesher(self,h=0.1):
        self.geometry = self.geometry.GenerateMesh(maxh=h)

        return self.geometry
        

class GelSolver():
    """
    Here we define the problem specific properties (constants and such)
    """
    def __init__(self, gel, geom, BC,h=1):
        self.gel = Gel(gel)
        self.geom = Geomerty(geom)
        self.mesh = Mesh(self.geom.mesher(h))
        self.fes = VectorH1(self.mesh, order=2, dirichletx = BC["x"], dirichlety = BC["y"], dirichletz = BC["z"])
        self.u = self.fes.TrialFunction()
        self.v = self.fes.TestFunction()
        self.F = Id(3) + Grad(self.u)
        self.BF = BilinearForm(self.fes, symmetric = True)
        self.Assembled = False
        self.KBTV = params.KBTV
        
    def Assemble_Bilinear_Form(self, form):
        """
        Remember to set parameters as NGsolve parameter objects
        """
        J = Det(self.F)
        phi0 = self.gel.phi0
        phi = self.gel.phi0
        gamma = self.gel.G/self.KBTV
        print(gamma)
        chi = self.gel.chi
        F = self.F
        if form == "EDP":
            print("Assembling using theorical variation")
            """
            Analitically calculated variation of the energy
            """
            # add F term
            self.BF += gamma * InnerProduct(self.F,Grad(self.v)) * dx
            # add non-linear term
            self.BF += -(1/params.N)*InnerProduct(Inv(self.F).trans,Grad(self.v)) * dx
            self.BF += J * InnerProduct(Inv(self.F).trans,Grad(self.v)) * log(1-self.gel.phi0/J) * dx
            self.BF += self.gel.phi0  * InnerProduct(Inv(self.F).trans,Grad(self.v)) * dx
            self.BF += self.gel.phi0**2 * self.gel.chi * InnerProduct(Inv(self.F).trans,Grad(self.v)) * dx
        elif form == "Functional":
            print("Assembling using numerical variation")
            """
            Numerically calculated variation of the energy
            """
    
            H = (J - phi0)*log(1-phi)  + phi0 * chi*(1-phi) + phi0/1000*log(phi)
            C = F.trans * F
            BF =  0.5*gamma*(Trace(C)) + H
            self.BF += Variation(BF.Compile() * dx)
        self.Assembled = True
    
    def NewtonSolver(self, MAX_ITS=100, tol=1e-6, damp = 0.5):
        
        if not self.Assembled:
            raise Exception("Bilinear form not assembled")
        self.gfu = GridFunction(self.fes)
        self.gfu.vec[:] = 0

        self.res = self.gfu.vec.CreateVector()
        self.w = self.gfu.vec.CreateVector()
        self.history = GridFunction(self.fes, multidim = 0)

        # here one should calculate the lambda target and stuff
        # self.lams = np.linspace(0,1,5) # newton dampers
        self.lams = [1]
        for lam in self.lams:
            for it in range(MAX_ITS):
                print(self.BF.Energy(self.gfu.vec))
                self.BF.Apply(self.gfu.vec, self.res)
                self.BF.AssembleLinearization(self.gfu.vec)
                self.inv = self.BF.mat.Inverse(self.fes.FreeDofs())
                damp = 0.5
                self.w.data = damp*self.inv * self.res
                self.gfu.vec.data -=  self.w
                
                stopcritval = sqrt(abs(InnerProduct(self.w,self.res)))
                if stopcritval < tol:
                    break

                self.history.AddMultiDimComponent(self.gfu.vec)
                    
        return self.gfu
problem = problems.problem1
problem_n = problem.pop(-1)

solver = GelSolver(*problem)

# solver.Assemble_Bilinear_Form("Functional")
form = "Functional"
# form = "EDP"
solver.Assemble_Bilinear_Form(form)

u = solver.NewtonSolver(MAX_ITS=250)


solver.mesh.ngmesh.Save(f"Sol_Problem1/mesh.vol")
pickle.dump(solver.mesh, open(f"Sol_Problem{problem_n}/mesh.pkl", "wb"))
pickle.dump(solver.history, open(f"Sol_Problem{problem_n}/history_{form}.pkl", "wb"))
pickle.dump(u, open(f"Sol_Problem{problem_n}/u_{form}.pkl", "wb"))
