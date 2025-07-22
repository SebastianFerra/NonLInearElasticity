import sys
import numpy as np
import scipy.optimize
import datetime
import csv

from ngsolve import *
from ngsolve.webgui import Draw

class gel_debonded2D:
    def __init__(self, length, thickness, delta, shear_modulus, entropic_unit, phi0, chi, mass_density):
        self.L = length      # measured in mm
        self.d = thickness    # measured in mm
        self.delta = delta   # dimensionless
        self.G = shear_modulus               # measured in MPa
        self.entropic_unit = entropic_unit  # measured in MPa
        self.phi0 = phi0
        self.chi =  chi  
        self.mass_density = mass_density

        self.gamma = self.G/self.entropic_unit # 0.0009516837481698391 self.compute_gamma( lamb =1.4874):
        self.lambda_target = 1.99  # 0.13/136.6 = gammafun(1.99)  up to rounding errors               

        def auxIsotropic(s):
            return s*self.dH(s*s*s) + self.gamma
        self.lambda_iso = scipy.optimize.fsolve(auxIsotropic, 1.7)

        def auxEnergyDensity(lambda1, lambda2, lambda3):
            gel=self
            phi0=gel.phi0;
            G=gel.G;
            chi=gel.chi;
            nu=gel.entropic_unit

            J= lambda1*lambda2*lambda3;
            phi = phi0/J;
            return 0.5*G*(lambda1**2 + lambda2**2 + lambda3**2) + nu*((J-phi0)*np.log(1-phi) + phi0*chi*(1-phi))

        lambda_iso = self.lambda_iso
        self.reference_energy_density = auxEnergyDensity(lambda_iso, lambda_iso, lambda_iso)                

    def phi(self, J):
        return self.phi0/J

    def H(self, J):
        return (J - self.phi0)*log(1-self.phi(J))  + self.phi0 * self.chi*(1-self.phi(J))

    def dH(self, J):
        return self.phi(J) + np.log(1-self.phi(J)) + self.chi * self.phi(J)**2
    
    # compute gamma using uniaxial approximation
    # def gammafun(self,lamb):
    #    return -self.dH(lamb)/lamb
    
    def Gfun(self, lamb):
        nu = self.entropic_unit
        return (-self.dH(lamb)/lamb)*nu
    
    
    # energy density
    def W(self, F):
        J = Det(F)
        C = F.trans* F
        
        gel=self
        G=gel.G
        nu=gel.entropic_unit
        reference_energy_density =  gel.reference_energy_density       
        
        return 0.5*G*Trace(C) + nu*gel.H(J) - reference_energy_density

class Solve_gel_bonded_imported_mesh:
    def __init__(self, gel, folder_name_suffix, order=2):         # corner_refinement capability not yet activated
        self.gel = gel
        self.order = order
        self.folder_name_suffix = folder_name_suffix
        self.filename_suffix = "_d={:.2f}_delta={:.3f}".format(self.gel.d, self.gel.delta)

    def add_mesh(self, index_delta):
        mesh_file = 'meshes' + self.folder_name_suffix + '/mesh{}.vol'.format(index_delta);
        self.mesh = Mesh(mesh_file)

    def Space(self):
        self.fes = VectorH1(self.mesh, order=self.order, dirichlet="bonded_interface", dirichlety="debonded_interface")
        # self.fes = VectorH1(self.mesh, order=self.order, dirichlet="bonded_interface")
        print('nDoF = {}'.format(self.fes.ndof))

    def model(self, prefactor_obstacle):
        u  = self.fes.TrialFunction()
        I = Id(self.mesh.dim)
        F = I + Grad(u)
        # Ft = I+ Grad(u).Trace()
        
        gravity = CoefficientFunction((0,-9.8))
        rho = self.gel.mass_density
        AA = prefactor_obstacle
        
        def negpart(var):
            #return max(-var,0)
            return (sqrt(var**2)-var)*0.5        
        
        # hydrogel model        
        self.a = BilinearForm(self.fes, symmetric=False)
        self.a += Variation(  self.gel.W(F).Compile() * dx)         
        self.a += Variation( -rho*InnerProduct(gravity, u)*dx )   #in [mJ] the gravitational energy
        self.a += Variation(  AA*negpart(y+u[1])**2 * dx)
        
    def Solve_incremental_softening(self, prefactor_obstacle):
        self.Space()
        self.gfu = GridFunction(self.fes)
        self.gfu.vec[:] = 0
        
        G_end = self.gel.G
        G0 = self.gel.Gfun(1.1);
        nIterations = 15
        self.gel.G = Parameter(G0)
        self.model(prefactor_obstacle)
        
        #From 0 to nIterations-1
        lambda_list = np.linspace(1.1, self.gel.lambda_target, nIterations, endpoint = False)
        #gamma_list = [self.gel.gammafun(la) for la in lambda_list]
        G_list = [self.gel.Gfun(la) for la in lambda_list]
        
        # final iteration
        G_list.append(G_end)
              
        filename='gridfunctions'+self.folder_name_suffix+\
            '/result_debonded2D'+self.filename_suffix+\
            "_order={}".format(self.order)

        tol=1e-3; maxits=100;
        
#        self.w = GridFunction(self.fes); self.w.vec[:] = 0
#        self.w0 = GridFunction(self.fes); self.w0.Set((x,y,z))   
        for numIteration in range(nIterations+1):
            G_i = G_list [numIteration]
            print("*** Iteration #", numIteration, ". Shear modulus G = ", G_i)

            if numIteration==nIterations:
                tol=1e-6; maxits=500;
            
            self.gel.G.Set(G_i)
            self.gfu, _,_ = SolveNonlinearMinProblem(a= self.a, gfu = self.gfu,\
                        FreeDofs =self.fes.FreeDofs(), maxits=maxits, tol=tol)

            #Draw(self.gfu, deformation=True)            
            #self.gfu.Save(filename + "_iter={:2}.gfu".format(numIteration))
            self.gfu.Save(filename + '.gfu')

            numIteration +=1 

#            self.w = self.w0 + self.gfu
#            print("Deformed point on the middle of the top surface: ", self.w(self.mesh(0,self.gel.d)))  

        # vtk = VTKOutput(ma=self.mesh,
        #         coefs=[self.gfu],
        #         names = ["displacement"],
        #         filename=filename, subdivision=3)   
        # vtk.Do()


def SolveNonlinearMinProblem(a, gfu, FreeDofs, tol=1e-08, maxits=50):#, scenes=None):
    res = gfu.vec.CreateVector()
    du  = gfu.vec.CreateVector()

    for it in range(maxits):
        print ("Newton iteration {:3}".format(it),end=", ")
        # print ("energy = {:16}".format(a.Energy(gfu.vec)),end="")

        # solve linearized problem:
        a.Apply (gfu.vec, res)
        a.AssembleLinearization (gfu.vec)
        inv = a.mat.Inverse(FreeDofs)
        alpha = 5e-2
        du.data = alpha * inv * res

        #update iteration
        gfu.vec.data -= du

        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        print ("<A u",it,", A u",it,">_{-1}^0.5 = ", stopcritval)
        if stopcritval < tol:
            break

        #for sc in scenes:
        #    sc.Redraw()

    return gfu, stopcritval, it


############
### Main ###
############

# Most commonly changed parameters

#L= 90.0
#d = 3.00
# order=3                # polynomial degree of finite elements

#indexes_deltas = [93]  

# Ex. if d=3.00 then the list of deltas where the indexes_deltas 
#   are given the corresponding indexes for the simularion
#   is in the file 'meshes3_00/deltas'.
# For example, the delta with index 93 for d=3.00 is delta=0.913

# It will be assumed that the arguments passed from shell
# are, in that order:
# the length, the thickness, the order for the finite element space, 
# and a list of indexes for delta.
# For example,
#    90 1.62 3 '4, 93'
# means L=90, d=1.62, order=3, and indexes_delta = [4,93]

data = sys.argv

L = float(data[1])
d = float(data[2])
order = int(data[3])

s = data[4]
indexes_deltas = s.split(",")
n = len(indexes_deltas)
for i in range(n):
    indexes_deltas[i] = int(indexes_deltas[i])

print('L={}, d={}, order={}'.format(L, d, order))
print('indexes_deltas={}'.format(indexes_deltas))


# More stable parameters
G = 0.13               # measured in MPa
entropic_unit = 136.6  # k_B*T/V_m measured in MPa
phi0 = 0.2
chi =  0.348           # J. Elast. 2022, Sect. 3.4
rho = 1.23e-6          # measured in kg/mm³
g = 9.8                # measured in m/s²
AA = 1e5

# Start counting time elapsed
start_time = datetime.datetime.now()

# Example: when d=1.62, folder_name_suffix=1_62
#          when d=15,   folder_name_suffix=15_00
folder_name_suffix = str(int(d)) + '_' + str(int(d%1*100)).zfill(2)

delta_values = np.loadtxt('meshes' + folder_name_suffix + '/deltas')
numberofdeltas = len(delta_values)

for index_delta in indexes_deltas:    
    delta = delta_values[index_delta];
    print("Mesh number = {}, delta={:.3f}".format(index_delta,delta))
    
    gel = gel_debonded2D(length=L, thickness=d, delta=delta, shear_modulus = G,\
                         entropic_unit=entropic_unit, phi0=phi0, chi=chi, mass_density=rho)
    
    modelling = Solve_gel_bonded_imported_mesh (gel,\
                folder_name_suffix=folder_name_suffix, order=order)
    modelling.add_mesh(index_delta)
    modelling.Solve_incremental_softening(prefactor_obstacle=AA)
    
    print("Time elapsed =" + str(datetime.datetime.now() - start_time))
