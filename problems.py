## generates list of problems for the user to solve
"""
problem1 = [gel params as dict, geom of problem as string, 
            BC for problem]
"""

problem1 = [{"chi" :0.45, "phi0" : 0.3, "G": 0.15647831497059816,"dim":3},
            "geo.stp",
            {"dir_cond" :"components","x":"front", "y": "bottom" , "z":"left"}
            ,1,
            "3D_FreeSwell"]

problem2 = [{"chi" :0.45, "phi0" : 0.3, "G": 0.13,"dim":2},"geo_2D_bonded.pkl", {"dir_cond": "faces","DIR_FACES":"bottom"},2, "2D_bonded"]


problem3 = [{"chi" :0.348, "phi0" : 0.2, "G": 0.13,"dim":3},"geo_3D_bonded.stp", {"dir_cond": "faces","DIR_FACES":"left"},2, "3D_bonded"]

problem4 = [{"chi" :0.45, "phi0" : 0.3, "G": 0.13,"dim":2},"geo_2D_bonded_delta90.pkl", {"dir_cond": "faces","DIR_FACES":"delta_face"},1, "2D_bond_delta90"]
bonded = [2,3,4]
free_swell = [1]