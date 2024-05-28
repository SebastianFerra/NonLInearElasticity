## generates list of problems for the user to solve
"""
problem1 = [gel params as dict, geom of problem as string, 
            BC for problem]
"""

problem1 = [{"chi" :0.45, "phi0" : 0.3, "G": 0.15647831497059816},
            "geo.stp",
            {"x":"front", "y": "bottom" , "z":"left"}
            ,1]