## generates list of problems for the user to solve
"""
problem1 = [gel params as dict, geom of problem as string, 
            BC for problem]
"""

problem1 = [{"chi" :0.348, "phi0" : 0.2, "G": 0.13},
            "box.stp",
            {"x":"front", "y": "bottom" , "z":"left"}
            ,1]