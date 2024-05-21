## generates list of problems for the user to solve
"""
problem1 = [gel params as dict, geom of problem as string, 
            BC for problem]
"""

problem1 = [{"chi" :1, "phi0" : 0.5, "G": 0.13},
            "box.step",
            {"x":"front", "y": "bottom" , "z":"left"}
            ]