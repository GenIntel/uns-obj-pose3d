### Coordinate Frames

The semantic axes of a camera are
   - x: right (pytorch3d: left)
   - y: bottom (pytorch3d: top)
   - z: front (pytorch3d: front)

The semantic axes of an object are
   - x: left (pytorch3d: left) 
   - y: back (pytorch3d: top)
   - z: top (pytorch3d: front)

This leads to the `cam_tform4x4_obj` for a camera looking straight at the front of an object of:  
[  [1,  0,  0,     0],  
   [0,  0, -1,     0],   
   [0, 1,  0, +dist],  
   [0,  0,  0,     0],
]
This transformation can be understood as `+270` or `-90` degrees rotation around the `x` axis.

Open3d uses as default `cam_tform4x4_obj`:   
[    [1,  0,  0,     0],  
   [0,  -1, 0,     0],   
   [0, 0,  -1, +dist],  
   [0,  0,  0,     0],
]

