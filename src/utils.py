import math

def load_track(filepath):
    """
    Loads track logic. Returns (walls, checkpoints).
    Each is a list of tuples: (x1, y1, x2, y2)
    """
    walls = []
    checkpoints = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    mode = 'walls'
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('#'):
            if 'checkpoint' in line.lower():
                mode = 'checkpoints'
            continue
            
        parts = [float(p.strip()) for p in line.split(',')]
        if len(parts) == 4:
            segment = tuple(parts)
            if mode == 'checkpoints':
                checkpoints.append(segment)
            else:
                walls.append(segment)
                
    return walls, checkpoints

def line_intersection(p1, p2, p3, p4):
    """
    Finds the intersection of two line segments (p1->p2 and p3->p4).
    Returns (x, y) if intersection exists, else None.
    p1, p2, p3, p4 are (x, y) tuples.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None
        
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x_int = x1 + t * (x2 - x1)
        y_int = y1 + t * (y2 - y1)
        return (x_int, y_int)
    return None

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
