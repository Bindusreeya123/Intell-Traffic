from collections import defaultdict,deque
LANE_Y = [300, 500]  # two lanes
lane_count = defaultdict(set)

def count_lanes(cx, cy, tid):
    for i, y in enumerate(LANE_Y):
        if abs(cy - y) < 5:
            lane_count[i].add(tid)

