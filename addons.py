#addons.py
import math

'''
def is_hand_raised(xy):
    wrist = xy[10]   # right wrist
    shoulder = xy[6] # right shoulder

    return wrist[1] < shoulder[1]

def is_pointing(xy):
    wrist = xy[10]
    elbow = xy[8]

    return abs(wrist[0] - elbow[0]) > 50
'''

def get_forearm_angle(xy, side="right"):
    if side == "right":
        elbow_idx, wrist_idx = 8, 10
    else:
        elbow_idx, wrist_idx = 7, 9

    if len(xy) <= max(elbow_idx, wrist_idx):
        return None

    elbow = xy[elbow_idx]
    wrist = xy[wrist_idx]
    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]

    if abs(dx) < 1 and abs(dy) < 1:
        return None

    return math.degrees(math.atan2(dy, dx))


def get_hand_turn_feedback(current_angle, previous_angle, change_threshold=20.0, sideways_band=25.0):
    if current_angle is None:
        return None

    if previous_angle is None:
        if abs(current_angle) <= sideways_band or abs(abs(current_angle) - 180) <= sideways_band:
            return "HAND SIDEWAYS"
        return None

    delta = abs(current_angle - previous_angle)
    if delta > 180:
        delta = 360 - delta

    if delta >= change_threshold:
        if abs(current_angle) <= sideways_band or abs(abs(current_angle) - 180) <= sideways_band:
            return "HAND SIDEWAYS"
        return "HAND TURNING"

    return None