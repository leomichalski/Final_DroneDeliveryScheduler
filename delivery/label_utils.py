HIGH_SPEED = 0
LOW_SPEED = 1
LOW_RISK = 2
NO_FLY = 3


def can_uav_fly(label):
    return label != NO_FLY
