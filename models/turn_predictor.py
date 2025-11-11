# Placeholder for sharp-turn predictor
def compute_turn_probability(ground_conf, curvature, imu=None):
    if ground_conf < 0.5:
        return 0.8
    return 0.2
