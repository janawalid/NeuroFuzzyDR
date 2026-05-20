import numpy as np

class OrientationFuzzy:
    def __init__(self):
        # Membership functions for disc position (0-1)
        self.disc_left = lambda x: np.interp(x, [0,0,0.2,0.4], [1,1,0,0])
        self.disc_center = lambda x: np.interp(x, [0.3,0.5,0.7], [0,1,0])
        self.disc_right = lambda x: np.interp(x, [0.6,0.8,1,1], [0,0,1,1])
        
        # Membership for notch ratio (0-1)
        self.notch_none = lambda x: np.interp(x, [0,0,0.05,0.15], [1,1,0,0])
        self.notch_partial = lambda x: np.interp(x, [0.1,0.2,0.35], [0,1,0])
        self.notch_full = lambda x: np.interp(x, [0.25,0.4,1,1], [0,0,1,1])
        
        # Output membership (inversion confidence 0-1)
        self.out_low = lambda x: np.interp(x, [0,0,0.3,0.5], [1,1,0,0])
        self.out_med = lambda x: np.interp(x, [0.3,0.5,0.7], [0,1,0])
        self.out_high = lambda x: np.interp(x, [0.5,0.7,1,1], [0,0,1,1])
        
        # Rule base (disc, notch) -> output
        self.rules = [
            (('left','none'), 'low'),
            (('left','partial'), 'low'),
            (('center','none'), 'low'),
            (('center','partial'), 'med'),
            (('right','partial'), 'high'),
            (('right','full'), 'high'),
            (('left','full'), 'med'),
            (('center','full'), 'high'),
        ]
    
    def evaluate(self, disc_pos, notch_ratio):
        # Fuzzify inputs
        disc_left = self.disc_left(disc_pos)
        disc_center = self.disc_center(disc_pos)
        disc_right = self.disc_right(disc_pos)
        notch_none = self.notch_none(notch_ratio)
        notch_partial = self.notch_partial(notch_ratio)
        notch_full = self.notch_full(notch_ratio)
        
        # Rule evaluation (Mamdani min)
        output_levels = {'low':0, 'med':0, 'high':0}
        for (disc_state, notch_state), out_state in self.rules:
            disc_val = {'left':disc_left, 'center':disc_center, 'right':disc_right}[disc_state]
            notch_val = {'none':notch_none, 'partial':notch_partial, 'full':notch_full}[notch_state]
            firing = min(disc_val, notch_val)
            output_levels[out_state] = max(output_levels[out_state], firing)
        
        # Defuzzification (centroid)
        x = np.linspace(0, 1, 100)
        y = np.zeros_like(x)
        for i, val in enumerate(x):
            low_act = self.out_low(val) * output_levels['low']
            med_act = self.out_med(val) * output_levels['med']
            high_act = self.out_high(val) * output_levels['high']
            y[i] = max(low_act, med_act, high_act)
        if np.sum(y) == 0:
            return 0
        centroid = np.sum(x*y) / np.sum(y)
        return centroid
    
    def compute(self, disc_pos, notch_ratio):
        conf = self.evaluate(disc_pos, notch_ratio)
        invert = conf > 0.6
        return invert, conf