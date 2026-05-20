import numpy as np

class AdjustmentFuzzy:
    def __init__(self):
        # Confidence membership
        self.conf_low = lambda x: np.interp(x, [0,0,0.3,0.5], [1,1,0,0])
        self.conf_med = lambda x: np.interp(x, [0.3,0.6,0.8], [0,1,0])
        self.conf_high = lambda x: np.interp(x, [0.6,0.8,1,1], [0,0,1,1])
        # Sharpness (0-200)
        self.sharp_blurry = lambda x: np.interp(x, [0,0,50,100], [1,1,0,0])
        self.sharp_ok = lambda x: np.interp(x, [50,100,150], [0,1,0])
        self.sharp_sharp = lambda x: np.interp(x, [100,150,200,200], [0,0,1,1])
        # Output delta (-1 to 1)
        self.delta_neg = lambda x: np.interp(x, [-1,-1,-0.3,0], [1,1,0,0])
        self.delta_zero = lambda x: np.interp(x, [-0.5,0,0.5], [0,1,0])
        self.delta_pos = lambda x: np.interp(x, [0,0.3,1,1], [0,0,1,1])
        
        self.rules = [
            (('low','blurry'), 'neg'),
            (('low','ok'), 'neg'),
            (('med','blurry'), 'neg'),
            (('med','ok'), 'zero'),
            (('high','blurry'), 'neg'),
            (('high','ok'), 'zero'),
            (('high','sharp'), 'pos'),
            (('med','sharp'), 'zero'),
            (('low','sharp'), 'zero'),
        ]
    
    def evaluate(self, confidence, sharpness):
        # fuzzify
        c_low = self.conf_low(confidence)
        c_med = self.conf_med(confidence)
        c_high = self.conf_high(confidence)
        s_blur = self.sharp_blurry(sharpness)
        s_ok = self.sharp_ok(sharpness)
        s_sharp = self.sharp_sharp(sharpness)
        
        out = {'neg':0, 'zero':0, 'pos':0}
        for (c_state, s_state), out_state in self.rules:
            c_val = {'low':c_low, 'med':c_med, 'high':c_high}[c_state]
            s_val = {'blurry':s_blur, 'ok':s_ok, 'sharp':s_sharp}[s_state]
            firing = min(c_val, s_val)
            out[out_state] = max(out[out_state], firing)
        
        # centroid defuzzification over [-1,1]
        x = np.linspace(-1, 1, 100)
        y = np.zeros_like(x)
        for i, val in enumerate(x):
            neg_act = self.delta_neg(val) * out['neg']
            zero_act = self.delta_zero(val) * out['zero']
            pos_act = self.delta_pos(val) * out['pos']
            y[i] = max(neg_act, zero_act, pos_act)
        if np.sum(y) == 0:
            return 0
        centroid = np.sum(x*y) / np.sum(y)
        return centroid
    
    def compute(self, confidence, sharpness):
        # normalize sharpness to 0-200 (Laplacian variance typical range)
        sharp_norm = min(200, sharpness)
        delta = self.evaluate(confidence, sharp_norm)
        return delta