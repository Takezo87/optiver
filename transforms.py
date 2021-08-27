from typing import Dict, Tuple

import numpy as np

class Standardize:
    '''
    batch level standardization
    '''

    def __init__(self, means=None, stds=None, axis:Tuple=(0,2)):
        self.means = np.array([
                0.99969482421875, 1.000321388244629, 0.9995064735412598, 1.0005191564559937,
                769.990177708821, 766.7345672818379, 959.3416027831918, 928.2202512713748,
                1.0000068043192514, 1.0000055320253616, 5.129816581143487e-08, 9.831598141593519e-08
            ])
        self.stds = np.array([
                0.0036880988627672195, 0.003687119111418724, 0.0037009266670793295, 0.0036990800872445107,
                5354.051690318169, 4954.947103063445, 6683.816183660414, 5735.299917793827,
                0.003689893218043926, 0.00370745215558702, 6.618708642293018e-07, 1.2508970015188411e-06
            ])   

    def setup(self, df):
        pass
        # self.means = 

    def apply_transform(self, batch:Dict) -> Dict:
        
        batch_tfmd=(batch['x']-self.means)/self.stds
        batch_tfmd.update(**{'y':batch['y']})
        return batch_tfmd
