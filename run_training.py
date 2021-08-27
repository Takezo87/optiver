from lit_model import *
from lit_dm import *
from tsai.models.TSTPlus import *

dm=DM()
dm.setup()

trainer = pl.Trainer(gpus=1)
model = TSTPlus(8, 1, 600, y_range=(0,0.08))
lit = BaseLitModel(model)

trainer.fit(lit, datamodule=dm)
