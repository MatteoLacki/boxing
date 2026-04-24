from opentimspy import OpenTIMS
import matplotlib.pyplot as plt

op = OpenTIMS(dataset_path)
maxtof = op.mz_to_tof(op.max_mz, 1)[0]
tofs = np.arange(maxtof + 1)
mzs = op.tof_to_mz(tofs, 1)
plt.plot(tofs, mzs)
plt.show()
