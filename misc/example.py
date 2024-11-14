import numpy as np
from matplotlib import pyplot as plt 

class Radar(object):
    def __init__(self, figure, title, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(title)
        self.angles = np.arange(0, 360, 360.0/self.n)
        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=title, fontsize=14)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(range(1, 6), angle=angle, labels=label)
            ax.spines['polar'].set_visible(False)
            ax.set_ylim(0, 5)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)


if __name__ == '__main__':
    fig = plt.figure(figsize=(8, 8))

    tit = ["T2I Retrieval-FT",  "Captioning-FT", "T2I Retrieval-ZS", "Captioning-ZS", "Face Recognition", "Facial Attribute Prediction"]  # 12x

    lab = [
        ['53', '56', '59', '62', '65'],
        ['33', '36', '39', '42', '45'],
        ['36', '39', '42', '45', '48'],
        ['4',  '6',  '8',  '10', '12'],
        ['99.20',  '99.40',  '99.60',  '99.80', '100'],
        ['91.20',  '91.40',  '91.60',  '91.80', '92']
    ]

    radar = Radar(fig, tit, lab)
    #radar.plot([62.88, 42.60, 46.35, 10.27, 99.73, 91.82],  '-', lw=2, color='b', alpha=0.4, label='FaceCPT')
    radar.plot([61.75, 40.43, 0,     0,     0,     0],      '-', lw=2, color='r', alpha=0.4, label='BLIP')
    """
    radar.plot([62.02, 36.90, 0,     0,     0,     0],      '-', lw=2, color='g', alpha=0.4, label='BLIP2')
    radar.plot([59.56, 42.17, 0,     0,     0,     0],      '-', lw=2, color='g', alpha=0.4, label='mPLUG')
    radar.plot([0,     0,     0,     0,     99.40, 91.39],  '-', lw=2, color='g', alpha=0.4, label='Faceptor')
    radar.plot([0,     0,     0,     0,     99.60, 91.88],  '-', lw=2, color='g', alpha=0.4, label='FRL')
    radar.plot([0,     0,     0,     0,     0,     91.79],  '-', lw=2, color='g', alpha=0.4, label='FaceXFormer')
    """
    radar.ax.legend()
    plt.show()
    #fig.savefig('my_result.svg')