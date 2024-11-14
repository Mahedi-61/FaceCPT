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
            ax.set_rgrids(range(1, 6), angle=angle, labels=label, fontsize=13)
            ax.spines['polar'].set_visible(False)
            ax.set_ylim(0, 5)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)


if __name__ == '__main__':
    fig = plt.figure(figsize=(8, 8))
    tit = ["T2I Retrieval-FT (MM)",  "T2I Retrieval-FT (F2T)", "T2I Retrieval-FT (CD)", "Captioning-FT (MM)", "Captioning-FT (CT)", "Captioning-FT (F2T)",
           "T2I Retrieval-ZS", "Captioning-ZS", "Face Recognition", "Facial Attribute Prediction"]  # 10x

    lab = [
        ['53', '56', '59', '62', '65'], #T2I Retrieval-FT (MM)
        ['38', '44', '50', '56', '62'], #T2I Retrieval-FT (F2T)
        ['42', '44', '46', '48', '50'], #T2I Retrieval-FT (CD)
        ['33', '36', '39', '42', '45'], #Captioning-FT (MM)
        ['22', '25', '28', '31', '34'], #Captioning-FT (CT)
        ['10', '11', '12', '13', '14'], #Captioning-FT (F2T)
        ['36', '39', '42', '45', '48'],
        ['4',  '6',  '8',  '10', '12'],
        ['99.20',  '99.40',  '99.60',  '99.80', '100'],
        ['91.20',  '91.40',  '91.60',  '91.80', '92']
    ]

    radar = Radar(fig, tit, lab)
    radar.plot([4.29,  4.69,  4.47,  4.2,   4.87,  4.76,    4.45,  4.14,  3.65,  4.10],    '-', lw=2, color='b', alpha=0.4, label='FaceCPT')
    radar.plot([3.92,  4.56,  3.78,  3.48,  2.18,  3.04,    0,     0,     0,     0],       '-', lw=2, color='r', alpha=0.4, label='BLIP')
    radar.plot([4.01,  4.6,   3.89,  3.3,   2.34,  2.92,    0,     0,     0,     0],       '-', lw=2, color='g', alpha=0.4, label='BLIP2')
    radar.plot([3.19,  2.78,  3.25,  3.13,  4.06,  3.56,    0,     0,     0,     0],       '-', lw=2, color='orange', alpha=0.4, label='mPLUG')
    radar.plot([0,     0,     0,    0,     0,         0,    0,     0,     2,     1.95],    '-', lw=2, color='brown', alpha=0.4, label='Faceptor')
    radar.plot([0,     0,     0,    0,     0,         0,    0,     0,     3,     4.4],     '-', lw=2, color='olive', alpha=0.4, label='FRL')
    radar.plot([0,     0,     0,    0,     0,         0,    0,     0,     0,     3.95],    '-', lw=2, color='coral', alpha=0.4, label='FaceXFormer')

    radar.ax.legend()
    #plt.show()
    fig.savefig('result.svg')