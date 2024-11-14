from pycirclize import Circos
import pandas as pd

# Create RPG jobs parameter dataframe (4 jobs, 8 parameters)
df = pd.DataFrame(
    data=[
        #R@10, 
        [62.88, 42.60, 46.35, 10.27, 99.73, 91.82], #FaceCPT
        [61.75, 40.43, 0,     0,     0,     0], #BLIP
        [62.02, 36.90, 0,     0,     0,     0], #BLIP2
        [59.56, 42.17, 0,     0,     0,     0], #mPLUG
        [0,     0,     0,     0,     99.40, 91.39], #Faceptor
        [0,     0,     0,     0,     99.60, 91.88], #FRL
        [0,     0,     0,     0,     0,     91.79], #FaceXFormer
    ],
    index=["FaceCPT", "BLIP", "BLIP2", "mPLUG", "Faceptor", "FRL", "FaceXFormer"],
    columns=["T2I Retrieval-FT", "Captioning-FT", "T2I Retrieval-ZS", "Captioning-ZS", "Face Recognition", "Facial Attribute Prediction"],
)
print(df)

# Initialize Circos instance for radar chart plot
circos = Circos.radar_chart(
    df,
    vmax=100,
    fill=True,
    circular=True,
    marker_size=6,
    bg_color=None,
    cmap=dict(FaceCPT="darkslategrey", BLIP="olive", BLIP2="orange", mPLUG="lightslategray", Faceptor="coral", FRL="firebrick", FaceXFormer="brown"), #cyan, orange, , , 
    grid_interval_ratio=0.20,
    label_kws_handler=lambda _: dict(style="italic"),
    line_kws_handler=lambda _: dict(lw=2, ls="solid"),
    marker_kws_handler=lambda _: dict(marker="s", ec="grey", lw=0.5),
)
circos.text("Perormance Comparison with SOTA Methods", r=125, size=15, weight="bold")

# Plot figure & set legend on upper right
fig = circos.plotfig()
_ = circos.ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.05, 1.05),
    fontsize=10,
    title="Face Tasks Related to Surveillance",
)
fig.savefig("example01.svg")