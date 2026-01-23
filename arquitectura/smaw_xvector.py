import sys

sys.path.append("/home/luis/PlotNeuralNet/")
from pycore.tikzeng import *


# Capa FC personalizada
def to_FC(
    name,
    n_filer=256,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
    height=20,
    depth=2,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        zlabel="""
        + str(n_filer)
        + """,
        fill=\FcColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


# Capa Conv1D personalizada
def to_Conv1D(
    name,
    n_filer=256,
    kernel=5,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=25,
    depth=3,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {RightBandedBox={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        xlabel={{"""
        + str(n_filer)
        + """, }},
        zlabel=k="""
        + str(kernel)
        + """,
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


arch = [
    to_head("/home/luis/PlotNeuralNet/"),
    to_cor(),
    to_begin(),
    # VGGish - Extractor de características
    to_Conv(
        name="vggish",
        s_filer="T",
        n_filer=128,
        offset="(0,0,0)",
        to="(0,0,0)",
        width=3,
        height=35,
        depth=35,
        caption="VGGish",
    ),
    # Conv1D Block 1 - 256 canales, kernel 5
    to_Conv1D(
        name="conv1",
        n_filer=256,
        kernel=5,
        offset="(2.5,0,0)",
        to="(vggish-east)",
        width=3,
        height=30,
        depth=30,
        caption="Conv1D",
    ),
    to_connection("vggish", "conv1"),
    # Conv1D Block 2 - 256 canales, kernel 3
    to_Conv1D(
        name="conv2",
        n_filer=256,
        kernel=3,
        offset="(2.5,0,0)",
        to="(conv1-east)",
        width=3,
        height=28,
        depth=28,
        caption="Conv1D",
    ),
    to_connection("conv1", "conv2"),
    # Conv1D Block 3 - 512 canales, kernel 3
    to_Conv1D(
        name="conv3",
        n_filer=512,
        kernel=3,
        offset="(2.5,0,0)",
        to="(conv2-east)",
        width=4,
        height=26,
        depth=26,
        caption="Conv1D",
    ),
    to_connection("conv2", "conv3"),
    # Stats Pooling - concatenación de media y desviación estándar
    to_Pool(
        name="stats",
        offset="(2.5,0,0)",
        to="(conv3-east)",
        width=5,
        height=22,
        depth=3,
        opacity=0.8,
        caption="Stats Pool",
    ),
    to_connection("conv3", "stats"),
    # FC Shared - 256 neuronas
    to_FC(
        name="fc_shared",
        n_filer=256,
        offset="(2.5,0,0)",
        to="(stats-east)",
        width=2,
        height=18,
        depth=3,
        caption="FC",
    ),
    to_connection("stats", "fc_shared"),
    # Head 1 - Placa (3 clases)
    to_SoftMax(
        name="head_placa",
        s_filer=3,
        offset="(3,3.5,0)",
        to="(fc_shared-east)",
        width=1.5,
        height=10,
        depth=10,
        opacity=0.9,
        caption="Placa",
    ),
    # Head 2 - Electrodo (4 clases)
    to_SoftMax(
        name="head_electrodo",
        s_filer=4,
        offset="(3,0,0)",
        to="(fc_shared-east)",
        width=1.5,
        height=12,
        depth=12,
        opacity=0.9,
        caption="Electrodo",
    ),
    # Head 3 - Corriente (2 clases)
    to_SoftMax(
        name="head_corriente",
        s_filer=2,
        offset="(3,-3.5,0)",
        to="(fc_shared-east)",
        width=1.5,
        height=8,
        depth=8,
        opacity=0.9,
        caption="Corriente",
    ),
    # Conexiones a los heads
    r"""\draw [connection]  (fc_shared-east) -- node {\midarrow} (head_placa-west);""",
    r"""\draw [connection]  (fc_shared-east) -- node {\midarrow} (head_electrodo-west);""",
    r"""\draw [connection]  (fc_shared-east) -- node {\midarrow} (head_corriente-west);""",
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")
    print(f"Archivo generado: {namefile}.tex")
    print(
        f"Para compilar: cd {'/'.join(namefile.split('/')[:-1])} && pdflatex {namefile.split('/')[-1]}.tex"
    )


if __name__ == "__main__":
    main()
