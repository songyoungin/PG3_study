import imageio

def imgs2gif(inpath, outpath):
    images = []
    for idx in range(250):
        img_name = inpath + "/generated_epoch%03d.png" % idx
        images.append(imageio.imread(img_name))
    imageio.mimsave(outpath + "/result.gif", images, fps=5)
    print("Generate GIF file complete!")

imgs2gif("samples", "samples")