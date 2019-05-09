import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

def reconst_orderdict(orderdict):
    """
    Parameters
    -----------
    orderdict : Orderdict
    """
    keys = []
    val_1 = []
    val_2 = []
    val_3 = []
    val_4 = []
    val_5 = []
    val_6 = []
    val_7 = []

    for key, vals in orderdict.items():
        keys.append(key)
        val_1.append(vals[0])
        val_2.append(vals[1])
        val_3.append(vals[2])
        val_4.append(vals[3])
        val_5.append(vals[4])
        val_6.append(vals[5])
        val_7.append(vals[6])

    return keys, val_1, val_2, val_3, val_4, val_5, val_6, val_7 

class AnimDrawer():

    def __init__(self, drawing_obj):
        """
        Parameters
        -----------
        drawing_obj : Orderdict
        """
        # drawing object
        self.keys, self.train_X, self.train_Y, self.test_X, self.ave_ys, self.var_ys, self.kernel_x, self.kernel_y = reconst_orderdict(drawing_obj)
        # animation
        self.anim_fig = plt.figure(dpi=150, figsize=(10., 3.5))
        self.axis_1 = self.anim_fig.add_subplot(121)
        self.axis_2 = self.anim_fig.add_subplot(122)

    def draw_anim(self, interval=50):
        # set axis
        self._set_axis()
        # set img
        self._set_img()

        animation = ani.FuncAnimation(self.anim_fig, self._update_anim, interval=interval, frames=len(self.keys)-1)

        # self.axis.legend()
        print('save_animation?')
        shuold_save_animation = int(input())

        if shuold_save_animation: 
            animation.save('gp_param.mp4', writer='ffmpeg')
            # animation.save("Sample.gif", writer = 'imagemagick') # gif保存

        plt.show()

    def _set_axis(self):
        self.axis_1.set_xlabel("x")
        self.axis_1.set_ylabel("y")
        # self.axis_1.set_aspect('equal', adjustable='box')

        # self.axis_1.set_xlim()
        # self.axis_1.set_ylim()

        self.axis_2.set_xlabel("x")
        self.axis_2.set_ylabel("y")
        # self.axis_2.set_aspect('equal', adjustable='box')

        self.axis_2.set_xlim(-3., 3.)
        self.axis_2.set_ylim(0., 2.)


    def _set_img(self):
        # train data
        self.data_img, = self.axis_1.plot([], [], "o", color="r")

        # average
        self.ave_img, = self.axis_1.plot([], [], color="g")

        # variance
        # self.var_img, = self.axis_1.fill_between([], [], [], color="b", alpha=0.15)

        # kernel
        self.kernel_img, = self.axis_2.plot([], [], color="b")

        # param
        self.txt_img = self.axis_2.text(0.05, 0.8, '', transform=self.axis_2.transAxes)

    def _update_anim(self, i):
        """
        """
        self.data_img.set_data([self.train_X[i], self.train_Y[i]])
        self.ave_img.set_data([self.test_X[i], self.ave_ys[i]])
        self.kernel_img.set_data([self.kernel_x[i], self.kernel_y[i]])
        self.txt_img.set_text(self.keys[i] + "\n y = TAU * exp (x / (2 * SIGMA * SIGMA)) + ETA")
        
        self.axis_1.collections.clear()
        self.axis_1.fill_between(self.test_X[i], self.ave_ys[i] - 2. * np.sqrt(self.var_ys[i]), self.ave_ys[i] + 2. * np.sqrt(self.var_ys[i]), color="b", alpha=0.15)