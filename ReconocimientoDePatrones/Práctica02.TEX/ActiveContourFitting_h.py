import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters import gaussian
from skimage.segmentation import active_contour


class ActiveContourFitting:

    def __init__(
            self,
            videoDir=None, init=None, imgextension=None,
            alpha=None, beta=None, gamma=None,
            gSigma=None, w_line=None, w_edge=None,
            snakeIterations=10,
            firstFrame=0, lastFrame=-1
    ):
        self.videoDir="./Amoeba-moving-Timelapse_frames/"
        self.imgextension=".ppm"

        self.w_line=-5,
        self.w_edge=0,
        self.alpha  =0.015 if alpha  is None else alpha
        self.beta   =10    if beta   is None else beta
        self.gamma  =0.001 if gamma  is None else gamma
        self.g_sigma=3     if gSigma is None else gSigma

        self.snakeIterations=snakeIterations
        self.firstFrame=firstFrame
        self.lastFrame=lastFrame
        self.iniv = None

        self.s = np.linspace(0, 2 * np.pi, 400)
        self.r = 100 + 100 * np.sin(self.s)
        self.c = 220 + 100 * np.cos(self.s)
        self.init = np.array([self.r, self.c]).T

        self.video=[]
        self.acvideo=[]
        self.vdframes=[]

    def setInit(self, x, y, height, width, n_sides):
        self.s = np.linspace(0, 2 * np.pi, n_sides)
        self.r = y + height * np.sin(self.s)
        self.c = x + width * np.cos(self.s)
        self.init = np.array([self.r, self.c]).T

    def showInitFrame(self,i=None):
        j = i if i is not None else 0
        cnt =                      \
            self.init if i is None \
            else self.acvideo[j]

        fig, ax = plt.subplots(figsize=(7, 7))
        img=skimage.io.imread(self.vdframes[j], as_gray=True)

        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(cnt[:, 1], cnt[:, 0], '-r', lw=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()


    def getVideo(self, videoDir=None, imgextension=None, memory=True):
        if videoDir is not None:
            self.videoDir=videoDir
        if imgextension is not None:
            self.imgextension=imgextension
        self.vdframes=[]
        self.video=[]
        # Get all filenames
        self.vdframes=[
            self.videoDir+frame
            for frame in os.listdir(self.videoDir)
            if frame.endswith(self.imgextension)
        ]
        self.lastFrame= self.lastFrame if self.lastFrame>-1 else len(os.listdir(videoDir))
        self.lastFrame - self.firstFrame
        # Sort by name
        self.vdframes.sort()

        if memory:
            for frame in self.vdframes:
                self.video.append(skimage.io.imread(frame, as_gray=True))

    def moreContrast(self):
        frame_index=0
        for frame in self.video:
            ## Add contrast
            frame=skimage.exposure.equalize_hist(frame)
            self.video[frame_index]=frame
            frame_index+=1

    def thresholdingFrames(self, tres_factor=1.8):
        frame_index=0
        for frame in self.video:
            ## Add contrast
            tres=skimage.filters.threshold_otsu(frame)
            tres *= tres_factor
            frame = frame > tres
            self.video[frame_index]=frame
            frame_index+=1


    def invertLevels(self):
        frame_index=0
        for frame in self.video:
            ## Add contrast
            frame=skimage.util.invert(frame)
            self.video[frame_index]=frame
            frame_index+=1

    def crop(self, x, y, height, width):
        frame_index=0
        for frame in self.video:
            ## Add contrast
            frame=frame[y:y+height, x:x+width]
            self.video[frame_index]=frame
            frame_index+=1

    def cropCoef(self, x=0, y=0, xCoef=2, yCoef=2):
        frame_index=0
        for frame in self.video:
            frame=frame[y:frame.shape[0]-frame.shape[0]//yCoef,
                        x:frame.shape[1]-frame.shape[1]//xCoef]
            self.video[frame_index]=frame
            frame_index+=1

    def getVideoFrames(self):
        for frame in self.vdframes:
            self.video.append(skimage.io.imread(frame, as_gray=True))

    # Play Video
    def playVideo(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        for frame in self.video:
            ax.clear()
            ax.imshow(frame, cmap=plt.cm.gray)
            plt.pause(0.04)


    def activeContourFitting(self, init=None):
        print("[INFO] Starting Active Contour Fitting...")
        if init is not None:
            self.init=init
        gobber = [ "-", "\\", "|", "/" ]
        gb_idx = 0
        self.acvideo=[]
        snake=self.init
        video = self.video[self.firstFrame:self.iniv]
        bin_num = 25
        bin_val = len(video) // bin_num
        bin_idx = 0
        for img in video:
            for i in range(self.snakeIterations):
                snake = active_contour(
                    gaussian(img, sigma=self.g_sigma, preserve_range=False),
#                    img,
                    snake,
                    alpha=self.alpha,
                    beta =self.beta,
                    gamma=self.gamma,
                    w_line=self.w_line,
                    w_edge=self.w_edge
                )
                print(gobber[gb_idx], end="\b")
                gb_idx = gb_idx + 1 if gb_idx < 3 else 0
            if bin_idx%bin_val==0:
                self.acvideo.append(snake)
            bin_idx+=1
            self.acvideo.append(snake)

    def showActiveContourFitting(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        video = self.video[self.firstFrame:self.iniv]
        for img, snake in zip(video, self.acvideo):
            ax.clear()
            ax.imshow(img, cmap=plt.cm.gray)
            ax.plot(snake[:, 1], snake[:, 0], '-b', lw=1)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis([0, img.shape[1], img.shape[0], 0])
            plt.pause(0.04)

    def saveActiveContourFitting(self, name="sequence"):
        # Remove trailing slash
        acvideoDir=self.videoDir.rsplit("/", 1)[0]+"_ActiveContourFitting_"+name+"/"
        if not os.path.exists(acvideoDir):
            os.makedirs(acvideoDir)
        fig, ax = plt.subplots(figsize=(7, 7))
        video = self.video[self.firstFrame:self.iniv]
        vdframes=self.vdframes[self.firstFrame:self.iniv]
        for img, frame, snake in zip(video, vdframes, self.acvideo):
            ax.clear()
            frame=frame.rsplit("/", 1)[1]
            ax.imshow(img, cmap=plt.cm.gray)
            ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis([0, img.shape[1], img.shape[0], 0])
            # Change extension to png
            plt.savefig(acvideoDir+frame.rsplit(".", 1)[0]+".png")
        plt.close()

    def saveVideo(self, name="video"):
        acvideoDir="/tmp/"+name+"_/"
        if not os.path.exists(acvideoDir):
            os.makedirs(acvideoDir)
        fig, ax = plt.subplots(figsize=(7, 7))
        idx=0
        vdframes=self.vdframes[self.firstFrame:self.iniv]
        video = self.video[self.firstFrame:self.iniv]
        for img, frame, snake in zip(video, vdframes, self.acvideo):
            frame=frame.rsplit("/", 1)[1]
            ax.clear()
            ax.imshow(img, cmap=plt.cm.gray)
            ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis([0, img.shape[1], img.shape[0], 0])
            # Change extension to png
            plt.savefig(f"{acvideoDir}frame{idx:04d}.png")
            idx+=1
        plt.close()
        os.system("ffmpeg -framerate 24 -i "+acvideoDir+"frame%04d.png -c:v libx264 -r 24 "+name+".mp4")

    def activeContour(self, videoDir=None, init=None, imgextension=None, save=False, show=False):
        self.getVideo(memory=False, videoDir=videoDir, imgextension=imgextension)
        if self.init is not None:
            self.init=init
        # Get snake
        ### if save
        if save and not show:
            fig, ax = plt.subplots(figsize=(7, 7))
            init=self.init
            acvideoDir=self.videoDir.rsplit("/", 1)[0]+"_ActiveContourFitting/"
            for frame in self.vdframes:
                ax.clear()
                img = skimage.io.imread(frame, as_gray=True)
                snake = active_contour(
                    gaussian(img, sigma=3, preserve_range=False),
                    init,
                    alpha=0.015,
                    beta=10,
                    gamma=0.001,
                )
                ax.imshow(img, cmap=plt.cm.gray)
                ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
                ax.set_xticks([]), ax.set_yticks([])
                ax.axis([0, img.shape[1], img.shape[0], 0])
                # Change extension to png
                plt.savefig(acvideoDir+self.vdframes[self.video.index(img)].rsplit(".", 1)[0]+".png")
                init=snake

        ### if show
        if show and not save:
            fig, ax = plt.subplots(figsize=(7, 7))
            init=self.init
            for frame in self.vdframes:
                img = skimage.io.imread(frame, as_gray=True)
                snake = active_contour(
                    gaussian(img, sigma=3, preserve_range=False),
                    init,
                    alpha=0.015,
                    beta=10,
                    gamma=0.001,
                )
                ax.imshow(img, cmap=plt.cm.gray)
                ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
                ax.set_xticks([]), ax.set_yticks([])
                ax.axis([0, img.shape[1], img.shape[0], 0])
                plt.pause(0.04)
                init=snake

        ### if save and show
        if save and show:
            fig, ax = plt.subplots(figsize=(7, 7))
            init=self.init
            for frame in self.vdframes:
                img = skimage.io.imread(frame, as_gray=True)
                snake = active_contour(
                    gaussian(img, sigma=3, preserve_range=False),
                    init,
                    alpha=0.015,
                    beta=10,
                    gamma=0.001,
                )
                ax.imshow(img, cmap=plt.cm.gray)
                ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
                ax.set_xticks([]), ax.set_yticks([])
                ax.axis([0, img.shape[1], img.shape[0], 0])
                # Change extension to png
                plt.savefig(acvideoDir+frame.rsplit(".", 1)[0]+".png")
                plt.pause(0.04)
                init=snake
