#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Ellipse
#from mpl_toolkits.basemap import Basemap

import ZigZag.TrackFileUtils as TrackFileUtils
import ZigZag.TrackUtils as TrackUtils
import ZigZag.ParamUtils as ParamUtils

from BRadar.io import LoadRastRadar
from BRadar.plotutils import MakeReflectPPI
from BRadar.maputils import LonLat2Cart

import numpy as np
import scipy.ndimage as ndimg
import os.path
from datetime import datetime

class RadarCache(object) :
    def __init__(self, files, cachewidth=3) :
        """
        Initialize a rolling caching reversible iterator object.

        Each iteration returns the radar data loaded from the files.
        If it is within the cache, then the data is loaded from memory.
        If it is not within the cache, then load the data from the file.

        *files*         list of strings
            List of filenames containing radar data.

        *cachewidth*    integer
            Width of the rolling cache. Must be greater than one.

        """
        if cachewidth < 2 :
            raise ValueError("cachewidth must be greater than 1")

        self._filenames = files
        self._cachewidth = cachewidth
        self._startIndex = 0
        self._endIndex = 0
        self._currIndex = -1

        self._cacheIndex = -1
        self._cacher = []

    def __iter__(self) :
        return self

    def curr(self, lookahead=0) :
        return self._cacher[self._cacheIndex + lookahead]

    def _check_cache_state(self) :
        """
        advance or step back the cache if needed, and adjust the index
        """
        # are we on the right edge of the cache?
        if self._cacheIndex >= len(self._cacher) :
            # is the cache at the maximum size?
            if len(self._cacher) >= self._cachewidth :
                self._cacher.pop(0)
                self._cacheIndex -= 1

            # add an item to the right edge of the cache
            self._cacher.append(LoadRastRadar(self._filenames[self._currIndex]))


        # are we on the left edge of the cache?
        elif self._cacheIndex <= 0 :
            # is the cache at the maximum size?
            if len(self._cacher) >= self._cachewidth :
                self._cacher.pop()

            # add an item to the left edge of the cache
            self._cacher.insert(0, LoadRastRadar(
                                    self._filenames[self._currIndex]))
            self._cacheIndex += 1


    def next(self) :
        if self._currIndex < (len(self._filenames) - 1) :
            self._currIndex += 1
        else :
            raise StopIteration

        self._cacheIndex += 1
        self._check_cache_state()
        
        return self.curr()

    def peek_next(self) :
        """
        Advance cache only if you absolutely have to.

        If there is nothing next, then return None.
        """
        if self._currIndex >= (len(self._filenames) - 1) :
            return None

        self._currIndex += 1
        self._check_cache_state()
        self._currIndex -= 1

        return self.curr(1)

    def prev(self) :
        if self._currIndex > 0 :
            self._currIndex -= 1
        else :
            raise StopIteration

        self._cacheIndex -= 1
        self._check_cache_state()

        return self.curr()

    def peek_prev(self) :
        """
        Step back the cache only if you absolutely have to.

        If there is nothing previous, then return None.
        """
        if self._currIndex <= 0 :
            return None

        self._currIndex -= 1
        self._check_cache_state()
        self._currIndex += 1

        return self.curr(-1)

    def __len__(self) :
        return len(self._filenames)

def ConsistentDomain(radarFiles) :
    minLat = None
    minLon = None
    maxLat = None
    maxLon = None

    for fname in radarFiles :
        data = LoadRastRadar(fname)
        if minLat is not None :
            minLat = min(data['lats'].min(), minLat)
            minLon = min(data['lons'].min(), minLon)
            maxLat = max(data['lats'].max(), maxLat)
            maxLon = max(data['lons'].max(), maxLon)
        else :
            minLat = data['lats'].min()
            minLon = data['lons'].min()
            maxLat = data['lats'].max()
            maxLon = data['lons'].max()

    return (minLat, minLon, maxLat, maxLon)

def GetBoundary(reslabels, index) :
    curmask = (reslabels == index)
    edgepoints = ndimg.binary_dilation(curmask) - curmask
    return np.argwhere(edgepoints)

def FitEllipse(p) :
    # Center
    k, h = np.mean(p, axis=0)

    covvar = np.cov(p, rowvar=0)
    var_x = covvar[1, 1]
    var_y = covvar[0, 0]
    var_xy = covvar[0, 1]
    tmp = np.sqrt(np.abs((var_x - var_y)**2 + (4 * var_xy**2)))

    eigen_x = (var_x + var_y + tmp) / 2.0
    eigen_y = (var_x + var_y - tmp) / 2.0

    # Axis lengths (half-lengths)
    a = 2 * np.sqrt(eigen_x)
    b = 2 * np.sqrt(eigen_y)

    v1 = var_xy / np.sqrt((eigen_x - var_x)**2 + var_xy**2)
    v2 = (eigen_x - var_x) / np.sqrt((eigen_x - var_x)**2 + var_xy**2)

    # Ellipse rotation (radians)
    t = np.arctan2(v2, v1)

    # Return full-length axis lengths.
    return h, k, 2*a, 2*b, t

def FitEllipses(reslabels, labels, xgrid, ygrid) :
    widths = []
    heights = []
    angles = []
    offsets = []

    for index in labels :
        p = GetBoundary(reslabels, index)

        if len(p) < 10 :
            # Not enough points to work with
            continue

        coords = np.array([(ygrid[pnt[0], pnt[1]],
                            xgrid[pnt[0], pnt[1]]) for pnt in p])

        h, k, a, b, t = FitEllipse(coords)
        offsets.append((h, k))                  # Center
        widths.append(a)                        # Width
        heights.append(b)                       # Height
        angles.append(np.rad2deg(t))            # Rotation

    return [Ellipse(xy, w, h, a, fc='none', lw=5) for xy, w, h, a in
                    zip(offsets, widths, heights, angles)]
    

def AnalyzeRadar(volume, tracks, falarms, radarFiles) :
    minLat, minLon, maxLat, maxLon = ConsistentDomain(radarFiles)

    fig = plt.figure()
    ax = fig.gca()

    radarData = RadarCache(radarFiles, 4)

    def process_press(event) :
        increms = {'left': -1, 'right': 1}
        increm_funcs = {'left': radarData.prev, 'right': radarData.next}
        if event.key in increms :
            if (0 <= (process_press.frameIndex + increms[event.key])
                  <= (len(radarData) - 1)) :
                process_press.frameIndex += increms[event.key]
                data = increm_funcs[event.key]()

                clustLabels, clustCnt = GetClusters(data['vals'][0])
                for ellip in process_press.curr_ellipses :
                    ellip.remove()
                process_press.curr_ellipses = FitEllipses(clustLabels,
                                                          range(1, clustCnt),
                                                          xs, ys)
                for ellip in process_press.curr_ellipses :
                    ellip.set_edgecolor('r')
                    ax.add_artist(ellip)
                im.set_data(data['vals'][0])

                prev_data = radarData.peek_prev()
                if process_press.prev_ellipses is not None :
                    for ellip in process_press.prev_ellipses :
                        ellip.remove()

                if prev_data is not None :
                    prevClusts, prevCnts = GetClusters(prev_data['vals'][0])
                    ghost_im.set_data(np.ma.masked_array(prev_data['vals'][0],
                                                        mask=(prevClusts == 0)))
                    process_press.prev_ellipses = FitEllipses(prevClusts,
                                                             range(1, prevCnts),
                                                              xs, ys)
                    for ellip in process_press.prev_ellipses :
                        ellip.set_edgecolor('r')
                        ellip.set_facecolor('k')
                        ellip.set_alpha(0.5)
                        ax.add_artist(ellip)
                else :
                    process_press.prev_ellipses = None
                    ghost_im.set_data(np.ma.masked_array(data['vals'][0],
                                                         mask=True))
                theDateTime = datetime.utcfromtimestamp(data['scan_time'])
                ax.set_title(theDateTime.strftime("%Y/%m/%d %H:%M:%S"))
                fig.canvas.draw()

    process_press.frameIndex = 0

    data = radarData.next()

    lons, lats = np.meshgrid(data['lons'], data['lats'])

    xs, ys = LonLat2Cart((minLon + maxLon)/2.0, (minLat + maxLat)/2.0,
                         lons, lats)

    clustLabels, clustCnt = GetClusters(data['vals'][0])
    process_press.curr_ellipses = FitEllipses(clustLabels, range(1, clustCnt),
                                              xs, ys)
    im = MakeReflectPPI(data['vals'][0], ys, xs,
                        ax=ax, colorbar=False, axis_labels=False)
    for ellip in process_press.curr_ellipses :
        ellip.set_edgecolor('r')
        ax.add_collection(ellip)
    ghost_im = MakeReflectPPI(np.ma.masked_array(data['vals'][0],
                                                 mask=True),
                              ys, xs,
                              ax=ax, colorbar=False, alpha=0.5,
                              axis_labels=False)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    process_press.prev_ellipses = None
    theDateTime = datetime.utcfromtimestamp(data['scan_time'])

    ax.set_title(theDateTime.strftime("%Y/%m/%d %H:%M:%S"))

    fig.canvas.mpl_connect('key_press_event', process_press)

    plt.show()


def GetClusters(radarData) :
    cleanmask = ndimg.binary_opening(radarData > 30,
                                     structure=np.ones((5, 5)))
    return ndimg.label(cleanmask)


def main(args) :
    dirName = os.path.join(args.path, args.scenario)
    paramFile = os.path.join(dirName, "simParams.conf")
    newSession = (not os.path.exists(paramFile))

    if newSession :
        sceneParams = dict()
        sceneParams.update(ParamUtils.simDefaults)
        sceneParams.update(ParamUtils.trackerDefaults)

        sceneParams.pop('seed')
        sceneParams.pop('totalTracks')
        sceneParams.pop('endTrackProb')
        sceneParams.pop('simConfFile')

        sceneParams['simName'] = args.scenario
    else :
        sceneParams = ParamsUtils.ReadSimulationParams(paramFile)


    origTrackFile = os.path.join(dirName, sceneParams['simTrackFile'])
    filtTrackFile = os.path.join(dirName, sceneParams['noisyTrackFile'])
    volumeFile = os.path.join(dirName, sceneParams['inputDataFile'])
    volumeLoc = os.path.dirname(volumeFile)

    if newSession :
        tracks, falarms = [], []
        volume = dict(frameCnt=0,
                      corner_filestem=sceneParams['corner_file'],
                      volume_data=[])
    else :
        tracks, falarms = TrackFileUtils.ReadTracks(origTrackFile)
        # dictionary with "volume_data", "frameCnt", "corner_filestem"
        #    and "volume_data" contains a list of dictionaries with
        #    "volTime", "frameNum", "stormCells" where "stormCells" is
        #    a numpy array of dtype corner_dtype
        volume = TrackFileUtils.ReadCorners(volumeFile, volumeLoc)


    #reflectData = [LoadRastRadar(radarFile) for radarFile in args.input_files]

    AnalyzeRadar(volume, tracks, falarms, args.input_files)


    # Final save
    SaveState(paramFile, sceneParams, volumeFile, volume,
              origTrackFile, filtTrackFile, tracks, falarms)


def SaveState(paramFile, params, volumeFile, volume,
              origTrackFile, filtTrackFile, tracks, falarms) :
    # Do I need to update the Params?
    TrackFileUtils.SaveCorners(volumeFile, volume['corner_filestem'],
                               volume['volume_data'])
    ParamUtils.SaveConfigFile(paramFile, params)
    TrackFileUtils.SaveTracks(origTrackFile, tracks, falarms)
    TrackFileUtils.SaveTracks(filtTrackFile, tracks, falarms)


if __name__ == '__main__' :
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the radar images to help build tracks")

    parser.add_argument("scenario", type=str,
                        help="Scenario name to save results to. Created if"
                             "it doesn't exist. If it does exist, then"
                             "resume the analysis session.",
                        metavar="SCENARIO", default=None)
    parser.add_argument("-i", "--input", dest="input_files", type=str,
                        help="Input radar files to display",
                        nargs='+', metavar="INPUT", default=None)
    parser.add_argument("-d", "--dir", dest="path", type=str,
                        help="Base directory for SCENARIO (not for INPUT).",
                        metavar="PATH", default='.')

    args = parser.parse_args()


    if args.scenario is None :
        parser.error("Missing SCENARIO!")

    if args.input_files is None or len(args.input_files) == 0 :
        parser.error("Missing input radar files!")

    args.input_files.sort()
    main(args)

