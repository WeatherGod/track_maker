#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Polygon
from matplotlib.widgets import Lasso
from matplotlib.nxutils import points_inside_poly
from matplotlib import rcParams
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
from os import makedirs
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
        filename = self._filenames[self._currIndex]
        # are we on the right edge of the cache?
        if self._cacheIndex >= len(self._cacher) :
            # is the cache at the maximum size?
            if len(self._cacher) >= self._cachewidth :
                self._cacher.pop(0)
                self._cacheIndex -= 1

            # add an item to the right edge of the cache
            self._cacher.append(LoadRastRadar(filename))


        # are we on the left edge of the cache?
        elif self._cacheIndex <= 0 :
            # is the cache at the maximum size?
            if len(self._cacher) >= self._cachewidth :
                self._cacher.pop()

            # add an item to the left edge of the cache
            self._cacher.insert(0, LoadRastRadar(filename))
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
    ellips = []

    for index in labels :
        p = GetBoundary(reslabels, index)

        if len(p) < 3 :
            ellips.append(None)
            # Not enough points to work with
            continue

        coords = np.array([(ygrid[pnt[0], pnt[1]],
                            xgrid[pnt[0], pnt[1]]) for pnt in p])

        h, k, a, b, t = FitEllipse(coords)
        ellips.append(Ellipse((h, k), a, b, np.rad2deg(t),
                              lw=2, fc='none', ec='r', zorder=2))

    return ellips


class Feature(object) :
    _orig_colors = {'contour': 'k', 'center': 'gray', 'ellip': 'r'}
    def __init__(self, contour=None, center=None, ellip=None) :
        self.objects = {}
        if contour is not None :
            self.objects['contour'] = contour
        if center is not None :
            self.objects['center'] = center
        if ellip is not None :
            self.objects['ellip'] = ellip

    def remove(self) :
        for key, item in self.objects.iteritems() :
            if item is not None :
                item.remove()

        self.objects = {}

    def cleanup(self, hold=()) :
        """
        Remove all objects (except those specified by *hold*.
        """
        allKeys = self.objects.keys()
        for key in allKeys :
            if key not in hold :
                discard = self.objects.pop(key)
                discard.remove()

    def select(self) :
        for key, item in self.objects.iteritems() :
            if item is not None :
                item.set_edgecolor('w')

    def deselect(self) :
        for key, item in self.objects.iteritems() :
            if item is not None :
                item.set_edgecolor(Feature._orig_colors.get(key, 'k'))

    def set_visible(self, visible) :
        for key, item in self.objects.iteritems() :
            if item is not None :
                item.set_visible(visible)

    def set_picker(self, picker) :
        for key, item in self.objects.iteritems() :
            if item is not None :
                item.set_picker(picker)

    def contains(self, event) :
        return any([item.contains(event)[0] for
                    item in self.objects.values()])
 


class RadarDisplay(object) :
    _increms = {'left': -1, 'right': 1}

    def __init__(self, volume, tracks, falarms, radarFiles) :
        """
        Create an interactive display for creating track data
        from radar data.
        """
        frameDiff = len(radarFiles) - len(volume['volume_data'])
        if frameDiff < 0 :
            raise ValueError("Previous session had more frames than available"
                             "input data frames")
        elif frameDiff > 0 :
            newFrames = [{"volTime": np.nan,
                          "frameNum": len(volume['volume_data']) + index,
                          "stormCells": np.array([],
                                                dtype=TrackUtils.corner_dtype)}
                         for index in xrange(frameDiff)]
            volume['volume_data'].extend(newFrames)
            volume['frame_cnt'] = len(radarFiles)



        minLat, minLon, maxLat, maxLon = ConsistentDomain(radarFiles)


        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.radarData = RadarCache(radarFiles, 4)
        
        self._increm_funcs = {'left': self.radarData.prev,
                              'right': self.radarData.next}

        self._im = None
        self._features = [[] for i in xrange(len(self.radarData))]
        for frameIndex in range(len(self.radarData)) :
            cells = volume['volume_data'][frameIndex]['stormCells']
            feats = self._features[frameIndex]
            for cellIndex in range(len(cells)) :
                newPoint = self._new_point(cells['xLocs'][cellIndex],
                                           cells['yLocs'][cellIndex])
                newFeat = Feature(center=newPoint)
                newFeat.set_visible(False)
                feats.append(newFeat)

        self._curr_selection = None

        self._tracks = [Line2D(track['xLocs'], track['yLocs'],
                               color='grey', lw=2, marker='.', picker=True)
                        for track in tracks]
        for track in self._tracks :
            self.ax.add_artist(track)

        self.volume = volume
        self.tracks = tracks
        self.falarms = falarms

        self.frameIndex = 0
        data = self.radarData.next()
        lons, lats = np.meshgrid(data['lons'], data['lats'])
        self.xs, self.ys = LonLat2Cart((minLon + maxLon)/2.0,
                                       (minLat + maxLat)/2.0,
                                        lons, lats)

        self._update_frame()

        self.ax.set_xlim(self.xs.min(), self.xs.max())
        self.ax.set_ylim(self.ys.min(), self.ys.max())
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')


        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        #self.fig.canvas.mpl_connect('button_release_event',
        #                             self.process_click)
        self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        self.fig.canvas.mpl_connect('pick_event', self.onpick)


        # Don't start in any mode
        self._mode = None

        # Need to remove some keys...
        rcParams['keymap.home'].remove('r')
        rcParams['keymap.back'].remove('left')
        rcParams['keymap.forward'].remove('right')
        rcParams['keymap.zoom'] = []
        rcParams['keymap.save'] = []

    def _new_point(self, x, y) :
        newpoint = Circle((x, y), color='gray', zorder=3, picker=None)
        self.ax.add_artist(newpoint)
        return newpoint

    def onpick(self, event) :
        pass

    def onlasso(self, verts) :
        newPoly = Polygon(verts, lw=2, edgecolor='k', facecolor='gray',
                          hatch='/', alpha=0.5, zorder=1, picker=None)
        self._features[self.frameIndex].append(Feature(contour=newPoly))
        self.fig.canvas.draw_idle()
        self.fig.canvas.widgetlock.release(self.curr_lasso)
        del self.curr_lasso
        self.curr_lasso = None
        self.ax.add_artist(newPoly)

    def onpress(self, event) :
        if self._mode == 'o' :
            # Outline mode
            if self.fig.canvas.widgetlock.locked() :
                return
            if event.inaxes is not self.ax :
                return

            self.curr_lasso = Lasso(event.inaxes, (event.xdata, event.ydata),
                                    self.onlasso)

            # Set a lock on drawing the lasso until finished
            self.fig.canvas.widgetlock(self.curr_lasso)

        elif self._mode == 's':
            # Selection mode
            if event.inaxes is not self.ax :
                return

            select = None
            for feat in self._features[self.frameIndex] :
                if feat.contains(event) :
                    print "Feature:", feat
                    select = feat

            if select is not None :
                if self._curr_selection is not None :
                    self._curr_selection.deselect()

                if self._curr_selection is select :
                    self._curr_selection = None
                else :
                    self._curr_selection = select
                    self._curr_selection.select()

            self.fig.canvas.draw_idle()


    def process_key(self, event) :
        if event.key in RadarDisplay._increms :
            if (0 <= (self.frameIndex + RadarDisplay._increms[event.key])
                  <= (len(self.radarData) - 1)) :
                lastFrame = self.frameIndex
                self.frameIndex += RadarDisplay._increms[event.key]

                # Update the radar data
                self._increm_funcs[event.key]()

                if self._curr_selection is not None :
                    self._curr_selection.deselect()
                    self._curr_selection = None

                # Update the frame
                self._update_frame(lastFrame)

                # Update the window
                self.fig.canvas.draw()

        elif event.key == 'r' :
            # Recalculate ellipsoids
            self._update_frame(force_recluster=True)
            self.fig.canvas.draw()

        elif event.key == 'c' :
            # Completely remove the features for this frame
            if self._curr_selection is not None :
                self._curr_selection.deselect()
                self._curr_selection = None

            for feat in self._features[self.frameIndex] :
                feat.remove()
            self._features[self.frameIndex] = []

            self._update_frame()
            self.fig.canvas.draw()

        elif event.key == 'd' :
            # Delete the currently selected artist
            if self._curr_selection is not None :
                self._curr_selection.deselect()
                self._curr_selection.remove()
                self._features[self.frameIndex].remove(self._curr_selection)
                self._curr_selection = None

            self.fig.canvas.draw()

        elif event.key == 's' :
            # set mode to "selection mode"
            self._mode = 's'

            # Just in case the canvas is still locked.
            if self.curr_lasso is not None :
                self.fig.canvas.widgetlock.release(self.curr_lasso)
                del self.curr_lasso
                self.curr_lasso = None
            print "Selection Mode"

        elif event.key == 'o' :
            # set mode to "outline mode"
            self._mode = 'o'
            print "Outline Mode"

    def _clear_frame(self, frame=None) :
        if frame is None :
            frame = self.frameIndex

        # Set the frame's ellipses to invisible
        for feat in self._features[frame] :
            feat.set_visible(False)
            #feat.set_picker(None)

    def get_clusters(self) :
        dataset = self.radarData.curr()
        data = dataset['vals'][0]

        flat_data = data[data >= -20]

        clustLabels = np.empty(data.shape, dtype=int)
        clustLabels[:] = -1

        if np.nanmin(flat_data) == np.nanmax(flat_data) :
            # can't cluster data with no change
            return clustLabels, 0

        bins = np.linspace(np.nanmin(flat_data),
                           np.nanmax(flat_data), 2**8)
        data_digitized = np.digitize(data.flat, bins[::-1])
        data_digitized.shape = data.shape
        data_digitized = data_digitized.astype('uint8')

        markers = np.zeros(data.shape, dtype=int)

        for index, feat in enumerate(self._features[self.frameIndex]) :
            if 'contour' in feat.objects :
                contr = feat.objects['contour']
                res = points_inside_poly(zip(self.xs.flat, self.ys.flat),
                                         contr.get_xy())
                res.shape = self.xs.shape
                markers[res] = index + 1

            # No contour available? Then fall back to just a point
            elif 'center' in feat.objects :
                cent = feat.objects['center']
                gridx, gridy = self._xy2grid(cent.center[0], cent.center[1])
                markers[gridy, gridx] = index + 1

            # TODO: work from an ellipse, if it exists?
            else :
                raise ValueError("Empty feature?")


        # Set anything less than 20 dBZ as background
        markers[np.isnan(data) | (data < 20)] = -1
        ndimg.watershed_ift(data_digitized, markers, output=clustLabels)
        clustCnt = len(self._features[self.frameIndex])

        cents = ndimg.center_of_mass(data, clustLabels,
                                     range(1, clustCnt + 1))
        ellipses = FitEllipses(clustLabels, range(1, clustCnt + 1),
                               self.xs, self.ys)

        for center, ellip, feat in zip(cents, ellipses,
                                       self._features[self.frameIndex]) :
            newPoint = self._new_point(self.xs[center], self.ys[center])
            if ellip is not None :
                self.ax.add_artist(ellip)

            # Remove any other objects that may exist before adding
            # new objects to the feature.
            feat.cleanup(['contour'])
            
            feat.objects['center'] = newPoint
            feat.objects['ellipse'] = ellip

        #print "clust count:", clustCnt
        return clustLabels, clustCnt

    def _xy2grid(self, x, y) :
        return (self.xs[0].searchsorted(x),
                self.ys[:, 0].searchsorted(y))


    def _update_frame(self, lastFrame=None, force_recluster=False) :
        if lastFrame is not None :
            self._clear_frame(lastFrame)

        data = self.radarData.curr()

        # Display current frame's radar image
        if self._im is None :
            self._im = MakeReflectPPI(data['vals'][0], self.ys, self.xs,
                                      meth='im', ax=self.ax, colorbar=False,
                                      axis_labels=False, zorder=0)
        else :
            self._im.set_data(data['vals'][0])

        if force_recluster or any([('center' not in feat.objects) for
                                   feat in self._features[self.frameIndex]]) :
            clustLabels, clustCnt = self.get_clusters()

        # Set contours for this frame to visible
        for feat in self._features[self.frameIndex] :
            feat.set_visible(True)
            #feat.set_picker(True)

        theDateTime = datetime.utcfromtimestamp(data['scan_time'])
        self.ax.set_title(theDateTime.strftime("%Y/%m/%d %H:%M:%S"))


   

def AnalyzeRadar(volume, tracks, falarms, radarFiles) :

    rd = RadarDisplay(volume, tracks, falarms, radarFiles)
    plt.show()


def GetClusters(radarData) :
    cleanmask = ndimg.binary_opening(radarData > 30,
                                     structure=np.ones((5, 5)))
    return ndimg.label(cleanmask)


def main(args) :
    dirName = os.path.join(args.path, args.scenario)

    # Create the directory if it doesn't exist already
    if not os.path.exists(dirName) :
        makedirs(dirName)

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
        sceneParams = ParamUtils.ReadSimulationParams(paramFile)


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
    print paramFile, volumeFile, origTrackFile, filtTrackFile
    TrackFileUtils.SaveCorners(volumeFile, volume['corner_filestem'],
                               volume['volume_data'],
                               path=os.path.dirname(volumeFile))
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

