#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Polygon
from matplotlib.lines import Line2D
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
from textwrap import dedent
from bisect import bisect_left
from collections import defaultdict

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
                              lw=4, fc='none', zorder=2,
                              ec=Feature.orig_colors['ellip'],
                              alpha=Feature.orig_alphas['ellip']))

    return ellips

class Track(object) :
    def __init__(self, x, y, frames, features=None) :
        self.obj = Line2D(x, y, color='b', lw=3, marker='.', ms=5)
        self.frames = frames
        self.features = features

    def __len__(self) :
        return len(self.frames)

    def get_data(self) :
        return zip(*self.obj.get_data())

    def add_feature(self, feature) :
        """
        Add a feature to the track at feature's frameNum.
        If there is already a feature at the frame, bump it out of
        the track.
        """
        if feature.frame in self.frames :
            index = self.frames.index(feature.frame)
            self.remove_feature(self.features[index])

        frameNum = feature.frame
        index = bisect_left(self.frames, frameNum)
        xs, ys = self.obj.get_data()
        xs = list(xs)
        ys = list(ys)
        x, y = feature.center()
        if index == len(self.frames) :
            self.frames.append(frameNum)
            self.features.append(feature)
            xs.append(x)
            ys.append(y)
        else :
            self.frames.insert(index, frameNum)
            self.features.insert(index, feature)
            xs.insert(index, x)
            ys.insert(index, y)

        self.obj.set_data(xs, ys)

    def update_frame(self, frameNum) :
        index = self.frames.index(frameNum)
        xs, ys = self.obj.get_data()
        x, y = self.features[index].center()
        xs[index] = x
        ys[index] = y
        self.obj.set_data(xs, ys)

    def remove_feature(self, feature) :
        index = self.features.index(feature)
        xs, ys = self.obj.get_data()
        xs = list(xs)
        ys = list(ys)
        xs.pop(index)
        ys.pop(index)
        self.frames.pop(index)
        self.features.pop(index)
        self.obj.set_data(xs, ys)
        if feature.track is self :
            feature.track = None

    def remove(self) :
        if self.obj is not None :
            self.obj.remove()


class StateManager(object) :
    """
    Manages the list of tracks and features to make sure any changes in
    one results in changes to the other.
    """
    def __init__(self) :
        self._tracks = []
        # Features organized by frameNum
        self._features = defaultdict(list)

    def load_features(self, tracks, falarms, volume) :
        #frameDiff = len(self.radarData) - len(self.volume['volume_data'])
        #if len(self.volume['volume_data']) > 0 :
        #    startIndex = self.volume['volume_data'][0]['frameNum']
        #else :
        #    startIndex = 0

        #volTimes = [vol['volTime'] for vol in self.volume['volume_data']]
        #if len(volTimes) > 0 :
        #    startTime = volTimes[0]
        #    if len(volTimes) > 1 :
        #        assumeDeltaT = np.median(np.diff(volTimes[::-1]))
        #    else :
        #        assumeDeltaT = 1.0
        #else :
        #    startTime = 0.0
        #    assumeDeltaT = 1.0

        #if frameDiff < 0 :
        #    raise ValueError("Previous session had more frames than available"
        #                     "input data frames")
        #elif frameDiff > 0 :
        #    newFrames = [{"volTime": (((len(self.volume['volume_data']) +
        #                                index) * assumeDeltaT) + startTime),
        #                  "frameNum": (len(self.volume['volume_data']) +
        #                               index + startIndex),
        #                  "stormCells": np.array([],
        #                                         dtype=TrackUtils.corner_dtype)
        #                  }
        #                 for index in xrange(frameDiff)]
        #    self.volume['volume_data'].extend(newFrames)
        #    self.volume['frame_cnt'] = len(self.radarData)

        # Features keyed by cornerID
        allKnownFeats = {}
        for index in xrange(len(volume['volume_data'])) :
            cells = volume['volume_data'][index]['stormCells']
            frameNum = volume['volume_data'][index]['frameNum']
            
            for cellIndex in xrange(len(cells)) :
                newPoint = self._new_point(cells['xLocs'][cellIndex],
                                           cells['yLocs'][cellIndex])
                newFeat = Feature(frameNum, center=newPoint,
                                  area=cells['sizes'][cellIndex])
                self._features[frameNum].append(newFeat)
                allKnownFeats[cells['cornerIDs'][cellIndex]] = newFeat

        for trackID, track in enumerate(tracks) :
            newTrack = Track(track['xLocs'], track['yLocs'], track['frameNums'])
            self._tracks.append(newTrack)

            feats = []
            for index, cornerID in enumerate(track['cornerIDs']) :
                if cornerID in allKnownFeats :
                    allKnownFeats[cornerID].track = newTrack
                    feats.append(allKnownFeats[cornerID])
                else :
                    newPoint = self._new_point(track['xLocs'][index],
                                               track['yLocs'][index])
                    newFeat = Feature(track['frameNums'][index],
                                      center=newPoint)
                    self._features[track['frameNums'][index]].append(newFeat)
                    feats.append(newFeat)
                    newFeat.track = newTrack
                    allKnownFeats[cornerID] = newFeat
            newTrack.features = feats

        for falarm in falarms :
            if falarm['cornerIDs'][0] not in allKnownFeats :
                newPoint = self._new_point(falarm['xLocs'][0],
                                           falarm['yLocs'][0])
                newFeat = Feature(falarm['frameNums'][0],
                                  center=newPoint)
                self._features[falarm['frameNums'][0]].append(newFeat)


    def save_features(self) :
        """
        Update the track and volume data.
        """
        ## Gather the times that were examined (and thus known)
        #data_times = []
        #data_frames = []
        #for index, aTime in enumerate(self.volTimes) :
        #    if aTime is not None :
        #        timeDiff = (aTime - self.volTimes[0]).total_seconds()
        #        data_times.append(timeDiff / 60.0)
        #        data_frames.append(index)

        #volTimes = [vol['volTime'] for vol in self.volume['volume_data']]

        # Assume that common known times between orig_volTime and
        # data_times are the same
        #for aTime, frame in zip(data_times, data_frames) :
        #    if np.isnan(volTimes[frame]) :
        #        volTimes[frame] = aTime

        #if np.sum(np.isfinite(volTimes)) == 0 :
        #    # There is no time information available to use
        #    # So just do a np.arange()
        #    volTimes = np.arange(len(self.volume))
        #else :
        #    # TODO: Try and fill in this data by assuming linear spacing
        #    pass


        # dictionary with "volume_data", "frameCnt", "corner_filestem"
        #    and "volume_data" contains a list of dictionaries with
        #    "volTime", "frameNum", "stormCells" where "stormCells" is
        #    a numpy array of dtype corner_dtype
        volume = {'frameCnt': 0,
                  'corner_filestem': '',
                  'volume_data': []}
        tracks = []
        falarms = []
               
        featIndex = 0
        # cornerIDs keyed by feature objects
        allKnownFeatures = {}
        for frameNum, features in self._features.iteritems() :
            # Any features that do not have a track are False Alarms
            for index, feat in enumerate(features) :
                allKnownFeatures[feat] = featIndex + index
                if feat.track is None :
                    pos = feat.center()
                    tmp = np.array([(pos[0], pos[1], featIndex + index,
                                     np.nan, np.nan, frameNum, 'F')],
                                   dtype=TrackUtils.base_track_dtype)
                    falarms.append(tmp)

            # TODO: get volume times working
            vol = {'volTime' : np.nan,
                   'frameNum' : frameNum,
                   'stormCells' : np.array([(feat.center()[0], feat.center()[1],
                                             feat.area(), featIndex + i)
                                            for i, feat in enumerate(features)],
                                           dtype=TrackUtils.corner_dtype)}
            volume['frameCnt'] += 1
            volume['volume_data'].append(vol)
            featIndex += len(features)

        
        for track in self._tracks :
            tmp = np.array([(pos[0], pos[1], allKnownFeatures[feat],
                             pos[0], pos[1], frameNum, 'M') for
                            pos, feat, frameNum in zip(track.get_data(),
                                                       track.features,
                                                       track.frames)],
                           dtype=TrackUtils.base_track_dtype)
            tracks.append(tmp)

        return tracks, falarms, volume

    def _new_point(self, x, y) :
        return Circle((x, y), fc='red', picker=None,
                      ec=Feature.orig_colors['center'], radius=6, lw=2,
                      alpha=Feature.orig_alphas['center'])


    def add_feature(self, feature) :
        # TODO: may need to do some mapping of frameIndex to frameNum
        self._features[feature.frame].append(feature)

    def remove_feature(self, feature) :
        self._features[feature.frame].remove(feature)
        self._cleanup_feature(feature)
        if len(self._features[feature.frame]) == 0 :
            self._features.pop(feature.frame)

    def _cleanup_feature(self, feature) :
        if feature.track is not None :
            # The remove_feature() function to be called could
            # set feature.track to None, so we preserve a reference
            # to that track so that we can clean it up afterwards.
            track = feature.track
            track.remove_feature(feature)
            if len(track) == 0 :
                self._tracks.remove(track)
                # There is no more track to show, so remove it
                # from the axes.
                track.remove()
            feature.track = None

    def clear_frame(self, frame) :
        frameCnt = len(self._features[frame])
        for featIndex in xrange(frameCnt) :
            feat = self._features[frame].pop()
            self._cleanup_feature(self, feat)
        
        self._features.pop(frame)

    def associate_features(self, feat1, feat2) :
        newTrack = None

        if feat1.track is not None :
            if feat2.track is not None :
                if feat1.track is feat2.track :
                    # Already associated, so no need to do anything
                    return
                feat2.track.remove_feature(feat2)

            feat2.track = feat1.track
            feat1.track.add_feature(feat2)

        elif feat2.track is not None :
            # Ah, maybe we are going backwards?
            feat1.track = feat2.track
            feat2.track.add_feature(feat1)
        else :
            # Neither had an existing track, so start a new one!
            if feat1.frame > feat2.frame :
                # Keep them sorted by frame
                tmp = feat2
                feat2 = feat1
                feat1 = tmp

            xs, ys = zip(feat1.center(), feat2.center())
            newTrack = Track(xs, ys, [feat1.frame, feat2.frame], [feat1, feat2])
            self._tracks.append(newTrack)
            feat1.track = newTrack
            feat2.track = newTrack

        return newTrack

class Feature(object) :
    orig_colors = {'contour': 'k', 'center': 'k', 'ellip': 'r'}
    orig_alphas = {'contour': 0.5, 'center': 0.75, 'ellip': 1.0}
    orig_zorders = {'contour': 1.0, 'center': 3.0, 'ellip': 2.0}
    def __init__(self, frame, contour=None, center=None, ellip=None,
                       area=None, track=None) :
        self.frame = frame
        self.track = track
        self.objects = {}
        if contour is not None :
            self.objects['contour'] = contour
        if center is not None :
            self.objects['center'] = center
        if ellip is not None :
            self.objects['ellip'] = ellip

        self.feat_area = area if (area is not None and
                                  not np.isnan(area)) else None

    def remove(self) :
        for key, item in self.objects.iteritems() :
            if item is not None :
                item.remove()

        self.objects = {}

    def center(self) :
        if self.objects.get('center', None) is not None :
            return self.objects['center'].center
        elif self.objects.get('ellip', None) is not None :
            return self.objects['ellip'].center
        elif self.objects.get('contour', None) is not None :
            return np.mean(self.objects['contour'].get_xy(), axis=0)

        return (np.nan, np.nan)

    def area(self) :
        if self.feat_area is not None :
            return self.feat_area
        elif self.objects.get('ellip', None) is not None :
            ellip = self.objects['ellip']
            return 4 * ellip.height * ellip.width
        # TODO: Get the correct equation...

        return np.nan

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
                item.set_edgecolor(Feature.orig_colors.get(key, 'k'))

    def get_visible(self) :
        """
        If any objects are visible, then return True.
        """
        return any(item.get_visible() for item in self.objects.values() if
                   item is not None)

    def set_visible(self, visible) :
        """
        Set visible state for all objects.
        """
        for key, item in self.objects.iteritems() :
            if item is not None :
                item.set_visible(visible)

    def set_zorder(self, zorder) :
        """
        Set the zorder of the objects of the feature.
        """
        for key, item in self.objects.iteritems() :
            if item is not None :
                item.set_zorder(zorder +
                                (Feature.orig_zorders.get(key, 0.0) /
                                 len(Feature.orig_zorders)))

    def set_picker(self, picker) :
        for key, item in self.objects.iteritems() :
            if item is not None :
                item.set_picker(picker)

    def set_alpha(self, alpha) :
        for key, item in self.objects.iteritems() :
            if item is not None :
                item.set_alpha(alpha * Feature.orig_alphas.get(key, 1.0))

    def contains(self, event) :
        return any([item.contains(event)[0] for
                    item in self.objects.values()])
 


class RadarDisplay(object) :
    _increms = {'left': -1, 'right': 1}

    def do_save_results(self) :
        return self._do_save

    def __init__(self, volume, tracks, falarms, radarFiles) :
        """
        Create an interactive display for creating track data
        from radar data.
        """
        minLat, minLon, maxLat, maxLon = ConsistentDomain(radarFiles)


        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.radarData = RadarCache(radarFiles, 4)
        
        self._increm_funcs = {'left': self.radarData.prev,
                              'right': self.radarData.next}

        self._im = None
        self._curr_selection = None
        self._alphaScale = 1.0
        self.curr_lasso = None
        self.volTimes = [None] * len(self.radarData)

        self.state = StateManager()
        self.state.load_features(tracks, falarms, volume)

        for feats in self.state._features.values() :
            for feat in feats :
                for obj in feat.objects.values() :
                    if obj is not None :
                        obj.set_visible(False)
                        self.ax.add_artist(obj)

        for track in self.state._tracks :
            track.obj.set_visible(True)
            self.ax.add_artist(track.obj)

        self.frameIndex = 0
        self._show_features = False
        data = self.radarData.next()
        lons, lats = np.meshgrid(data['lons'], data['lats'])
        self.xs, self.ys = LonLat2Cart((minLon + maxLon)/2.0,
                                       (minLat + maxLat)/2.0,
                                        lons, lats)

        self.update_frame()

        self.ax.set_xlim(self.xs.min(), self.xs.max())
        self.ax.set_ylim(self.ys.min(), self.ys.max())
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')


        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        #self.fig.canvas.mpl_connect('button_release_event',
        #                             self.process_click)
        self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        #self.fig.canvas.mpl_connect('pick_event', self.onpick)
        #self._savefeats_cid = self.fig.canvas.mpl_connect('close_event',
        #                                                  self.onclose)
        self._do_save = True

        # Start in outline mode
        self._mode = 'o'

        # Need to remove some keys...
        rcParams['keymap.fullscreen'] = []
        rcParams['keymap.home'].remove('r')
        rcParams['keymap.home'].remove('h')
        rcParams['keymap.back'].remove('left')
        rcParams['keymap.forward'].remove('right')
        rcParams['keymap.forward'].remove('v')
        rcParams['keymap.zoom'] = []
        rcParams['keymap.save'] = []

        print "Welcome to Track Maker! (press 'h' for menu of options)"

    def onpick(self, event) :
        """
        Track picker handler
        """
        pass

    #def onclose(self, event) :
    #    """
    #    Trigger a saving of data (Note: this isn't a saving to files,
    #    but to the track lists and volume lists).
    #    """
    #    self.save_features()


    def onlasso(self, verts) :
        """
        Creation of the contour polygon, which selects the initial
        region for watershed clustering.
        """
        newPoly = Polygon(verts, lw=2, fc='gray', hatch='/', zorder=1,
                          ec=Feature.orig_colors['contour'],
                          alpha=Feature.orig_alphas['contour'], picker=None)
        self.state.add_feature(Feature(self.frameIndex, contour=newPoly))
        self.fig.canvas.draw_idle()
        self.fig.canvas.widgetlock.release(self.curr_lasso)
        del self.curr_lasso
        self.curr_lasso = None
        self.ax.add_artist(newPoly)

    def onpress(self, event) :
        """
        Button-press handler
        """
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

            curr_select = None
            for feat in self.state._features[self.frameIndex] :
                if feat.contains(event) :
                    curr_select = feat
            prev_select = self._curr_selection

            if curr_select is not None :
                if prev_select is not None :
                    # TODO: Make a tracking association
                    # TODO: May need some mapping of frameIndex to frameNum
                    # This is only valid if I am able to see the previous
                    # selection. This logic also makes it impossible to
                    # have feat1 be the same object as feat2
                    if (prev_select.get_visible() and
                        self.frameIndex != prev_select.frame) :
                        # We are making associations across frames!
                        tmp = self.state.associate_features(prev_select,
                                                            curr_select)

                        if tmp is not None :
                            self.ax.add_artist(tmp.obj)

                    prev_select.deselect()

                if prev_select is curr_select :
                    self._curr_selection = None
                else :
                    self._curr_selection = curr_select
                    curr_select.select()

                self.fig.canvas.draw_idle()


    def process_key(self, event) :
        """
        Key-press handler
        """
        if event.key in RadarDisplay._increms :
            if (0 <= (self.frameIndex + RadarDisplay._increms[event.key])
                  <= (len(self.radarData) - 1)) :
                lastFrame = self.frameIndex
                self.frameIndex += RadarDisplay._increms[event.key]

                # Update the radar data
                self._increm_funcs[event.key]()

                #if self._curr_selection is not None :
                #    self._curr_selection.deselect()
                #    self._curr_selection = None

                # Update the frame
                self.update_frame(lastFrame, hold_recluster=True)

        elif event.key == '+' :
            self._alphaScale = min(100.0, self._alphaScale * 2)
            self.update_frame(hold_recluster=True)

        elif event.key == '-' :
            self._alphaScale = max(0.001, self._alphaScale / 2.0)
            self.update_frame(hold_recluster=True)

        elif event.key == 'r' :
            # Recalculate ellipsoids
            self.update_frame(force_recluster=True, hold_recluster=False)

        elif event.key == 'c' :
            # Completely remove the features for this frame
            if (self._curr_selection is not None and
                self._curr_selection.frame == self.frameIndex) :
                self._curr_selection.deselect()
                self._curr_selection = None

            self.state.clear_features(self.frameIndex)

            self.update_frame()

        elif event.key == 'd' :
            # Delete the currently selected artist (if in the current frame)
            if (self._curr_selection is not None and
                self._curr_selection.frame == self.frameIndex) :
                self._curr_selection.deselect()
                self._curr_selection.remove()
                self.state.remove_feature(self._curr_selection)
                self._curr_selection = None

            self.fig.canvas.draw_idle()

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

        elif event.key == 'f' :
            # Show/Hide all identified features across time
            self._show_features = (not self._show_features)
            print "Show features:", self._show_features
            self.update_frame(hold_recluster=True)

        elif event.key == 'v' :
            # Toogle save
            self._do_save = (not self._do_save)
            print "Do Save:", self._do_save
            #if not self._do_save :
            #    if self._savefeats_cid is not None :
            #        self.fig.canvas.mpl_disconnect(self._savefeats_cid)
            #        self._savefeats_cid = None
            #else :
            #    if self._savefeats_cid is None :
            #        self._savefeats_cid = self.fig.canvas.mpl_connect(
            #                                "close_event", self.onclose)

        #elif event.key == 'V' :
        #    # Save features to memory NOW!
        #    print "Converting to track and volume objects, NOW!"
        #    self.save_features()

        elif event.key == 'h' :
            # Print helpful Menu
            print dedent("""
                Track Maker
                ===========

                Key         Action
                ------      -----------------------------
                h           Show this helpful menu
                right       Step forward by one frame
                left        Step back by one frame
                +, -        Rescale transparency as a function of time
                o           Outline mode
                s           Selection mode
                r           (re)cluster this frame
                c           clear this frame of existing features
                d           delete the currently selected feature
                s           show/hide all features across all time
                v           toggle saving features upon closing figure
                                (this is useful if you detect an error and
                                 do not want to save bad data).

                Current Values
                --------------
                    Current Frame: %d of %d
                    Current Mode: %s
                    Do save upon figure close: %s
                    Show all features: %s
                """ % (self.frameIndex + 1, len(self.radarData), self._mode,
                       self._do_save, self._show_features))


    def _clear_frame(self, frame=None) :
        if frame is None :
            frame = self.frameIndex

        # Set the frame's features to invisible
        for feat in self.state._features[frame] :
            feat.set_visible(False)
            # Also reset their alpha values
            feat.set_alpha(1.0)
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

        for index, feat in enumerate(self.state._features[self.frameIndex]) :
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
        clustCnt = len(self.state._features[self.frameIndex])

        cents = ndimg.center_of_mass(data, clustLabels,
                                     range(1, clustCnt + 1))
        ellipses = FitEllipses(clustLabels, range(1, clustCnt + 1),
                               self.xs, self.ys)

        for center, ellip, feat in zip(cents, ellipses,
                                       self.state._features[self.frameIndex]) :
            # Remove any other objects that may exist before adding
            # new objects to the feature.
            feat.cleanup(['contour'])

            if ellip is None :
                continue

            cent_indx = tuple(np.floor(center).astype(int).tolist())
            # TODO: clean this up!
            newPoint = self.state._new_point(self.xs[cent_indx],
                                             self.ys[cent_indx])
            self.ax.add_artist(ellip)

            feat.objects['center'] = newPoint
            feat.objects['ellip'] = ellip

            if feat.track is not None :
                feat.track.update_frame(self.frameIndex)

        #print "clust count:", clustCnt
        return clustLabels, clustCnt

    def _xy2grid(self, x, y) :
        return (self.xs[0].searchsorted(x),
                self.ys[:, 0].searchsorted(y))


    def update_frame(self, lastFrame=None,
                           force_recluster=False, hold_recluster=False) :
        """
        Redraw the current frame.  Calculate clusters if needed.

        *lastFrame*         int (None)
            If specified, make this frame's features invisible.

        *force_recluster*   boolean (False)
            If True, do a recluster, even if it seems like it isn't needed.
            Can be over-ridden by *hold_recluster*.

        *hold_recluster*    boolean (False)
            If True, then don't do a recluster, even if needed or
            *force_recluster* is True.
        """
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
                            feat in self.state._features[self.frameIndex]]) :
            if not hold_recluster :
                clustLabels, clustCnt = self.get_clusters()

        # Set features for this frame to visible
        for feat in self.state._features[self.frameIndex] :
            feat.set_visible(True)
            # Return alpha back to normal
            feat.set_alpha(1.0)
            # Put it on top
            feat.set_zorder(len(self.radarData))
            #feat.set_picker(True)

        # Show the other features
        if self._show_features :
            # How much alpha should change for each frame from frameIndex
            # The closer to self.frameIndex, the more opaque
            alphaIncrem = 1.0 / len(self.radarData)
            for frameIndex, features in self.state._features.iteritems() :
                if frameIndex == self.frameIndex :
                    continue

                framesFrom = np.abs(self.frameIndex - frameIndex)
                timeAlpha = ((1.0 - (framesFrom * alphaIncrem)) **
                             (1.0/self._alphaScale))
                zorder = len(self.radarData) - framesFrom
                for feat in features :
                    feat.set_visible(True)
                    feat.set_alpha(timeAlpha)
                    feat.set_zorder(zorder)
        else :
            # Make sure that these items are hidden
            for frameIndex, features in self.state._features.iteritems() :
                if frameIndex != self.frameIndex :
                    for feat in features :
                        feat.set_visible(False)
                        # Return alpha to normal
                        feat.set_alpha(1.0)

        if self.volTimes[self.frameIndex] is None :
            theDateTime = datetime.utcfromtimestamp(data['scan_time'])
            self.volTimes[self.frameIndex] = theDateTime
        else :
            theDateTime = self.volTimes[self.frameIndex]

        self.ax.set_title(theDateTime.strftime("%Y/%m/%d %H:%M:%S"))
        self.fig.canvas.draw_idle()


   

def AnalyzeRadar(volume, tracks, falarms, radarFiles) :

    rd = RadarDisplay(volume, tracks, falarms, radarFiles)
    plt.show()

    return rd.do_save_results(), rd.state

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
        sceneParams['corner_file'] = volume['corner_filestem']


    #reflectData = [LoadRastRadar(radarFile) for radarFile in args.input_files]

    do_save, state = AnalyzeRadar(volume, tracks, falarms, args.input_files)

    if do_save :
        # Create the directory if it doesn't exist already
        if not os.path.exists(dirName) :
            makedirs(dirName)

        tracks, falarms, volume = state.save_features()
        volume['corner_filestem'] = sceneParams['corner_file']

        # Final save
        SaveState(paramFile, sceneParams, volumeFile, volume,
                  origTrackFile, filtTrackFile, tracks, falarms)


def SaveState(paramFile, params, volumeFile, volume,
              origTrackFile, filtTrackFile, tracks, falarms) :
    # Do I need to update the Params?
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

