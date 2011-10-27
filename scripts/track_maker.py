#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.widgets import Lasso
from matplotlib.nxutils import points_inside_poly
#from mpl_toolkits.basemap import Basemap

import ZigZag.TrackFileUtils as TrackFileUtils
import ZigZag.TrackUtils as TrackUtils
import ZigZag.ParamUtils as ParamUtils

from BRadar.io import RadarCache, LoadRastRadar
from BRadar.plotutils import RadarDisplay, BaseControlSys
from BRadar.maputils import LonLat2Cart
import BRadar.radarsites as radarsites

import numpy as np
import scipy.ndimage as ndimg
import os.path
from os import makedirs
from datetime import datetime
from textwrap import dedent
from bisect import bisect_left
from collections import defaultdict
from cPickle import load, dump

def ConsistentDomain(radarFiles) :
    minLat = None
    minLon = None
    maxLat = None
    maxLon = None
    volTimes = []

    for fname in radarFiles :
        data = LoadRastRadar(fname)

        volTimes.append(datetime.utcfromtimestamp(data['scan_time']))
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

    return (minLat, minLon, maxLat, maxLon, volTimes)

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
        self.obj = Line2D(x, y, color='b', lw=2, marker='.', ms=5,
                          zorder=(max(frames) + 1), alpha=0.7)
        self.frames = list(frames)
        self.features = features

        if features is not None :
            # Color these feature dots (if they have them) to indicate
            # that they have been associated.
            for feat in features :
                feat.mark_associated(True)
                self._bookkeeping(feat)

    def __len__(self) :
        return len(self.frames)

    def get_data(self) :
        return zip(*self.obj.get_data())

    def _bookkeeping(self, feat) :
        if feat.track is not None and feat.track is not self :
            feat.track.remove_feature(feat)

        feat.track = self

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
        self.frames.insert(index, frameNum)
        self.features.insert(index, feature)
        xs.insert(index, x)
        ys.insert(index, y)
        self.obj.set_zorder(max(self.frames) + 1)

        # Change face color to indicate that it is associated with a track
        feature.mark_associated(True)
        self.obj.set_data(xs, ys)

        self._bookkeeping(feature)

    def join(self, track) :
        """
        Merge this track with another track.

        Also performs the needed bookkeeping.
        """
        # First, make sure they do not share any frames
        shared_frames = (set(self.frames) & set(track.frames))
        if len(shared_frames) > 0 :
            raise ValueError("Can not join these tracks, they share frames.")

        if len(track.frames) == 0 :
            # Don't need to do anything...
            return


        xs, ys = self.obj.get_data()
        xs = list(xs)
        ys = list(ys)

        insertInds = np.searchsorted(self.frames, track.frames)
        newxs, newys = track.obj.get_data()

        # Going in reverse so that the insertInds values are valid
        for ind, feat, frame, newx, newy in reversed(zip(insertInds,
                                                         track.features,
                                                         track.frames,
                                                         newxs, newys)) :
            self.frames.insert(ind, frame)
            self.features.insert(ind, feat)
            xs.insert(ind, newx)
            ys.insert(ind, newy)
            self._bookkeeping(feat)
            feat.mark_associated(True)

        self.obj.set_zorder(max(self.frames) + 1)
        self.obj.set_data(xs, ys)


    def update_frame(self, frameNum) :
        """
        Re-assess the point of the track at *frameNum*.

        Useful when the Feature object gets an updated center location.
        """
        index = self.frames.index(frameNum)
        xs, ys = self.obj.get_data()
        x, y = self.features[index].center()
        xs[index] = x
        ys[index] = y
        self.obj.set_data(xs, ys)

    def remove_feature(self, feature) :
        """
        Remove *feature* from this track.
        Also performs any needed bookkeeping if it hasn't
        occured already.
        """
        index = self.features.index(feature)
        xs, ys = self.obj.get_data()
        xs = list(xs)
        ys = list(ys)
        xs.pop(index)
        ys.pop(index)
        self.frames.pop(index)
        self.features.pop(index)

        # Reset the face color to indicate that this feature is no longer
        # associated with a track
        feature.mark_associated(False)
        self.obj.set_data(xs, ys)

        # Only modify this if it refers to itself
        if feature.track is self :
            feature.track = None

    def breakup(self, feature) :
        """
        Breakup this track at the point where *feature* is.
        This truncates *self* and returns a new Track object
        that is composed of the features from *feature* and
        later.
        """
        index = self.features.index(feature)
        xs, ys = self.obj.get_data()
        newTrack = Track(xs[index:], ys[index:], self.frames[index:])
        for feat in self.features[index:] :
            feat.track = newTrack
        newTrack.features = self.features[index:]

        self.frames = self.frames[:index]
        self.features = self.features[:index]
        self.obj.set_data(xs[:index], ys[:index])

        return newTrack

    def remove(self) :
        if self.obj is not None :
            self.obj.remove()


class StateManager(object) :
    """
    Manages the list of tracks and features to make sure any changes in
    one results in changes to the other.
    """
    def __init__(self, volTimes=None) :
        self._tracks = []
        # Features organized by frameNum
        self._features = defaultdict(list)
        self._volTimes = volTimes

    def load_features(self, tracks, falarms, volume, polygons) :
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
        for frameID, vol in enumerate(volume['volume_data']) :
            cells = vol['stormCells']
            frameNum = vol['frameNum']
            
            for cellIndex in xrange(len(cells)) :
                newPoint = StateManager._new_point(cells['xLocs'][cellIndex],
                                                   cells['yLocs'][cellIndex])
                cornerID = cells['cornerIDs'][cellIndex]
                newPoly = None
                if cornerID in polygons :
                    newPoly = StateManager._new_polygon(polygons[cornerID])

                newFeat = Feature(frameNum, center=newPoint, contour=newPoly,
                                  area=cells['sizes'][cellIndex])
                self._features[frameNum].append(newFeat)
                allKnownFeats[cornerID] = newFeat

        for trackID, track in enumerate(tracks) :
            newTrack = Track(track['xLocs'], track['yLocs'], track['frameNums'])
            self._tracks.append(newTrack)

            feats = []
            for index, cornerID in enumerate(track['cornerIDs']) :
                if cornerID in allKnownFeats :
                    newFeat = allKnownFeats[cornerID]
                else :
                    newPoint = StateManager._new_point(track['xLocs'][index],
                                                       track['yLocs'][index])
                    newPoly = None
                    if cornerID in polygons :
                        newPoly = StateManager._new_polygon(polygons[cornerID])

                    newFeat = Feature(track['frameNums'][index],
                                      center=newPoint, contour=newPoly)
                    self._features[track['frameNums'][index]].append(newFeat)
                    allKnownFeats[cornerID] = newFeat

                newFeat.track = newTrack
                feats.append(newFeat)
                # Color the point to indicate that it is
                # associated with a track.
                newFeat.mark_associated(True)

            newTrack.features = feats

        for falarm in falarms :
            if falarm['cornerIDs'][0] not in allKnownFeats :
                newPoint = StateManager._new_point(falarm['xLocs'][0],
                                                   falarm['yLocs'][0])
                newPoly = None
                cornerID = falarm['cornerIDs'][0]
                if cornerID in polygons :
                    newPoly = StateManager._new_polygon(polygons[cornerID])

                newFeat = Feature(falarm['frameNums'][0],
                                  center=newPoint, contour=newPoly)
                self._features[falarm['frameNums'][0]].append(newFeat)


    def save_features(self, xs, ys) :
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

        # Use the self._volTimes array, if available and if the frame numbers
        # for the features make sense.
        if (self._volTimes is not None and
            len(self._volTimes) >= max(self._features.keys()) and
            len(self._volTimes) > 0) :
            # times in minutes
            volTimes = [(aTime - self._volTimes[0]).total_seconds()/60.0 for
                        aTime in self._volTimes]
        else :
            # Just assume volume times of increments of one.
            volTimes = np.arange(max(self._features.keys()))

        gridx_grads = np.gradient(xs)
        gridy_grads = np.gradient(ys)

        # dictionary with "volume_data", "frameCnt", "corner_filestem"
        #    and "volume_data" contains a list of dictionaries with
        #    "volTime", "frameNum", "stormCells" where "stormCells" is
        #    a numpy array of dtype corner_dtype
        volume = {'frameCnt': 0,
                  'corner_filestem': '',
                  'volume_data': []}
        tracks = []
        falarms = []

        # dict of (cornerID, polygon) key/value pairs
        polygons = {}

        featIndex = 0
        # cornerIDs keyed by feature objects
        allKnownFeatures = {}
        for frameNum, features in self._features.iteritems() :
            # Any features that do not have a track are False Alarms
            for index, feat in enumerate(features) :
                allKnownFeatures[feat] = featIndex + index

                if np.isnan(feat.area()) and ('contour' in feat.objects) :
                    feat.feat_area = Feature.calc_area_polygon(xs, ys,
                                                               feat.get_xy(),
                                                               gridx_grads,
                                                               gridy_grads)


                if feat.track is None :
                    pos = feat.center()
                    tmp = np.array([(pos[0], pos[1], featIndex + index,
                                     np.nan, np.nan, frameNum, 'F')],
                                   dtype=TrackUtils.base_track_dtype)
                    falarms.append(tmp)

            # TODO: get volume times working
            vol = {'volTime' : volTimes[frameNum],
                   'frameNum' : frameNum,
                   'stormCells' : np.array([(feat.center()[0], feat.center()[1],
                                             feat.area(), featIndex + i)
                                            for i, feat in enumerate(features)],
                                           dtype=TrackUtils.corner_dtype)}
            polygons.update((i + featIndex, feat.get_xy()) for
                             i, feat in enumerate(features))
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

        return tracks, falarms, volume, polygons

    @staticmethod
    def _new_point(x, y) :
        return Circle((x, y), fc='red', picker=None,
                      ec=Feature.orig_colors['center'], radius=4, lw=2,
                      alpha=Feature.orig_alphas['center'])

    @staticmethod
    def _new_polygon(verts) :
        return Polygon(verts, lw=2, fc='gray', hatch='/', zorder=1,
                       ec=Feature.orig_colors['contour'],
                       alpha=Feature.orig_alphas['contour'],
                       picker=None)

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

    def _cleanup(self) :
        """
        Remove any empty tracks.
        """
        for track in reversed(self._tracks) :
            if len(track) == 0 :
                self._tracks.remove(track)
                track.remove()

    def clear_frame(self, frame) :
        frameCnt = len(self._features[frame])
        for featIndex in xrange(frameCnt) :
            feat = self._features[frame].pop()
            feat.remove()
            feat.track = None

        self._cleanup()
        self._features.pop(frame)

    def associate_features(self, feat1, feat2, assoc_act) :
        """
        Perform the association of *feat1* and *feat2* into a single
        track. If a new track is created, then return it.  Otherwise,
        return None.

        *assoc_act*     int : [0, 1, 2]
            If both features already belong to their own tracks,
            then the process of association will remove *feat2*
            from its track and add it to *feat1*'s track.
            This action is called 'steal', and happens when
            *assoc_act* is 0.

            However, if *assoc_act* is 1, then this association
            process will 'join' the tracks together into one track, so
            long as none of features share a common frame.
            If a collision occurs, and exception is raised.

            Next, if *assoc_act* is 2, then *feat2* is removed from
            *feat1*'s track. This is the 'removal' mode.

            Lastly, if *assoc_act* is 3, then this association process
            will 'breakup' the tracks into two separate tracks.
            A new track is created and is returned.

            For the last two, nothing happens if both features do not
            belong to the same track.
        """
        newTrack = None

        # If either feature is missing its point, add it.
        # I would rather this be elsewhere, but the refactor
        # should handle that.
        if 'center' not in feat1.objects :
            newCent = StateManager._new_point(*feat1.center())
            feat1.objects['center'] = newCent
            feat1.objects['contour'].get_axes().add_artist(newCent)
            newCent.set_visible(True)

        if 'center' not in feat2.objects :
            newCent = StateManager._new_point(*feat2.center())
            feat2.objects['center'] = newCent
            feat2.objects['contour'].get_axes().add_artist(newCent)
            newCent.set_visible(True)

        if assoc_act in [0, 1] :
            # Steal Mode
            if feat1.track is not None :
                if feat2.track is not None :
                    if feat1.track is feat2.track :
                        # Already associated, so no need to do anything
                        return newTrack
                    else :
                        if assoc_act == 1 :
                            feat1.track.join(feat2.track)
                        else :
                            feat1.track.add_feature(feat2)
                    self._cleanup()
                else :
                    feat1.track.add_feature(feat2)

            elif feat2.track is not None :
                # Ah, maybe we are going backwards?
                feat2.track.add_feature(feat1)
            else :
                # Neither had an existing track, so start a new one!
                if feat1.frame > feat2.frame :
                    # Keep them sorted by frame
                    feat1, feat2 = feat2, feat1

                xs, ys = zip(feat1.center(), feat2.center())
                newTrack = Track(xs, ys, [feat1.frame, feat2.frame],
                                 [feat1, feat2])
                self._tracks.append(newTrack)

        elif assoc_act in [2, 3] :
            if (feat1.track is not None and feat2.track is not None
                and feat1.track is feat2.track) :
                if assoc_act == 2 :
                    # Removal mode
                    feat1.track.remove_feature(feat2)
                    self._cleanup()
                else :
                    # Breakup mode
                    newTrack = feat1.track.breakup(feat2)
                    self._tracks.append(newTrack)

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

    def mark_associated(self, assoc=True) :
        if 'center' in self.objects :
            self.objects['center'].set_facecolor('c' if assoc else 'r')

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

    def get_xy(self) :
        if self.objects.get('contour', None) is not None :
            return self.objects['contour'].get_xy()
        elif self.objects.get('ellip', None) is not None :
            return self.objects['ellip'].get_verts()
        elif self.objects.get('center', None) is not None :
            return [self.objects['center'].center]

        return [(np.nan, np.nan)]

    def area(self) :
        if self.feat_area is not None :
            return self.feat_area
        elif self.objects.get('ellip', None) is not None :
            ellip = self.objects['ellip']
            return (np.pi * ellip.height * ellip.width) / 4.0

        return np.nan

    @staticmethod
    def calc_area_mask(featMask, dxdj, dxdi, dydj, dydi) :
        """
        Core portion of calculating an area assuming that
        there is a gradient information of the domain grid
        and a boolean mask indicating where in the grid the feature
        exists.

        Approximates the area by assuming rhombus-shaped voxels.
        """
        a = np.hypot(dxdj[featMask], dydj[featMask])
        b = np.hypot(dxdi[featMask], dydi[featMask])
        dA = a * b
        return np.sum(dA)

    @staticmethod
    def calc_area_polygon(xs, ys, verts,
                          gridx_grads=None, gridy_grads=None) :
        """
        Calculate the area of a feature based on knowledge
        of the domain grid gradients and the polygon of the
        feature.

        *xs*, *ys* are the domain grids. 2-D numpy arrays.

        *verts* must be a list of x-y tuples.

        *gridx_grads*, *gridy_grads* are expected to be tuples from
            np.gradient() of *xs* and *ys* respectively.
            If not supplied, it will be calculated from *xs*, *ys*.

        Works by determining the mask and calling
        :meth:`calc_area_mask` to get the area.
        """
        if gridx_grads is None :
            gridx_grads = np.gradient(xs)
        if gridy_grads is None :
            gridy_grads = np.gradient(ys)

        res = points_inside_poly(zip(xs.flat, ys.flat), verts)
        res.shape = xs.shape

        return Feature.calc_area_mask(res, gridx_grads[0], gridx_grads[1],
                                           gridy_grads[0], gridy_grads[1])


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



class TM_ControlSys(BaseControlSys) :
    _assoc_mode_list = ['steal', 'join', 'remove', 'breakup']
    def __init__(self, fig, ax, rd, state) :
        BaseControlSys.__init__(self, fig, rd)

        self.ax = ax
        self._curr_selection = None
        self._group_selection = []
        self._alphaScale = 1.0
        self._curr_lasso = None
        self._visible = {}
        self.state = state

        for frame, feats in self.state._features.iteritems() :
            isCurrFrame = (frame == self.rd.frameIndex)
            for feat in feats :
                for obj in feat.objects.values() :
                    if obj is not None :
                        obj.set_visible(isCurrFrame)
                        ax.add_artist(obj)

        for track in self.state._tracks :
            track.obj.set_visible(True)
            ax.add_artist(track.obj)

        self._show_features = False
        self._do_save = True

        # Start in outline mode
        self._mode = 'o'

        # Start in 'steal' association mode
        self._assoc_mode = 0

        fig.canvas.mpl_connect('button_press_event', self.onpress)

        # TODO: Add to keymap
        self.keymap['+'] = {'func': self.increm_feature_alpha,
                            'help': "Increase feature opacity"}
        self.keymap['-'] = {'func': self.decrem_feature_alpha,
                            'help': "Decrement feature opacity"}
        self.keymap['r'] = {'func': self.recalculate_ellipses,
                            'help': "Recalculate this frame's features"}
        self.keymap['c'] = {'func': self.clear_frame,
                            'help': "Clear/Remove all features in this frame"}
        self.keymap['d'] = {'func': self.delete_feature,
                            'help': "Delete selected feature(s)"}
        self.keymap['s'] = {'func': self.selection_mode,
                            'help': "Selection mode"}
        self.keymap['S'] = {'func': self.multi_select_mode,
                            'help': "Multi-Selection Mode"}
        self.keymap['o'] = {'func': self.outline_mode,
                            'help': "Outline mode"}
        self.keymap['a'] = {'func': self.toggle_assoc_mode,
                            'help': "Toggle association mode "
                                    + str(TM_ControlSys._assoc_mode_list)}
        self.keymap['f'] = {'func': self.show_all_features,
                            'help': "Toggle displaying all features"}
        self.keymap['v'] = {'func': self.toggle_save,
                            'help': "To save, or not save?"}
        self.keymap['h'] = {'func': self.print_menu,
                            'help': "Display this helpful menu"}


        self._clean_mplkeymap()

        print "Welcome to Track Maker! (press 'h' for menu of options)"

    def onpick(self, event) :
        """
        Track picker handler
        """
        pass

    def do_save_results(self) :
        return self._do_save

    def onlasso(self, verts) :
        """
        Creation of the contour polygon, which selects the initial
        region for watershed clustering.
        """
        newPoly = Polygon(verts, lw=2, fc='gray', hatch='/', zorder=1,
                          ec=Feature.orig_colors['contour'],
                          alpha=Feature.orig_alphas['contour'], picker=None)
        newFeat = Feature(self.rd.frameIndex, contour=newPoly)
        self.state.add_feature(newFeat)
        self.fig.canvas.draw_idle()
        self.fig.canvas.widgetlock.release(self._curr_lasso)
        del self._curr_lasso
        self._curr_lasso = None
        self.ax.add_artist(newPoly)

    def lasso_selection(self, verts) :
        """
        Create a polygon for selecting many features.

        Such a selection can only be used for deletion
        """
        frame_feats = self.state._features[self.rd.frameIndex]
        cents = [feat.center() for feat in frame_feats]
        res = points_inside_poly(cents, verts)

        # Reset any existing selections
        if self._curr_selection is not None :
            self._curr_selection.deselect()
            self._curr_selection = None

        for feat in self._group_selection :
            feat.deselect()            

        self._group_selection = []
        for feat, inside in zip(frame_feats, res) :
            if inside :
                self._group_selection.append(feat)
                feat.select()

        self.fig.canvas.draw_idle()
        self.fig.canvas.widgetlock.release(self._curr_lasso)
        del self._curr_lasso
        self._curr_lasso = None



    def onpress(self, event) :
        """
        Button-press handler
        """
        if self.fig.canvas.widgetlock.locked() :
            return
        if event.inaxes is not self.ax :
            return

        if self._mode == 'o' :
            # Outline mode
            self._curr_lasso = Lasso(event.inaxes, (event.xdata, event.ydata),
                                     self.onlasso)

            # Set a lock on drawing the lasso until finished
            self.fig.canvas.widgetlock(self._curr_lasso)

        elif self._mode == 'S' :
            # Outline mode
            self._curr_lasso = Lasso(event.inaxes, (event.xdata, event.ydata),
                                     self.lasso_selection)

            # Set a lock on drawing the lasso until finished
            self.fig.canvas.widgetlock(self._curr_lasso)

        elif self._mode == 's':
            # Selection mode

            # Reset any group selections
            for feat in self._group_selection :
                feat.deselect()
            self._group_selection = []

            curr_select = None
            for feat in self.state._features[self.rd.frameIndex] :
                if feat.contains(event) :
                    curr_select = feat
            prev_select = self._curr_selection

            if curr_select is not None :
                if prev_select is not None :
                    # This is only valid if I am able to see the previous
                    # selection. This logic also makes it impossible to
                    # have feat1 be the same object as feat2
                    if (prev_select.get_visible() and
                        self.rd.frameIndex != prev_select.frame) :
                        # Perform an association action across frames!
                        tmp = self.state.associate_features(prev_select,
                                                            curr_select,
                                                            self._assoc_mode)

                        if tmp is not None :
                            self.ax.add_artist(tmp.obj)

                    prev_select.deselect()

                if prev_select is curr_select :
                    self._curr_selection = None
                else :
                    self._curr_selection = curr_select
                    curr_select.select()

                self.fig.canvas.draw_idle()

    def increm_feature_alpha(self) :
        self._alphaScale = min(100.0, self._alphaScale * 2)
        self.update_frame(hold_recluster=True)

    def decrem_feature_alpha(self) :
        self._alphaScale = max(0.001, self._alphaScale / 2.0)
        self.update_frame(hold_recluster=True)

    def recalculate_ellipses(self) :
        # Recalculate ellipsoids
        self.update_frame(force_recluster=True, hold_recluster=False)

    def clear_frame(self) :
        # Completely remove the features for this frame
        if (self._curr_selection is not None and
            self._curr_selection.frame == self.rd.frameIndex) :
            self._curr_selection.deselect()
            self._curr_selection = None

        for index in reversed(range(len(self._group_selection))) :
            feat = self._group_selection[index]
            if feat.frame == self.rd.frameIndex :
                feat.deselect()
                self._group_selection.pop(index)
            

        self.state.clear_frame(self.rd.frameIndex)
        #self.update_frame()

    def delete_feature(self) :
        # Delete the currently selected artist (if in the current frame)
        if (self._curr_selection is not None and
            self._curr_selection.frame == self.rd.frameIndex) :
            self._curr_selection.deselect()
            self._curr_selection.remove()
            self.state.remove_feature(self._curr_selection)
            self._curr_selection = None

        for index in reversed(range(len(self._group_selection))) :
            feat = self._group_selection[index]
            if feat.frame == self.rd.frameIndex :
                feat.deselect()
                feat.remove()
                self.state.remove_feature(feat)
                self._group_selection.pop(index)

    def selection_mode(self) :
        # set mode to "selection mode"
        self._mode = 's'

        # Just in case the canvas is still locked.
        if self._curr_lasso is not None :
            self.fig.canvas.widgetlock.release(self._curr_lasso)
            del self._curr_lasso
            self._curr_lasso = None
        print "Selection Mode"

    def multi_select_mode(self) :
        # Set mode to multi-select
        self._mode = 'S'
        print "Multi-Selection Mode"

    def outline_mode(self) :
        # set mode to "outline mode"
        self._mode = 'o'
        print "Outline Mode"

    def toggle_assoc_mode(self) :
        self._assoc_mode = ((self._assoc_mode + 1) %
                            len(TM_ControlSys._assoc_mode_list))
        print "Association Action:", \
              TM_ControlSys._assoc_mode_list[self._assoc_mode]

    def show_all_features(self) :
        # Show/Hide all identified features across time
        self._show_features = (not self._show_features)
        print "Show features:", self._show_features
        self.update_frame(hold_recluster=True)

    def toggle_save(self) :
        # Toogle save
        self._do_save = (not self._do_save)
        print "Do Save:", self._do_save

    #elif event.key == 'V' :
    #    # Save features to memory NOW!
    #    print "Converting to track and volume objects, NOW!"
    #    self.save_features()

    def print_menu(self) :
        # Print helpful Menu
        print dedent("""
            Track Maker
            ===========

            Key         Action
            ------      -----------------------------""")

        for key, act in self.keymap.iteritems() :
            print "%11s %s" % (key, act['help'])

        print dedent("""
            Current Values
            --------------
                Current Frame: %d of %d
                Current Mode: %s
                Association Method: %s
                Do save upon figure close: %s
                Show all features: %s
            """ % (self.rd.frameIndex + 1,
                   len(self.rd.radarData), self._mode,
                   TM_ControlSys._assoc_mode_list[self._assoc_mode],
                   self._do_save, self._show_features))


    def _clear_frame(self, frame=None) :
        if frame is None :
            frame = self.rd.frameIndex

        # Toggle the visibility
        visibleToggle = (not self._visible[frame])
        self._visible[frame] = visibleToggle


        # Set the frame's features to the visible boolean
        for feat in self.state._features[frame] :
            feat.set_visible(visibleToggle)

    def get_clusters(self) :
        dataset = self.rd.radarData.curr()
        data = dataset['vals'][0]

        flat_data = data[data >= -40]

        clustLabels = np.empty(data.shape, dtype=int)
        clustLabels[:] = -1

        if np.nanmin(flat_data) == np.nanmax(flat_data) :
            # can't cluster data with no change
            return clustLabels, 0

        bad_data = (np.isnan(data) | (data <= 0.0))

        bins = np.linspace(np.nanmin(flat_data),
                           np.nanmax(flat_data), 2**8)
        data_digitized = np.digitize(data.flat, bins)
        data_digitized.shape = data.shape
        data_digitized = data_digitized.astype('uint8')

        markers = np.zeros(data.shape, dtype=int)

        for index, feat in enumerate(self.state._features[self.rd.frameIndex]) :
            if 'contour' in feat.objects :
                contr = feat.objects['contour']
                res = points_inside_poly(zip(self.rd.xs.flat, self.rd.ys.flat),
                                         contr.get_xy())
                res.shape = self.rd.xs.shape
                markers[res] = index + 1

            # No contour available? Then fall back to just a point
            elif 'center' in feat.objects :
                cent = feat.objects['center']
                gridx, gridy = self._xy2grid(cent.center[0], cent.center[1])
                markers[gridy, gridx] = index + 1

            # TODO: work from an ellipse, if it exists?
            else :
                raise ValueError("Empty feature?")


        markers[bad_data] = -1
        ndimg.watershed_ift(data_digitized, markers, output=clustLabels)
        clustCnt = len(self.state._features[self.rd.frameIndex])

        cents = ndimg.center_of_mass(data**2, clustLabels,
                                     range(1, clustCnt + 1))
        ellipses = FitEllipses(clustLabels, range(1, clustCnt + 1),
                               self.rd.xs, self.rd.ys)

        for center, ellip, feat in zip(cents, ellipses,
                                    self.state._features[self.rd.frameIndex]) :
            # Remove any other objects that may exist before adding
            # new objects to the feature.
            feat.cleanup(['contour'])

            if ellip is None :
                continue

            cent_indx = tuple(np.floor(center).astype(int).tolist())
            # TODO: clean this up!
            newPoint = self.state._new_point(self.rd.xs[cent_indx],
                                             self.rd.ys[cent_indx])
            self.ax.add_artist(ellip)
            self.ax.add_artist(newPoint)

            feat.objects['center'] = newPoint
            feat.objects['ellip'] = ellip

            if feat.track is not None :
                feat.track.update_frame(self.rd.frameIndex)

        #print "clust count:", clustCnt
        return clustLabels, clustCnt

    def _xy2grid(self, x, y) :
        return (self.rd.xs[0].searchsorted(x),
                self.rd.ys[:, 0].searchsorted(y))

    def update_frame(self, lastFrame=None,
                           force_recluster=False, hold_recluster=False) :
        """
        Redraw the features in the display.  Calculate clusters if needed.

        *lastFrame*         int (None)
            If specified, make that frame's features invisible.

        *force_recluster*   boolean (False)
            If True, do a recluster, even if it seems like it isn't needed.
            Can be over-ridden by *hold_recluster*.

        *hold_recluster*    boolean (False)
            If True, then don't do a recluster, even if needed or
            *force_recluster* is True.
        """
        if lastFrame is not None :
            # Make sure there is an entry in self._visible so
            # that it can be set to False in _clear_frame()
            self._visible[lastFrame] = True
            self._clear_frame(lastFrame)

        if force_recluster or any([('center' not in feat.objects) for
                            feat in self.state._features[self.rd.frameIndex]]) :
            if not hold_recluster :
                clustLabels, clustCnt = self.get_clusters()

        self._visible[self.rd.frameIndex] = True
        # Set features for this frame to visible
        for feat in self.state._features[self.rd.frameIndex] :
            feat.set_visible(True)
            # Return alpha back to normal
            feat.set_alpha(1.0)
            # Put it on top
            feat.set_zorder(len(self.rd.radarData))

        # Show the other features
        if self._show_features :
            # How much alpha should change for each frame from frameIndex
            # The closer to self.frameIndex, the more opaque
            alphaIncrem = 1.0 / len(self.rd.radarData)
            for frameIndex, features in self.state._features.iteritems() :
                if frameIndex == self.rd.frameIndex :
                    continue

                framesFrom = np.abs(self.rd.frameIndex - frameIndex)
                timeAlpha = ((1.0 - (framesFrom * alphaIncrem)) **
                             (1.0/self._alphaScale))
                zorder = len(self.rd.radarData) - framesFrom
                for feat in features :
                    feat.set_visible(True)
                    feat.set_alpha(timeAlpha)
                    feat.set_zorder(zorder)
        else :
            # Make sure that these items are hidden
            for frameIndex, features in self.state._features.iteritems() :
                if frameIndex != self.rd.frameIndex :
                    for feat in features :
                        feat.set_visible(False)
                        # Return alpha to normal
                        feat.set_alpha(1.0)

    def step_back(self) :
        lastFrame = self.rd.frameIndex
        BaseControlSys.step_back(self)

        if lastFrame != self.rd.frameIndex :
            self.update_frame(lastFrame, hold_recluster=True)

    def step_forward(self) :
        lastFrame = self.rd.frameIndex
        BaseControlSys.step_forward(self)

        if lastFrame != self.rd.frameIndex :
            self.update_frame(lastFrame, hold_recluster=True)



def AnalyzeRadar(volume, tracks, falarms, polygons, radarFiles,
                 useOldTransform=False) :

    minLat, minLon, maxLat, maxLon, volTimes = ConsistentDomain(radarFiles)

    radarData = RadarCache(radarFiles, 4)
    state = StateManager(volTimes)
    state.load_features(tracks, falarms, volume, polygons)

    data = radarData.curr()
    radarName = data['station']
    if radarName == 'NWRT' :
        radarName = 'PAR'
    radarSite = radarsites.ByName(radarName)[0]

    lons, lats = np.meshgrid(data['lons'], data['lats'])

    cent_lon, cent_lat = (((minLon + maxLon) / 2.0,
                           (minLat + maxLat) / 2.0)
                         if useOldTransform else
                          (radarSite['LON'], radarSite['LAT']))

    xs, ys = LonLat2Cart(cent_lon, cent_lat,
                         lons, lats)

    fig = plt.figure()
    ax = fig.gca()
    rd = RadarDisplay(ax, radarData, xs, ys)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')

    cs = TM_ControlSys(fig, ax, rd, state)

    return radarData, state, rd, cs

def main(args) :
    dirName = os.path.join(args.path, args.scenario)

    paramFile = os.path.join(dirName, "simParams.conf")

    if os.path.exists(paramFile) :
        sceneParams = ParamUtils.ReadSimulationParams(paramFile)
    else :
        sceneParams = dict()
        sceneParams.update(ParamUtils.simDefaults)
        sceneParams.update(ParamUtils.trackerDefaults)

        sceneParams.pop('seed')
        sceneParams.pop('totalTracks')
        sceneParams.pop('endTrackProb')
        sceneParams.pop('simConfFile')

        sceneParams['simName'] = args.scenario



    origTrackFile = os.path.join(dirName, sceneParams['simTrackFile'])
    filtTrackFile = os.path.join(dirName, sceneParams['noisyTrackFile'])
    volumeFile = os.path.join(dirName, sceneParams['inputDataFile'])
    volumeLoc = os.path.dirname(volumeFile)


    tracks, falarms = [], []

    if os.path.exists(origTrackFile) :
        tracks, falarms = TrackFileUtils.ReadTracks(origTrackFile)
        

    volume = dict(frameCnt=0,
                  corner_filestem=sceneParams['corner_file'],
                  volume_data=[])
    if os.path.exists(volumeFile) :
        # dictionary with "volume_data", "frameCnt", "corner_filestem"
        #    and "volume_data" contains a list of dictionaries with
        #    "volTime", "frameNum", "stormCells" where "stormCells" is
        #    a numpy array of dtype corner_dtype
        volume = TrackFileUtils.ReadCorners(volumeFile, volumeLoc)
        sceneParams['corner_file'] = volume['corner_filestem']

    # TODO: Temporary!
    # I would rather not do pickling, but I am pressed for time.
    # Plus, if I intend to make it a part of the ZigZag system,
    # then the name should be stored in the parameters file.
    polygonfile = os.path.join(dirName, "polygons.foo")
    polygons = {}

    if os.path.exists(polygonfile) :
        polygons = load(open(polygonfile, 'rb'))

    radarData, state, rd, cs = AnalyzeRadar(volume, tracks, falarms, polygons,
                                            args.input_files,
                                            args.useOldCoords)

    # Activate the display.
    plt.show()

    if cs.do_save_results() :
        # Create the directory if it doesn't exist already
        if not os.path.exists(dirName) :
            makedirs(dirName)

        tracks, falarms, volume, polygons = state.save_features(rd.xs, rd.ys)
        volume['corner_filestem'] = sceneParams['corner_file']

        # Final save
        SaveState(paramFile, sceneParams, volumeFile, volume,
                  origTrackFile, filtTrackFile, tracks, falarms,
                  polygonfile, polygons)

def SortVolume(volumes) :
    frames = [aVol['frameNum'] for aVol in volumes]
    args = np.argsort(frames)
    new_volumes = [None] * (max(frames) - min(frames) + 1)
    for index, arg in enumerate(args) :
        # TODO: Fill in missing frames
        new_volumes[index] = volumes[arg]

    return new_volumes

def SaveState(paramFile, params, volumeFile, volume,
              origTrackFile, filtTrackFile, tracks, falarms,
              polygonfile, polygons) :
    # Do I need to update the Params?
    volume['volume_data'] = SortVolume(volume['volume_data'])
    TrackFileUtils.SaveCorners(volumeFile, volume['corner_filestem'],
                               volume['volume_data'],
                               path=os.path.dirname(volumeFile))
    ParamUtils.SaveConfigFile(paramFile, params)
    TrackFileUtils.SaveTracks(origTrackFile, tracks, falarms)
    TrackFileUtils.SaveTracks(filtTrackFile, tracks, falarms)

    # TODO: Save polygons in a better format
    dump(polygons, open(polygonfile, 'wb'))


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
    parser.add_argument("--old_coords", dest='useOldCoords',
                        action='store_true',
                        help='Use old coordinate transform')

    args = parser.parse_args()


    if args.scenario is None :
        parser.error("Missing SCENARIO!")

    if args.input_files is None or len(args.input_files) == 0 :
        parser.error("Missing input radar files!")

    args.input_files.sort()
    main(args)

