#-------------------------------------------------------------------------------
# Name:        max_inscribing_rectangle.py
# Purpose:     get maximum inscribing rectangle for polygon
#
# Author:      Xander
#
# Created:     26-08-2016
#-------------------------------------------------------------------------------

def main():
    import arcpy
    arcpy.env.overwriteOutput = True

    fc = r'C:\GeoNet\MaximumAreaInscribedRectangle\data.gdb\polygon'
    fc_out = r'C:\GeoNet\MaximumAreaInscribedRectangle\data.gdb\rectangles01'

    # get polygon
    polygon = arcpy.da.SearchCursor(fc, ('SHAPE@')).next()[0]
    sr = polygon.spatialReference

    # determine outline, extent and diagonal
    polyline = polygon.boundary()
    extent = polygon.extent
    diagonal_length = getPythagoras(extent.width, extent.height)

    lst_rectangles = []
    # first loop for start point of first side
    for i in range(10):
        perc_p1 = float(i) / 10
        pntg1 = polyline.positionAlongLine(perc_p1, True)

        # second loop end point of first side
        for j in range(10):
            frac = float(j) / 20
            perc_p2 = perc_p1 + 0.25 + frac
            if perc_p2 > 1:
                perc_p2 -= 1
            pntg2 = polyline.positionAlongLine(perc_p2, True)

            # process as side
            # - get angle between points
            angle = getAngle(pntg1, pntg2)

            # - create perpendicual lines at start and end
            pntg1a = pntg1.pointFromAngleAndDistance(angle + 90, diagonal_length, 'PLANAR')
            pntg1b = pntg1.pointFromAngleAndDistance(angle - 90, diagonal_length, 'PLANAR')
            line1 = createStraightLine(pntg1a, pntg1b)

            pntg2a = pntg2.pointFromAngleAndDistance(angle + 90, diagonal_length, 'PLANAR')
            pntg2b = pntg2.pointFromAngleAndDistance(angle - 90, diagonal_length, 'PLANAR')
            line2 = createStraightLine(pntg2a, pntg2b)

            # - intersect by polygon (asume single parts)
            line1cut = checkInvertedLine(line1.intersect(polygon, 2), pntg1)
            line2cut = checkInvertedLine(line2.intersect(polygon, 2), pntg2)

            # - determine shortest, cut other by length shortest
            length1 = line1cut.length
            length2 = line2cut.length
            if length2 < length1:
                # cut line 1
                line1cut = line1cut.segmentAlongLine(0, length2, False)
            else:
                # cut line 2
                line2cut = line2cut.segmentAlongLine(0, length1, False)

            # - form rectangle and add to list
            rectangle = createRectanglePolygon(line1cut, line2cut)
            lst_rectangles.append(rectangle)

            # process point pair as diagonal of rectangle?

    # write output
    arcpy.CopyFeatures_management(lst_rectangles, fc_out)


def createRectanglePolygon(line1, line2):
    sr = line1.spatialReference
    pnt1a = line1.firstPoint
    pnt1b = line1.lastPoint
    pnt2a = line2.firstPoint
    pnt2b = line2.lastPoint
    return arcpy.Polygon(arcpy.Array([pnt1a, pnt1b,
                                      pnt2b, pnt2a,
                                      pnt1a]), sr)


def checkInvertedLine(line, pntg):
    # start of line should be near pntg
    sr = line.spatialReference
    pnt_start = line.firstPoint
    pnt_end = line.lastPoint
    dist_start = getPythagoras(pnt_start.X-pntg.firstPoint.X,
                               pnt_start.Y-pntg.firstPoint.Y)
    dist_end = getPythagoras(pnt_end.X-pntg.firstPoint.X,
                               pnt_end.Y-pntg.firstPoint.Y)
    if dist_end < dist_start:
        # flip
        return arcpy.Polyline(arcpy.Array([pnt_end, pnt_start]), sr)
    else:
        return line

def createStraightLine(pntg1, pntg2):
    sr = pntg1.spatialReference
    return arcpy.Polyline(arcpy.Array([pntg1.firstPoint, pntg2.firstPoint]), sr)

def getAngle(pntg1, pntg2):
    '''Define angle between two points'''
    return pntg1.angleAndDistanceTo(pntg2, 'PLANAR')[0]

def getPythagoras(w, h):
    import math
    return math.hypot(w, h)


if __name__ == '__main__':
    main()