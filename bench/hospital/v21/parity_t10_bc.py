"""parity_t10_bc: generic barcode locator (parallel thin dark bars, grouped by spacing): bar groups in a picture.  Fixed morphology, no template of any card."""
import numpy as np, cv2
def barcode_groups(rgb, min_bars=7):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); loc=cv2.blur(g,(51,51))
    dark=((g.astype(int)<loc.astype(int)-40)&(g<120)).astype(np.uint8)
    n,lab,st,cen=cv2.connectedComponentsWithStats(dark,4)
    bars=[]
    for i in range(1,n):
        x,y,w,h,a=st[i]; long_,short=max(w,h),min(w,h)
        if long_>=18 and long_<=160 and short<=9 and a>=0.55*w*h and long_/max(short,1)>=3.5:
            bars.append(dict(c=cen[i],ori='v' if h>w else 'h',L=long_,box=(x,y,w,h)))
    out=[]
    for ori in ('v','h'):
        bs=[b for b in bars if b['ori']==ori]
        used=set()
        for i,b in enumerate(bs):
            if i in used: continue
            grp=[i]; used.add(i); stack=[i]
            while stack:
                k=stack.pop()
                for j,c in enumerate(bs):
                    if j in used: continue
                    d=bs[k]['c']-c['c']
                    across=abs(d[0]) if ori=='v' else abs(d[1]); along=abs(d[1]) if ori=='v' else abs(d[0])
                    if across<=14 and along<=0.5*max(bs[k]['L'],c['L']): used.add(j); grp.append(j); stack.append(j)
            if len(grp)>=min_bars:
                xs=[bs[k]['box'] for k in grp]; x0=min(b[0] for b in xs); y0=min(b[1] for b in xs); x1=max(b[0]+b[2] for b in xs); y1=max(b[1]+b[3] for b in xs)
                out.append(dict(centre=np.array([(x0+x1)/2,(y0+y1)/2]),box=(x0,y0,x1-x0,y1-y0),n=len(grp),ori=ori))
    return out


def plate_centre(rgb, cx, cy, r=60):
    """centre (px) of the bright rectangle (label plate: white fill, dark print) that holds the point (cx, cy): the printed text is not centred on its plate, the plate's own outline is.
    Flood fill of the bright pixels (V > 150, low saturation) from the nearest bright pixel to the seed, bounding box centre.  Returns (px, py, w, h) or None."""
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); bright = ((hsv[..., 2] > 150) & (hsv[..., 1] < 70)).astype(np.uint8)
    h, w = bright.shape; x0, x1, y0, y1 = max(0, int(cx) - r), min(w, int(cx) + r), max(0, int(cy) - r), min(h, int(cy) + r)
    ys, xs = np.nonzero(bright[y0:y1, x0:x1])
    if len(xs) == 0: return None
    k = int(np.argmin((xs + x0 - cx) ** 2 + (ys + y0 - cy) ** 2)); seed = (int(xs[k] + x0), int(ys[k] + y0))
    m = np.zeros((h + 2, w + 2), np.uint8); mask = bright.copy()
    cv2.floodFill(mask, m, seed, 2, flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)) if False else None
    n, lab, st, cen = cv2.connectedComponentsWithStats(bright, 4)
    i = int(lab[seed[1], seed[0]])
    if i == 0: return None
    x, y, ww, hh, a = st[i]
    return float(x + ww / 2), float(y + hh / 2), int(ww), int(hh)
