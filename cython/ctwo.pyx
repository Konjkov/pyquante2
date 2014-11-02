cimport cints
cimport chgp
import ctypes
cimport numpy as np

STUFF = "Hi" # define init??

def ERI(a,b,c,d):
    if d.contracted:
        return sum(cd*ERI(pd,c,a,b) for (cd,pd) in d)
    return cints.coulomb_repulsion(
           a.origin[0],a.origin[1],a.origin[2],a.norm,
           a.powers[0],a.powers[1],a.powers[2],a.exponent,
           b.origin[0],b.origin[1],b.origin[2],b.norm,
           b.powers[0],b.powers[1],b.powers[2],b.exponent,
           c.origin[0],c.origin[1],c.origin[2],c.norm,
           c.powers[0],c.powers[1],c.powers[2],c.exponent,
           d.origin[0],d.origin[1],d.origin[2],d.norm,
           d.powers[0],d.powers[1],d.powers[2],d.exponent)

def ERI_hgp(a,b,c,d):
    if a.contracted and b.contracted and c.contracted and d.contracted:
        return chgp.contr_hrr(len(a.coefs),a.origin[0],a.origin[1],a.origin[2],<double *>np.PyArray_DATA(a.pnorms),
                    a.powers[0],a.powers[1],a.powers[2],<double *>np.PyArray_DATA(a.pexps),<double *>np.PyArray_DATA(a.coefs),
                    len(b.coefs),b.origin[0],b.origin[1],b.origin[2],<double *>np.PyArray_DATA(b.pnorms),
                    b.powers[0],b.powers[1],b.powers[2],<double *>np.PyArray_DATA(b.pexps),<double *>np.PyArray_DATA(b.coefs),
                    len(c.coefs),c.origin[0],c.origin[1],c.origin[2],<double *>np.PyArray_DATA(c.pnorms),
                    c.powers[0],c.powers[1],c.powers[2],<double *>np.PyArray_DATA(c.pexps),<double *>np.PyArray_DATA(c.coefs),
                    len(d.coefs),d.origin[0],d.origin[1],d.origin[2],<double *>np.PyArray_DATA(d.pnorms),
                    d.powers[0],d.powers[1],d.powers[2],<double *>np.PyArray_DATA(d.pexps),<double *>np.PyArray_DATA(d.coefs))
    if d.contracted:
        return sum(cd*ERI_hgp(pd,c,a,b) for (cd,pd) in d)
    return chgp.hrr(
        a.origin[0],a.origin[1],a.origin[2],a.norm,
        a.powers[0],a.powers[1],a.powers[2],a.exponent,
        b.origin[0],b.origin[1],b.origin[2],b.norm,
        b.powers[0],b.powers[1],b.powers[2],b.exponent,
        c.origin[0],c.origin[1],c.origin[2],c.norm,
        c.powers[0],c.powers[1],c.powers[2],c.exponent,
        d.origin[0],d.origin[1],d.origin[2],d.norm,
        d.powers[0],d.powers[1],d.powers[2],d.exponent)

# The following are only for debugging and can be deleted after ERI_hgp works:
def vrr(double xa,double ya,double za,double norma,int la,int ma,int na,double alphaa,
        double xb,double yb,double zb,double normb,double alphab,
        double xc,double yc,double zc,double normc,int lc,int mc,int nc,double alphac,
        double xd,double yd,double zd,double normd,double alphad,int m):
    return chgp.vrr(xa,ya,za,norma,la,ma,na,alphaa,
                    xb,yb,zb,normb,alphab,
                    xc,yc,zc,normc,lc,mc,nc,alphac,
                    xd,yd,zd,normd,alphad,m)

def vrr_nonrecursive(double xa,double ya,double za,double norma,int la,int ma,int na,double alphaa,
        double xb,double yb,double zb,double normb,double alphab,
        double xc,double yc,double zc,double normc,int lc,int mc,int nc,double alphac,
        double xd,double yd,double zd,double normd,double alphad,int m):
        return chgp.vrr_nonrecursive(xa,ya,za,norma,la,ma,na,alphaa,
                                     xb,yb,zb,normb,alphab,
                                     xc,yc,zc,normc,lc,mc,nc,alphac,
                                     xd,yd,zd,normd,alphad,m)
def vrr_recursive(double xa,double ya,double za,double norma,int la,int ma,int na,double alphaa,
        double xb,double yb,double zb,double normb,double alphab,
        double xc,double yc,double zc,double normc,int lc,int mc,int nc,double alphac,
        double xd,double yd,double zd,double normd,double alphad,int m):
        return chgp.vrr_recursive(xa,ya,za,norma,la,ma,na,alphaa,
                                  xb,yb,zb,normb,alphab,
                                  xc,yc,zc,normc,lc,mc,nc,alphac,
                                  xd,yd,zd,normd,alphad,m)
