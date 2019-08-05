/*  Copyright (c) 2013, Robert Wang, email: robertwgh (at) gmail.com
    All rights reserved. https://github.com/robertwgh/ezSIFT

    Description: Detect keypoints and extract descriptors from an input image.

    Revision history:
        September 15th, 2013: initial version.
        July 2nd, 2018: code refactor.
*/

#include <iostream>
#include <list>

#include<typeinfo>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define EZSIFT_H

#include <vector>

#define IMAGE_UTILITY_H

#define COMMOM_H

#include <math.h>
#include <stdarg.h>
#include <stdio.h>

#include <cassert>
#include <cstring>
#include <ctype.h>

#define TIMER_H

#include <time.h>

#define __GUTIL_VECTOR_H__

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#include <math.h>

/* ========================================================== */
/* Zero out a 2D vector */

#define VEC_ZERO_2(a)                                                          \
                                                                               \
    {                                                                          \
        (a)[0] = (a)[1] = 0.0;                                                 \
    }

/* ========================================================== */
/* Zero out a 3D vector */

#define VEC_ZERO(a)                                                            \
                                                                               \
    {                                                                          \
        (a)[0] = (a)[1] = (a)[2] = 0.0;                                        \
    }

/* ========================================================== */
/* Zero out a 4D vector */

#define VEC_ZERO_4(a)                                                          \
                                                                               \
    {                                                                          \
        (a)[0] = (a)[1] = (a)[2] = (a)[3] = 0.0;                               \
    }

/* ========================================================== */
/* Vector copy */

#define VEC_COPY_2(b, a)                                                       \
                                                                               \
    {                                                                          \
        (b)[0] = (a)[0];                                                       \
        (b)[1] = (a)[1];                                                       \
    }

/* ========================================================== */
/* Copy 3D vector */

#define VEC_COPY(b, a)                                                         \
                                                                               \
    {                                                                          \
        (b)[0] = (a)[0];                                                       \
        (b)[1] = (a)[1];                                                       \
        (b)[2] = (a)[2];                                                       \
    }

/* ========================================================== */
/* Copy 4D vector */

#define VEC_COPY_4(b, a)                                                       \
                                                                               \
    {                                                                          \
        (b)[0] = (a)[0];                                                       \
        (b)[1] = (a)[1];                                                       \
        (b)[2] = (a)[2];                                                       \
        (b)[3] = (a)[3];                                                       \
    }

/* ========================================================== */
/* Vector difference */

#define VEC_DIFF_2(v21, v2, v1)                                                \
                                                                               \
    {                                                                          \
        (v21)[0] = (v2)[0] - (v1)[0];                                          \
        (v21)[1] = (v2)[1] - (v1)[1];                                          \
    }

/* ========================================================== */
/* Vector difference */

#define VEC_DIFF(v21, v2, v1)                                                  \
                                                                               \
    {                                                                          \
        (v21)[0] = (v2)[0] - (v1)[0];                                          \
        (v21)[1] = (v2)[1] - (v1)[1];                                          \
        (v21)[2] = (v2)[2] - (v1)[2];                                          \
    }

/* ========================================================== */
/* Vector difference */

#define VEC_DIFF_4(v21, v2, v1)                                                \
                                                                               \
    {                                                                          \
        (v21)[0] = (v2)[0] - (v1)[0];                                          \
        (v21)[1] = (v2)[1] - (v1)[1];                                          \
        (v21)[2] = (v2)[2] - (v1)[2];                                          \
        (v21)[3] = (v2)[3] - (v1)[3];                                          \
    }

/* ========================================================== */
/* Vector sum */

#define VEC_SUM_2(v21, v2, v1)                                                 \
                                                                               \
    {                                                                          \
        (v21)[0] = (v2)[0] + (v1)[0];                                          \
        (v21)[1] = (v2)[1] + (v1)[1];                                          \
    }

/* ========================================================== */
/* Vector sum */

#define VEC_SUM(v21, v2, v1)                                                   \
                                                                               \
    {                                                                          \
        (v21)[0] = (v2)[0] + (v1)[0];                                          \
        (v21)[1] = (v2)[1] + (v1)[1];                                          \
        (v21)[2] = (v2)[2] + (v1)[2];                                          \
    }

/* ========================================================== */
/* Vector sum */

#define VEC_SUM_4(v21, v2, v1)                                                 \
                                                                               \
    {                                                                          \
        (v21)[0] = (v2)[0] + (v1)[0];                                          \
        (v21)[1] = (v2)[1] + (v1)[1];                                          \
        (v21)[2] = (v2)[2] + (v1)[2];                                          \
        (v21)[3] = (v2)[3] + (v1)[3];                                          \
    }

/* ========================================================== */
/* scalar times vector */

#define VEC_SCALE_2(c, a, b)                                                   \
                                                                               \
    {                                                                          \
        (c)[0] = (a) * (b)[0];                                                 \
        (c)[1] = (a) * (b)[1];                                                 \
    }

/* ========================================================== */
/* scalar times vector */

#define VEC_SCALE(c, a, b)                                                     \
                                                                               \
    {                                                                          \
        (c)[0] = (a) * (b)[0];                                                 \
        (c)[1] = (a) * (b)[1];                                                 \
        (c)[2] = (a) * (b)[2];                                                 \
    }

/* ========================================================== */
/* scalar times vector */

#define VEC_SCALE_4(c, a, b)                                                   \
                                                                               \
    {                                                                          \
        (c)[0] = (a) * (b)[0];                                                 \
        (c)[1] = (a) * (b)[1];                                                 \
        (c)[2] = (a) * (b)[2];                                                 \
        (c)[3] = (a) * (b)[3];                                                 \
    }

/* ========================================================== */
/* accumulate scaled vector */

#define VEC_ACCUM_2(c, a, b)                                                   \
                                                                               \
    {                                                                          \
        (c)[0] += (a) * (b)[0];                                                \
        (c)[1] += (a) * (b)[1];                                                \
    }

/* ========================================================== */
/* accumulate scaled vector */

#define VEC_ACCUM(c, a, b)                                                     \
                                                                               \
    {                                                                          \
        (c)[0] += (a) * (b)[0];                                                \
        (c)[1] += (a) * (b)[1];                                                \
        (c)[2] += (a) * (b)[2];                                                \
    }

/* ========================================================== */
/* accumulate scaled vector */

#define VEC_ACCUM_4(c, a, b)                                                   \
                                                                               \
    {                                                                          \
        (c)[0] += (a) * (b)[0];                                                \
        (c)[1] += (a) * (b)[1];                                                \
        (c)[2] += (a) * (b)[2];                                                \
        (c)[3] += (a) * (b)[3];                                                \
    }

/* ========================================================== */
/* Vector dot product */

#define VEC_DOT_PRODUCT_2(c, a, b)                                             \
                                                                               \
    {                                                                          \
        c = (a)[0] * (b)[0] + (a)[1] * (b)[1];                                 \
    }

/* ========================================================== */
/* Vector dot product */

#define VEC_DOT_PRODUCT_3(c, a, b)                                             \
                                                                               \
    {                                                                          \
        c = (a)[0] * (b)[0] + (a)[1] * (b)[1] + (a)[2] * (b)[2];               \
    }

/* ========================================================== */
/* Vector dot product */

#define VEC_DOT_PRODUCT_4(c, a, b)                                             \
                                                                               \
    {                                                                          \
        c = (a)[0] * (b)[0] + (a)[1] * (b)[1] + (a)[2] * (b)[2] +              \
            (a)[3] * (b)[3];                                                   \
    }

/* ========================================================== */
/* vector impact parameter (squared) */

#define VEC_IMPACT_SQ(bsq, direction, position)                                \
                                                                               \
    {                                                                          \
        gleDouble vlen, llel;                                                  \
        VEC_DOT_PRODUCT(vlen, position, position);                             \
        VEC_DOT_PRODUCT(llel, direction, position);                            \
        bsq = vlen - llel * llel;                                              \
    }

/* ========================================================== */
/* vector impact parameter */

#define VEC_IMPACT(bsq, direction, position)                                   \
                                                                               \
    {                                                                          \
        VEC_IMPACT_SQ(bsq, direction, position);                               \
        bsq = sqrt(bsq);                                                       \
    }

/* ========================================================== */
/* Vector length */

#define VEC_LENGTH_2(vlen, a)                                                  \
                                                                               \
    {                                                                          \
        vlen = a[0] * a[0] + a[1] * a[1];                                      \
        vlen = sqrt(vlen);                                                     \
    }

/* ========================================================== */
/* Vector length */

#define VEC_LENGTH(vlen, a)                                                    \
                                                                               \
    {                                                                          \
        vlen = (a)[0] * (a)[0] + (a)[1] * (a)[1];                              \
        vlen += (a)[2] * (a)[2];                                               \
        vlen = sqrt(vlen);                                                     \
    }

/* ========================================================== */
/* Vector length */

#define VEC_LENGTH_4(vlen, a)                                                  \
                                                                               \
    {                                                                          \
        vlen = (a)[0] * (a)[0] + (a)[1] * (a)[1];                              \
        vlen += (a)[2] * (a)[2];                                               \
        vlen += (a)[3] * (a)[3];                                               \
        vlen = sqrt(vlen);                                                     \
    }

/* ========================================================== */
/* distance between two points */

#define VEC_DISTANCE(vlen, va, vb)                                             \
                                                                               \
    {                                                                          \
        gleDouble tmp[4];                                                      \
        VEC_DIFF(tmp, vb, va);                                                 \
        VEC_LENGTH(vlen, tmp);                                                 \
    }

/* ========================================================== */
/* Vector length */

#define VEC_CONJUGATE_LENGTH(vlen, a)                                          \
                                                                               \
    {                                                                          \
        vlen = 1.0 - a[0] * a[0] - a[1] * a[1] - a[2] * a[2];                  \
        vlen = sqrt(vlen);                                                     \
    }

/* ========================================================== */
/* Vector length */

#define VEC_NORMALIZE(a)                                                       \
                                                                               \
    {                                                                          \
        double vlen;                                                           \
        VEC_LENGTH(vlen, a);                                                   \
        if (vlen != 0.0) {                                                     \
            vlen = 1.0 / vlen;                                                 \
            a[0] *= vlen;                                                      \
            a[1] *= vlen;                                                      \
            a[2] *= vlen;                                                      \
        }                                                                      \
    }

/* ========================================================== */
/* Vector length */

#define VEC_RENORMALIZE(a, newlen)                                             \
                                                                               \
    {                                                                          \
        double vlen;                                                           \
        VEC_LENGTH(vlen, a);                                                   \
        if (vlen != 0.0) {                                                     \
            vlen = newlen / vlen;                                              \
            a[0] *= vlen;                                                      \
            a[1] *= vlen;                                                      \
            a[2] *= vlen;                                                      \
        }                                                                      \
    }

/* ========================================================== */
/* 3D Vector cross product yeilding vector */

#define VEC_CROSS_PRODUCT(c, a, b)                                             \
                                                                               \
    {                                                                          \
        c[0] = (a)[1] * (b)[2] - (a)[2] * (b)[1];                              \
        c[1] = (a)[2] * (b)[0] - (a)[0] * (b)[2];                              \
        c[2] = (a)[0] * (b)[1] - (a)[1] * (b)[0];                              \
    }

/* ========================================================== */
/* Vector perp -- assumes that n is of unit length
 * accepts vector v, subtracts out any component parallel to n */

#define VEC_PERP(vp, v, n)                                                     \
                                                                               \
    {                                                                          \
        double vdot;                                                           \
                                                                               \
        VEC_DOT_PRODUCT(vdot, v, n);                                           \
        vp[0] = (v)[0] - (vdot) * (n)[0];                                      \
        vp[1] = (v)[1] - (vdot) * (n)[1];                                      \
        vp[2] = (v)[2] - (vdot) * (n)[2];                                      \
    }

/* ========================================================== */
/* Vector parallel -- assumes that n is of unit length
 * accepts vector v, subtracts out any component perpendicular to n */

#define VEC_PARALLEL(vp, v, n)                                                 \
                                                                               \
    {                                                                          \
        double vdot;                                                           \
                                                                               \
        VEC_DOT_PRODUCT(vdot, v, n);                                           \
        vp[0] = (vdot) * (n)[0];                                               \
        vp[1] = (vdot) * (n)[1];                                               \
        vp[2] = (vdot) * (n)[2];                                               \
    }

/* ========================================================== */
/* Vector reflection -- assumes n is of unit length */
/* Takes vector v, reflects it against reflector n, and returns vr */

#define VEC_REFLECT(vr, v, n)                                                  \
                                                                               \
    {                                                                          \
        double vdot;                                                           \
                                                                               \
        VEC_DOT_PRODUCT(vdot, v, n);                                           \
        vr[0] = (v)[0] - 2.0 * (vdot) * (n)[0];                                \
        vr[1] = (v)[1] - 2.0 * (vdot) * (n)[1];                                \
        vr[2] = (v)[2] - 2.0 * (vdot) * (n)[2];                                \
    }

/* ========================================================== */
/* Vector blending */
/* Takes two vectors a, b, blends them together */

#define VEC_BLEND(vr, sa, a, sb, b)                                            \
                                                                               \
    {                                                                          \
                                                                               \
        vr[0] = (sa) * (a)[0] + (sb) * (b)[0];                                 \
        vr[1] = (sa) * (a)[1] + (sb) * (b)[1];                                 \
        vr[2] = (sa) * (a)[2] + (sb) * (b)[2];                                 \
    }

/* ========================================================== */
/* Vector print */

#define VEC_PRINT_2(a)                                                         \
                                                                               \
    {                                                                          \
        double vlen;                                                           \
        VEC_LENGTH_2(vlen, a);                                                 \
        printf(" a is %f %f length of a is %f \n", a[0], a[1], vlen);          \
    }

/* ========================================================== */
/* Vector print */

#define VEC_PRINT(a)                                                           \
                                                                               \
    {                                                                          \
        double vlen;                                                           \
        VEC_LENGTH(vlen, (a));                                                 \
        printf(" a is %f %f %f length of a is %f \n", (a)[0], (a)[1], (a)[2],  \
               vlen);                                                          \
    }

/* ========================================================== */
/* Vector print */

#define VEC_PRINT_4(a)                                                         \
                                                                               \
    {                                                                          \
        double vlen;                                                           \
        VEC_LENGTH_4(vlen, (a));                                               \
        printf(" a is %f %f %f %f length of a is %f \n", (a)[0], (a)[1],       \
               (a)[2], (a)[3], vlen);                                          \
    }

/* ========================================================== */
/* print matrix */

#define MAT_PRINT_4X4(mmm)                                                     \
    {                                                                          \
        int i, j;                                                              \
        printf("matrix mmm is \n");                                            \
        if (mmm == NULL) {                                                     \
            printf(" Null \n");                                                \
        }                                                                      \
        else {                                                                 \
            for (i = 0; i < 4; i++) {                                          \
                for (j = 0; j < 4; j++) {                                      \
                    printf("%f ", mmm[i][j]);                                  \
                }                                                              \
                printf(" \n");                                                 \
            }                                                                  \
        }                                                                      \
    }

/* ========================================================== */
/* print matrix */

#define MAT_PRINT_3X3(mmm)                                                     \
    {                                                                          \
        int i, j;                                                              \
        printf("matrix mmm is \n");                                            \
        if (mmm == NULL) {                                                     \
            printf(" Null \n");                                                \
        }                                                                      \
        else {                                                                 \
            for (i = 0; i < 3; i++) {                                          \
                for (j = 0; j < 3; j++) {                                      \
                    printf("%f ", mmm[i][j]);                                  \
                }                                                              \
                printf(" \n");                                                 \
            }                                                                  \
        }                                                                      \
    }

/* ========================================================== */
/* print matrix */

#define MAT_PRINT_2X3(mmm)                                                     \
    {                                                                          \
        int i, j;                                                              \
        printf("matrix mmm is \n");                                            \
        if (mmm == NULL) {                                                     \
            printf(" Null \n");                                                \
        }                                                                      \
        else {                                                                 \
            for (i = 0; i < 2; i++) {                                          \
                for (j = 0; j < 3; j++) {                                      \
                    printf("%f ", mmm[i][j]);                                  \
                }                                                              \
                printf(" \n");                                                 \
            }                                                                  \
        }                                                                      \
    }

/* ========================================================== */
/* initialize matrix */

#define IDENTIFY_MATRIX_3X3(m)                                                 \
                                                                               \
    {                                                                          \
        m[0][0] = 1.0;                                                         \
        m[0][1] = 0.0;                                                         \
        m[0][2] = 0.0;                                                         \
                                                                               \
        m[1][0] = 0.0;                                                         \
        m[1][1] = 1.0;                                                         \
        m[1][2] = 0.0;                                                         \
                                                                               \
        m[2][0] = 0.0;                                                         \
        m[2][1] = 0.0;                                                         \
        m[2][2] = 1.0;                                                         \
    }

/* ========================================================== */
/* initialize matrix */

#define IDENTIFY_MATRIX_4X4(m)                                                 \
                                                                               \
    {                                                                          \
        m[0][0] = 1.0;                                                         \
        m[0][1] = 0.0;                                                         \
        m[0][2] = 0.0;                                                         \
        m[0][3] = 0.0;                                                         \
                                                                               \
        m[1][0] = 0.0;                                                         \
        m[1][1] = 1.0;                                                         \
        m[1][2] = 0.0;                                                         \
        m[1][3] = 0.0;                                                         \
                                                                               \
        m[2][0] = 0.0;                                                         \
        m[2][1] = 0.0;                                                         \
        m[2][2] = 1.0;                                                         \
        m[2][3] = 0.0;                                                         \
                                                                               \
        m[3][0] = 0.0;                                                         \
        m[3][1] = 0.0;                                                         \
        m[3][2] = 0.0;                                                         \
        m[3][3] = 1.0;                                                         \
    }

/* ========================================================== */
/* matrix copy */

#define COPY_MATRIX_2X2(b, a)                                                  \
                                                                               \
    {                                                                          \
        b[0][0] = a[0][0];                                                     \
        b[0][1] = a[0][1];                                                     \
                                                                               \
        b[1][0] = a[1][0];                                                     \
        b[1][1] = a[1][1];                                                     \
    }

/* ========================================================== */
/* matrix copy */

#define COPY_MATRIX_2X3(b, a)                                                  \
                                                                               \
    {                                                                          \
        b[0][0] = a[0][0];                                                     \
        b[0][1] = a[0][1];                                                     \
        b[0][2] = a[0][2];                                                     \
                                                                               \
        b[1][0] = a[1][0];                                                     \
        b[1][1] = a[1][1];                                                     \
        b[1][2] = a[1][2];                                                     \
    }

/* ========================================================== */
/* matrix copy */

#define COPY_MATRIX_3X3(b, a)                                                  \
                                                                               \
    {                                                                          \
        b[0][0] = a[0][0];                                                     \
        b[0][1] = a[0][1];                                                     \
        b[0][2] = a[0][2];                                                     \
                                                                               \
        b[1][0] = a[1][0];                                                     \
        b[1][1] = a[1][1];                                                     \
        b[1][2] = a[1][2];                                                     \
                                                                               \
        b[2][0] = a[2][0];                                                     \
        b[2][1] = a[2][1];                                                     \
        b[2][2] = a[2][2];                                                     \
    }

/* ========================================================== */
/* matrix copy */

#define COPY_MATRIX_4X4(b, a)                                                  \
                                                                               \
    {                                                                          \
        b[0][0] = a[0][0];                                                     \
        b[0][1] = a[0][1];                                                     \
        b[0][2] = a[0][2];                                                     \
        b[0][3] = a[0][3];                                                     \
                                                                               \
        b[1][0] = a[1][0];                                                     \
        b[1][1] = a[1][1];                                                     \
        b[1][2] = a[1][2];                                                     \
        b[1][3] = a[1][3];                                                     \
                                                                               \
        b[2][0] = a[2][0];                                                     \
        b[2][1] = a[2][1];                                                     \
        b[2][2] = a[2][2];                                                     \
        b[2][3] = a[2][3];                                                     \
                                                                               \
        b[3][0] = a[3][0];                                                     \
        b[3][1] = a[3][1];                                                     \
        b[3][2] = a[3][2];                                                     \
        b[3][3] = a[3][3];                                                     \
    }

/* ========================================================== */
/* matrix transpose */

#define TRANSPOSE_MATRIX_2X2(b, a)                                             \
                                                                               \
    {                                                                          \
        b[0][0] = a[0][0];                                                     \
        b[0][1] = a[1][0];                                                     \
                                                                               \
        b[1][0] = a[0][1];                                                     \
        b[1][1] = a[1][1];                                                     \
    }

/* ========================================================== */
/* matrix transpose */

#define TRANSPOSE_MATRIX_3X3(b, a)                                             \
                                                                               \
    {                                                                          \
        b[0][0] = a[0][0];                                                     \
        b[0][1] = a[1][0];                                                     \
        b[0][2] = a[2][0];                                                     \
                                                                               \
        b[1][0] = a[0][1];                                                     \
        b[1][1] = a[1][1];                                                     \
        b[1][2] = a[2][1];                                                     \
                                                                               \
        b[2][0] = a[0][2];                                                     \
        b[2][1] = a[1][2];                                                     \
        b[2][2] = a[2][2];                                                     \
    }

/* ========================================================== */
/* matrix transpose */

#define TRANSPOSE_MATRIX_4X4(b, a)                                             \
                                                                               \
    {                                                                          \
        b[0][0] = a[0][0];                                                     \
        b[0][1] = a[1][0];                                                     \
        b[0][2] = a[2][0];                                                     \
        b[0][3] = a[3][0];                                                     \
                                                                               \
        b[1][0] = a[0][1];                                                     \
        b[1][1] = a[1][1];                                                     \
        b[1][2] = a[2][1];                                                     \
        b[1][3] = a[3][1];                                                     \
                                                                               \
        b[2][0] = a[0][2];                                                     \
        b[2][1] = a[1][2];                                                     \
        b[2][2] = a[2][2];                                                     \
        b[2][3] = a[3][2];                                                     \
                                                                               \
        b[3][0] = a[0][3];                                                     \
        b[3][1] = a[1][3];                                                     \
        b[3][2] = a[2][3];                                                     \
        b[3][3] = a[3][3];                                                     \
    }

/* ========================================================== */
/* multiply matrix by scalar */

#define SCALE_MATRIX_2X2(b, s, a)                                              \
                                                                               \
    {                                                                          \
        b[0][0] = (s)*a[0][0];                                                 \
        b[0][1] = (s)*a[0][1];                                                 \
                                                                               \
        b[1][0] = (s)*a[1][0];                                                 \
        b[1][1] = (s)*a[1][1];                                                 \
    }

/* ========================================================== */
/* multiply matrix by scalar */

#define SCALE_MATRIX_3X3(b, s, a)                                              \
                                                                               \
    {                                                                          \
        b[0][0] = (s)*a[0][0];                                                 \
        b[0][1] = (s)*a[0][1];                                                 \
        b[0][2] = (s)*a[0][2];                                                 \
                                                                               \
        b[1][0] = (s)*a[1][0];                                                 \
        b[1][1] = (s)*a[1][1];                                                 \
        b[1][2] = (s)*a[1][2];                                                 \
                                                                               \
        b[2][0] = (s)*a[2][0];                                                 \
        b[2][1] = (s)*a[2][1];                                                 \
        b[2][2] = (s)*a[2][2];                                                 \
    }

/* ========================================================== */
/* multiply matrix by scalar */

#define SCALE_MATRIX_4X4(b, s, a)                                              \
                                                                               \
    {                                                                          \
        b[0][0] = (s)*a[0][0];                                                 \
        b[0][1] = (s)*a[0][1];                                                 \
        b[0][2] = (s)*a[0][2];                                                 \
        b[0][3] = (s)*a[0][3];                                                 \
                                                                               \
        b[1][0] = (s)*a[1][0];                                                 \
        b[1][1] = (s)*a[1][1];                                                 \
        b[1][2] = (s)*a[1][2];                                                 \
        b[1][3] = (s)*a[1][3];                                                 \
                                                                               \
        b[2][0] = (s)*a[2][0];                                                 \
        b[2][1] = (s)*a[2][1];                                                 \
        b[2][2] = (s)*a[2][2];                                                 \
        b[2][3] = (s)*a[2][3];                                                 \
                                                                               \
        b[3][0] = s * a[3][0];                                                 \
        b[3][1] = s * a[3][1];                                                 \
        b[3][2] = s * a[3][2];                                                 \
        b[3][3] = s * a[3][3];                                                 \
    }

/* ========================================================== */
/* multiply matrix by scalar */

#define ACCUM_SCALE_MATRIX_2X2(b, s, a)                                        \
                                                                               \
    {                                                                          \
        b[0][0] += (s)*a[0][0];                                                \
        b[0][1] += (s)*a[0][1];                                                \
                                                                               \
        b[1][0] += (s)*a[1][0];                                                \
        b[1][1] += (s)*a[1][1];                                                \
    }

/* +========================================================== */
/* multiply matrix by scalar */

#define ACCUM_SCALE_MATRIX_3X3(b, s, a)                                        \
                                                                               \
    {                                                                          \
        b[0][0] += (s)*a[0][0];                                                \
        b[0][1] += (s)*a[0][1];                                                \
        b[0][2] += (s)*a[0][2];                                                \
                                                                               \
        b[1][0] += (s)*a[1][0];                                                \
        b[1][1] += (s)*a[1][1];                                                \
        b[1][2] += (s)*a[1][2];                                                \
                                                                               \
        b[2][0] += (s)*a[2][0];                                                \
        b[2][1] += (s)*a[2][1];                                                \
        b[2][2] += (s)*a[2][2];                                                \
    }

/* +========================================================== */
/* multiply matrix by scalar */

#define ACCUM_SCALE_MATRIX_4X4(b, s, a)                                        \
                                                                               \
    {                                                                          \
        b[0][0] += (s)*a[0][0];                                                \
        b[0][1] += (s)*a[0][1];                                                \
        b[0][2] += (s)*a[0][2];                                                \
        b[0][3] += (s)*a[0][3];                                                \
                                                                               \
        b[1][0] += (s)*a[1][0];                                                \
        b[1][1] += (s)*a[1][1];                                                \
        b[1][2] += (s)*a[1][2];                                                \
        b[1][3] += (s)*a[1][3];                                                \
                                                                               \
        b[2][0] += (s)*a[2][0];                                                \
        b[2][1] += (s)*a[2][1];                                                \
        b[2][2] += (s)*a[2][2];                                                \
        b[2][3] += (s)*a[2][3];                                                \
                                                                               \
        b[3][0] += (s)*a[3][0];                                                \
        b[3][1] += (s)*a[3][1];                                                \
        b[3][2] += (s)*a[3][2];                                                \
        b[3][3] += (s)*a[3][3];                                                \
    }

/* +========================================================== */
/* matrix product */
/* c[x][y] = a[x][0]*b[0][y]+a[x][1]*b[1][y]+a[x][2]*b[2][y]+a[x][3]*b[3][y];*/

#define MATRIX_PRODUCT_2X2(c, a, b)                                            \
                                                                               \
    {                                                                          \
        c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];                       \
        c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];                       \
                                                                               \
        c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];                       \
        c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];                       \
    }

/* ========================================================== */
/* matrix product */
/* c[x][y] = a[x][0]*b[0][y]+a[x][1]*b[1][y]+a[x][2]*b[2][y]+a[x][3]*b[3][y];*/

#define MATRIX_PRODUCT_3X3(c, a, b)                                            \
                                                                               \
    {                                                                          \
        c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];   \
        c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];   \
        c[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];   \
                                                                               \
        c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];   \
        c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];   \
        c[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];   \
                                                                               \
        c[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];   \
        c[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];   \
        c[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];   \
    }

/* ========================================================== */
/* matrix product */
/* c[x][y] = a[x][0]*b[0][y]+a[x][1]*b[1][y]+a[x][2]*b[2][y]+a[x][3]*b[3][y];*/

#define MATRIX_PRODUCT_4X4(c, a, b)                                            \
                                                                               \
    {                                                                          \
        c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0] +  \
                  a[0][3] * b[3][0];                                           \
        c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1] +  \
                  a[0][3] * b[3][1];                                           \
        c[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2] +  \
                  a[0][3] * b[3][2];                                           \
        c[0][3] = a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3] +  \
                  a[0][3] * b[3][3];                                           \
                                                                               \
        c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0] +  \
                  a[1][3] * b[3][0];                                           \
        c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1] +  \
                  a[1][3] * b[3][1];                                           \
        c[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2] +  \
                  a[1][3] * b[3][2];                                           \
        c[1][3] = a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3] +  \
                  a[1][3] * b[3][3];                                           \
                                                                               \
        c[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0] +  \
                  a[2][3] * b[3][0];                                           \
        c[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1] +  \
                  a[2][3] * b[3][1];                                           \
        c[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2] +  \
                  a[2][3] * b[3][2];                                           \
        c[2][3] = a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3] +  \
                  a[2][3] * b[3][3];                                           \
                                                                               \
        c[3][0] = a[3][0] * b[0][0] + a[3][1] * b[1][0] + a[3][2] * b[2][0] +  \
                  a[3][3] * b[3][0];                                           \
        c[3][1] = a[3][0] * b[0][1] + a[3][1] * b[1][1] + a[3][2] * b[2][1] +  \
                  a[3][3] * b[3][1];                                           \
        c[3][2] = a[3][0] * b[0][2] + a[3][1] * b[1][2] + a[3][2] * b[2][2] +  \
                  a[3][3] * b[3][2];                                           \
        c[3][3] = a[3][0] * b[0][3] + a[3][1] * b[1][3] + a[3][2] * b[2][3] +  \
                  a[3][3] * b[3][3];                                           \
    }

/* ========================================================== */
/* matrix times vector */

#define MAT_DOT_VEC_2X2(p, m, v)                                               \
                                                                               \
    {                                                                          \
        p[0] = m[0][0] * v[0] + m[0][1] * v[1];                                \
        p[1] = m[1][0] * v[0] + m[1][1] * v[1];                                \
    }

/* ========================================================== */
/* matrix times vector */

#define MAT_DOT_VEC_3X3(p, m, v)                                               \
                                                                               \
    {                                                                          \
        p[0] = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2];               \
        p[1] = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2];               \
        p[2] = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2];               \
    }

/* ========================================================== */
/* matrix times vector */

#define MAT_DOT_VEC_4X4(p, m, v)                                               \
                                                                               \
    {                                                                          \
        p[0] =                                                                 \
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3]; \
        p[1] =                                                                 \
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3]; \
        p[2] =                                                                 \
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3]; \
        p[3] =                                                                 \
            m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3]; \
    }

/* ========================================================== */
/* vector transpose times matrix */
/* p[j] = v[0]*m[0][j] + v[1]*m[1][j] + v[2]*m[2][j]; */

#define VEC_DOT_MAT_3X3(p, v, m)                                               \
                                                                               \
    {                                                                          \
        p[0] = v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0];               \
        p[1] = v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1];               \
        p[2] = v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2];               \
    }

/* ========================================================== */
/* affine matrix times vector */
/* The matrix is assumed to be an affine matrix, with last two
 * entries representing a translation */

#define MAT_DOT_VEC_2X3(p, m, v)                                               \
                                                                               \
    {                                                                          \
        p[0] = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2];                      \
        p[1] = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2];                      \
    }

/* ========================================================== */
/* inverse transpose of matrix times vector
 *
 * This macro computes inverse transpose of matrix m,
 * and multiplies vector v into it, to yeild vector p
 *
 * DANGER !!! Do Not use this on normal vectors!!!
 * It will leave normals the wrong length !!!
 * See macro below for use on normals.
 */

#define INV_TRANSP_MAT_DOT_VEC_2X2(p, m, v)                                    \
                                                                               \
    {                                                                          \
        gleDouble det;                                                         \
                                                                               \
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0];                           \
        p[0] = m[1][1] * v[0] - m[1][0] * v[1];                                \
        p[1] = -m[0][1] * v[0] + m[0][0] * v[1];                               \
                                                                               \
        /* if matrix not singular, and not orthonormal, then renormalize */    \
        if ((det != 1.0) && (det != 0.0)) {                                    \
            det = 1.0 / det;                                                   \
            p[0] *= det;                                                       \
            p[1] *= det;                                                       \
        }                                                                      \
    }

/* ========================================================== */
/* transform normal vector by inverse transpose of matrix
 * and then renormalize the vector
 *
 * This macro computes inverse transpose of matrix m,
 * and multiplies vector v into it, to yeild vector p
 * Vector p is then normalized.
 */

#define NORM_XFORM_2X2(p, m, v)                                                \
                                                                               \
    {                                                                          \
        double mlen;                                                           \
                                                                               \
        if ((m[0][1] != 0.0) || (m[1][0] != 0.0) || (m[0][0] != m[1][1])) {    \
            p[0] = m[1][1] * v[0] - m[1][0] * v[1];                            \
            p[1] = -m[0][1] * v[0] + m[0][0] * v[1];                           \
                                                                               \
            mlen = p[0] * p[0] + p[1] * p[1];                                  \
            mlen = 1.0 / sqrt(mlen);                                           \
            p[0] *= mlen;                                                      \
            p[1] *= mlen;                                                      \
        }                                                                      \
        else {                                                                 \
            VEC_COPY_2(p, v);                                                  \
        }                                                                      \
    }

/* ========================================================== */
/* outer product of vector times vector transpose
 *
 * The outer product of vector v and vector transpose t yeilds
 * dyadic matrix m.
 */

#define OUTER_PRODUCT_2X2(m, v, t)                                             \
                                                                               \
    {                                                                          \
        m[0][0] = v[0] * t[0];                                                 \
        m[0][1] = v[0] * t[1];                                                 \
                                                                               \
        m[1][0] = v[1] * t[0];                                                 \
        m[1][1] = v[1] * t[1];                                                 \
    }

/* ========================================================== */
/* outer product of vector times vector transpose
 *
 * The outer product of vector v and vector transpose t yeilds
 * dyadic matrix m.
 */

#define OUTER_PRODUCT_3X3(m, v, t)                                             \
                                                                               \
    {                                                                          \
        m[0][0] = v[0] * t[0];                                                 \
        m[0][1] = v[0] * t[1];                                                 \
        m[0][2] = v[0] * t[2];                                                 \
                                                                               \
        m[1][0] = v[1] * t[0];                                                 \
        m[1][1] = v[1] * t[1];                                                 \
        m[1][2] = v[1] * t[2];                                                 \
                                                                               \
        m[2][0] = v[2] * t[0];                                                 \
        m[2][1] = v[2] * t[1];                                                 \
        m[2][2] = v[2] * t[2];                                                 \
    }

/* ========================================================== */
/* outer product of vector times vector transpose
 *
 * The outer product of vector v and vector transpose t yeilds
 * dyadic matrix m.
 */

#define OUTER_PRODUCT_4X4(m, v, t)                                             \
                                                                               \
    {                                                                          \
        m[0][0] = v[0] * t[0];                                                 \
        m[0][1] = v[0] * t[1];                                                 \
        m[0][2] = v[0] * t[2];                                                 \
        m[0][3] = v[0] * t[3];                                                 \
                                                                               \
        m[1][0] = v[1] * t[0];                                                 \
        m[1][1] = v[1] * t[1];                                                 \
        m[1][2] = v[1] * t[2];                                                 \
        m[1][3] = v[1] * t[3];                                                 \
                                                                               \
        m[2][0] = v[2] * t[0];                                                 \
        m[2][1] = v[2] * t[1];                                                 \
        m[2][2] = v[2] * t[2];                                                 \
        m[2][3] = v[2] * t[3];                                                 \
                                                                               \
        m[3][0] = v[3] * t[0];                                                 \
        m[3][1] = v[3] * t[1];                                                 \
        m[3][2] = v[3] * t[2];                                                 \
        m[3][3] = v[3] * t[3];                                                 \
    }

/* +========================================================== */
/* outer product of vector times vector transpose
 *
 * The outer product of vector v and vector transpose t yeilds
 * dyadic matrix m.
 */

#define ACCUM_OUTER_PRODUCT_2X2(m, v, t)                                       \
                                                                               \
    {                                                                          \
        m[0][0] += v[0] * t[0];                                                \
        m[0][1] += v[0] * t[1];                                                \
                                                                               \
        m[1][0] += v[1] * t[0];                                                \
        m[1][1] += v[1] * t[1];                                                \
    }

/* +========================================================== */
/* outer product of vector times vector transpose
 *
 * The outer product of vector v and vector transpose t yeilds
 * dyadic matrix m.
 */

#define ACCUM_OUTER_PRODUCT_3X3(m, v, t)                                       \
                                                                               \
    {                                                                          \
        m[0][0] += v[0] * t[0];                                                \
        m[0][1] += v[0] * t[1];                                                \
        m[0][2] += v[0] * t[2];                                                \
                                                                               \
        m[1][0] += v[1] * t[0];                                                \
        m[1][1] += v[1] * t[1];                                                \
        m[1][2] += v[1] * t[2];                                                \
                                                                               \
        m[2][0] += v[2] * t[0];                                                \
        m[2][1] += v[2] * t[1];                                                \
        m[2][2] += v[2] * t[2];                                                \
    }

/* +========================================================== */
/* outer product of vector times vector transpose
 *
 * The outer product of vector v and vector transpose t yeilds
 * dyadic matrix m.
 */

#define ACCUM_OUTER_PRODUCT_4X4(m, v, t)                                       \
                                                                               \
    {                                                                          \
        m[0][0] += v[0] * t[0];                                                \
        m[0][1] += v[0] * t[1];                                                \
        m[0][2] += v[0] * t[2];                                                \
        m[0][3] += v[0] * t[3];                                                \
                                                                               \
        m[1][0] += v[1] * t[0];                                                \
        m[1][1] += v[1] * t[1];                                                \
        m[1][2] += v[1] * t[2];                                                \
        m[1][3] += v[1] * t[3];                                                \
                                                                               \
        m[2][0] += v[2] * t[0];                                                \
        m[2][1] += v[2] * t[1];                                                \
        m[2][2] += v[2] * t[2];                                                \
        m[2][3] += v[2] * t[3];                                                \
                                                                               \
        m[3][0] += v[3] * t[0];                                                \
        m[3][1] += v[3] * t[1];                                                \
        m[3][2] += v[3] * t[2];                                                \
        m[3][3] += v[3] * t[3];                                                \
    }

/* +========================================================== */
/* determinant of matrix
 *
 * Computes determinant of matrix m, returning d
 */

#define DETERMINANT_2X2(d, m)                                                  \
                                                                               \
    {                                                                          \
        d = m[0][0] * m[1][1] - m[0][1] * m[1][0];                             \
    }

/* ========================================================== */
/* determinant of matrix
 *
 * Computes determinant of matrix m, returning d
 */

#define DETERMINANT_3X3(d, m)                                                  \
                                                                               \
    {                                                                          \
        d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]);                 \
        d -= m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]);                \
        d += m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);                \
    }

/* ========================================================== */
/* i,j,th cofactor of a 4x4 matrix
 *
 */

#define COFACTOR_4X4_IJ(fac, m, i, j)                                          \
                                                                               \
    {                                                                          \
        int ii[4], jj[4], k;                                                   \
                                                                               \
        /* compute which row, columnt to skip */                               \
        for (k = 0; k < i; k++)                                                \
            ii[k] = k;                                                         \
        for (k = i; k < 3; k++)                                                \
            ii[k] = k + 1;                                                     \
        for (k = 0; k < j; k++)                                                \
            jj[k] = k;                                                         \
        for (k = j; k < 3; k++)                                                \
            jj[k] = k + 1;                                                     \
                                                                               \
        (fac) = m[ii[0]][jj[0]] * (m[ii[1]][jj[1]] * m[ii[2]][jj[2]] -         \
                                   m[ii[1]][jj[2]] * m[ii[2]][jj[1]]);         \
        (fac) -= m[ii[0]][jj[1]] * (m[ii[1]][jj[0]] * m[ii[2]][jj[2]] -        \
                                    m[ii[1]][jj[2]] * m[ii[2]][jj[0]]);        \
        (fac) += m[ii[0]][jj[2]] * (m[ii[1]][jj[0]] * m[ii[2]][jj[1]] -        \
                                    m[ii[1]][jj[1]] * m[ii[2]][jj[0]]);        \
                                                                               \
        /* compute sign */                                                     \
        k = i + j;                                                             \
        if (k != (k / 2) * 2) {                                                \
            (fac) = -(fac);                                                    \
        }                                                                      \
    }

/* ========================================================== */
/* determinant of matrix
 *
 * Computes determinant of matrix m, returning d
 */

#define DETERMINANT_4X4(d, m)                                                  \
                                                                               \
    {                                                                          \
        double cofac;                                                          \
        COFACTOR_4X4_IJ(cofac, m, 0, 0);                                       \
        d = m[0][0] * cofac;                                                   \
        COFACTOR_4X4_IJ(cofac, m, 0, 1);                                       \
        d += m[0][1] * cofac;                                                  \
        COFACTOR_4X4_IJ(cofac, m, 0, 2);                                       \
        d += m[0][2] * cofac;                                                  \
        COFACTOR_4X4_IJ(cofac, m, 0, 3);                                       \
        d += m[0][3] * cofac;                                                  \
    }

/* ========================================================== */
/* cofactor of matrix
 *
 * Computes cofactor of matrix m, returning a
 */

#define COFACTOR_2X2(a, m)                                                     \
                                                                               \
    {                                                                          \
        a[0][0] = (m)[1][1];                                                   \
        a[0][1] = -(m)[1][0];                                                  \
        a[1][0] = -(m)[0][1];                                                  \
        a[1][1] = (m)[0][0];                                                   \
    }

/* ========================================================== */
/* cofactor of matrix
 *
 * Computes cofactor of matrix m, returning a
 */

#define COFACTOR_3X3(a, m)                                                     \
                                                                               \
    {                                                                          \
        a[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];                       \
        a[0][1] = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]);                    \
        a[0][2] = m[1][0] * m[2][1] - m[1][1] * m[2][0];                       \
        a[1][0] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);                    \
        a[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];                       \
        a[1][2] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);                    \
        a[2][0] = m[0][1] * m[1][2] - m[0][2] * m[1][1];                       \
        a[2][1] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);                    \
   a[2][2] = m[0][0]*m[1][1] - m[0][1]*m[1][0]);                               \
    }

/* ========================================================== */
/* cofactor of matrix
 *
 * Computes cofactor of matrix m, returning a
 */

#define COFACTOR_4X4(a, m)                                                     \
                                                                               \
    {                                                                          \
        int i, j;                                                              \
                                                                               \
        for (i = 0; i < 4; i++) {                                              \
            for (j = 0; j < 4; j++) {                                          \
                COFACTOR_4X4_IJ(a[i][j], m, i, j);                             \
            }                                                                  \
        }                                                                      \
    }

/* ========================================================== */
/* adjoint of matrix
 *
 * Computes adjoint of matrix m, returning a
 * (Note that adjoint is just the transpose of the cofactor matrix)
 */

#define ADJOINT_2X2(a, m)                                                      \
                                                                               \
    {                                                                          \
        a[0][0] = (m)[1][1];                                                   \
        a[1][0] = -(m)[1][0];                                                  \
        a[0][1] = -(m)[0][1];                                                  \
        a[1][1] = (m)[0][0];                                                   \
    }

/* ========================================================== */
/* adjoint of matrix
 *
 * Computes adjoint of matrix m, returning a
 * (Note that adjoint is just the transpose of the cofactor matrix)
 */

#define ADJOINT_3X3(a, m)                                                      \
                                                                               \
    {                                                                          \
        a[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];                       \
        a[1][0] = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]);                    \
        a[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0];                       \
        a[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);                    \
        a[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];                       \
        a[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);                    \
        a[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1];                       \
        a[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);                    \
   a[2][2] = m[0][0]*m[1][1] - m[0][1]*m[1][0]);                               \
    }

/* ========================================================== */
/* adjoint of matrix
 *
 * Computes adjoint of matrix m, returning a
 * (Note that adjoint is just the transpose of the cofactor matrix)
 */

#define ADJOINT_4X4(a, m)                                                      \
                                                                               \
    {                                                                          \
        int i, j;                                                              \
                                                                               \
        for (i = 0; i < 4; i++) {                                              \
            for (j = 0; j < 4; j++) {                                          \
                COFACTOR_4X4_IJ(a[j][i], m, i, j);                             \
            }                                                                  \
        }                                                                      \
    }

/* ========================================================== */
/* compute adjoint of matrix and scale
 *
 * Computes adjoint of matrix m, scales it by s, returning a
 */

#define SCALE_ADJOINT_2X2(a, s, m)                                             \
                                                                               \
    {                                                                          \
        a[0][0] = (s)*m[1][1];                                                 \
        a[1][0] = -(s)*m[1][0];                                                \
        a[0][1] = -(s)*m[0][1];                                                \
        a[1][1] = (s)*m[0][0];                                                 \
    }

/* ========================================================== */
/* compute adjoint of matrix and scale
 *
 * Computes adjoint of matrix m, scales it by s, returning a
 */

#define SCALE_ADJOINT_3X3(a, s, m)                                             \
                                                                               \
    {                                                                          \
        a[0][0] = (s) * (m[1][1] * m[2][2] - m[1][2] * m[2][1]);               \
        a[1][0] = (s) * (m[1][2] * m[2][0] - m[1][0] * m[2][2]);               \
        a[2][0] = (s) * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);               \
                                                                               \
        a[0][1] = (s) * (m[0][2] * m[2][1] - m[0][1] * m[2][2]);               \
        a[1][1] = (s) * (m[0][0] * m[2][2] - m[0][2] * m[2][0]);               \
        a[2][1] = (s) * (m[0][1] * m[2][0] - m[0][0] * m[2][1]);               \
                                                                               \
        a[0][2] = (s) * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);               \
        a[1][2] = (s) * (m[0][2] * m[1][0] - m[0][0] * m[1][2]);               \
        a[2][2] = (s) * (m[0][0] * m[1][1] - m[0][1] * m[1][0]);               \
    }

/* ========================================================== */
/* compute adjoint of matrix and scale
 *
 * Computes adjoint of matrix m, scales it by s, returning a
 */

#define SCALE_ADJOINT_4X4(a, s, m)                                             \
                                                                               \
    {                                                                          \
        int i, j;                                                              \
                                                                               \
        for (i = 0; i < 4; i++) {                                              \
            for (j = 0; j < 4; j++) {                                          \
                COFACTOR_4X4_IJ(a[j][i], m, i, j);                             \
                a[j][i] *= s;                                                  \
            }                                                                  \
        }                                                                      \
    }

/* ========================================================== */
/* inverse of matrix
 *
 * Compute inverse of matrix a, returning determinant m and
 * inverse b
 */

#define INVERT_2X2(b, det, a)                                                  \
                                                                               \
    {                                                                          \
        double tmp;                                                            \
        DETERMINANT_2X2(det, a);                                               \
        tmp = 1.0 / (det);                                                     \
        SCALE_ADJOINT_2X2(b, tmp, a);                                          \
    }

/* ========================================================== */
/* inverse of matrix
 *
 * Compute inverse of matrix a, returning determinant m and
 * inverse b
 */

#define INVERT_3X3(b, det, a)                                                  \
                                                                               \
    {                                                                          \
        float tmp;                                                             \
        DETERMINANT_3X3(det, a);                                               \
        tmp = 1.0f / (det);                                                    \
        SCALE_ADJOINT_3X3(b, tmp, a);                                          \
    }

/* ========================================================== */
/* inverse of matrix
 *
 * Compute inverse of matrix a, returning determinant m and
 * inverse b
 */

#define INVERT_4X4(b, det, a)                                                  \
                                                                               \
    {                                                                          \
        double tmp;                                                            \
        DETERMINANT_4X4(det, a);                                               \
        tmp = 1.0 / (det);                                                     \
        SCALE_ADJOINT_4X4(b, tmp, a);                                          \
    }

/* ========================================================== */
#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

namespace ezsift {

/****************************************
 * Constant parameters
 ***************************************/

// default number of sampled intervals per octave
    static int SIFT_INTVLS = 3;

// default sigma for initial gaussian smoothing
    static float SIFT_SIGMA = 1.6f;

// the radius of Gaussian filter kernel;
// Gaussian filter mask will be (2*radius+1)x(2*radius+1).
// People use 2 or 3 most.
    static float SIFT_GAUSSIAN_FILTER_RADIUS = 3.0f;

// default threshold on keypoint contrast |D(x)|
    static float SIFT_CONTR_THR = 8.0f; // 8.0f;

// default threshold on keypoint ratio of principle curvatures
    static float SIFT_CURV_THR = 10.0f;

// The keypoint refinement smaller than this threshold will be discarded.
    static float SIFT_KEYPOINT_SUBPiXEL_THR = 0.6f;

// double image size before pyramid construction?
    static bool SIFT_IMG_DBL = false; // true;

// assumed gaussian blur for input image
    static float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
    static int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
    static int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
    static int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
    static float SIFT_ORI_SIG_FCTR =
            1.5f; // Can affect the orientation computation.

// determines the radius of the region used in orientation assignment
    static float SIFT_ORI_RADIUS =
            3 * SIFT_ORI_SIG_FCTR; // Can affect the orientation computation.

// orientation magnitude relative to max that results in new feature
    static float SIFT_ORI_PEAK_RATIO = 0.8f;

// maximum number of orientations for each keypoint location
// static const float SIFT_ORI_MAX_ORI = 4;

// determines the size of a single descriptor orientation histogram
    static float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
    static float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
    static float SIFT_INT_DESCR_FCTR = 512.f;

// default width of descriptor histogram array
    static int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
    static int SIFT_DESCR_HIST_BINS = 8;

// default value of the nearest-neighbour distance ratio threshold
// |DR_nearest|/|DR_2nd_nearest|<SIFT_MATCH_NNDR_THR is considered as a match.
    static float SIFT_MATCH_NNDR_THR = 0.65f;

#if 0
    // intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
    typedef float sift_wt;
    static const int SIFT_FIXPT_SCALE = 1;
#endif


/****************************************
 * Definitions
 ***************************************/

template <typename T>

class Image {
public:
    int w;
    int h;
    T *data;

    Image();

    Image(int _w, int _h);

    // Copy construction function
    Image(const Image<T> &img);

    ~Image();

    Image<T> &operator=(const Image<T> &img);

    void init(int _w, int _h);

    void reinit(int _w, int _h);

    void release();

    int read_pgm(const char *filename);
    int write_pgm(const char *filename);

    Image<unsigned char> to_uchar() const;

    Image<float> to_float() const;

    // Upsample the image by 2x, linear interpolation.
    Image<T> upsample_2x() const;

    // Downsample the image by 2x, nearest neighbor interpolation.
    Image<T> downsample_2x() const;
};

struct ImagePPM {
    int w;
    int h;
    unsigned char *img_r;
    unsigned char *img_g;
    unsigned char *img_b;
};

#define DEGREE_OF_DESCRIPTORS (128)
    struct SiftKeypoint {
        int octave;   // octave number
        int layer;    // layer number
        float rlayer; // real number of layer number

        float r;     // normalized row coordinate
        float c;     // normalized col coordinate
        float scale; // normalized scale

        float ri;          // row coordinate in that layer.
        float ci;          // column coordinate in that layer.
        float layer_scale; // the scale of that layer

        float ori; // orientation in degrees.
        float mag; // magnitude

        float descriptors[DEGREE_OF_DESCRIPTORS];
    };

// Match pair structure. Use for interest point matching.
    struct MatchPair {
        int r1;
        int c1;
        int r2;
        int c2;
    };

#ifdef _WIN32
    #include <windows.h>
#if !defined(_WINSOCK2API_) && !defined(_WINSOCKAPI_)
struct timeval {
    long tv_sec;
    long tv_usec;
};
#endif
#else //_WIN32
#include <sys/time.h>
#endif //_WIN32

    template <typename timer_dt>
    class Timer {
    public:
        Timer();
        ~Timer(){};

        void start();
        void stop();
        timer_dt get_time();
        timer_dt stop_get();
        timer_dt stop_get_start();

#ifdef _WIN32
        double freq;
    LARGE_INTEGER start_time;
    LARGE_INTEGER finish_time;
#else  //_WIN32
        struct timeval start_time;
        struct timeval finish_time;
#endif //_WIN32
    };

// Definition
#ifdef _WIN32
    int gettimeofday(struct timeval *tv, int t)
{
    union {
        long long ns100;
        FILETIME ft;
    } now;

    GetSystemTimeAsFileTime(&now.ft);
    tv->tv_usec = (long)((now.ns100 / 10LL) % 1000000LL);
    tv->tv_sec = (long)((now.ns100 - 116444736000000000LL) / 10000000LL);
    return (0);
} // gettimeofday()
#endif //_WIN32

    template <typename timer_dt>
    Timer<timer_dt>::Timer()
    {
#ifdef _WIN32
        LARGE_INTEGER tmp;
    QueryPerformanceFrequency((LARGE_INTEGER *)&tmp);
    freq = (double)tmp.QuadPart / 1000.0;
#endif
    }

    template <typename timer_dt>
    void Timer<timer_dt>::start()
    {
#ifdef _WIN32
        QueryPerformanceCounter((LARGE_INTEGER *)&start_time);
#else  //_WIN32
        gettimeofday(&start_time, 0);
#endif //_WIN32
    }

    template <typename timer_dt>
    void Timer<timer_dt>::stop()
    {
#ifdef _WIN32
        QueryPerformanceCounter((LARGE_INTEGER *)&finish_time);
#else  //_WIN32
        gettimeofday(&finish_time, 0);
#endif //_WIN32
    }

    template <typename timer_dt>
    timer_dt Timer<timer_dt>::get_time()
    {
        timer_dt interval = 0.0f;

#ifdef _WIN32
        interval =
        (timer_dt)((double)(finish_time.QuadPart - start_time.QuadPart) / freq);
#else
        // time difference in milli-seconds
        interval = (timer_dt)(1000.0 * (finish_time.tv_sec - start_time.tv_sec) +
                              (0.001 * (finish_time.tv_usec - start_time.tv_usec)));
#endif //_WIN32

        return interval;
    }

    template <typename timer_dt>
    timer_dt Timer<timer_dt>::stop_get()
    {
        timer_dt interval;
        stop();
        interval = get_time();

        return interval;
    }

// Stop the timer, get the time interval, then start the timer again.
    template <typename timer_dt>
    timer_dt Timer<timer_dt>::stop_get_start()
    {
        timer_dt interval;
        stop();
        interval = get_time();
        start();

        return interval;
    }



// Optimization options
#define USE_FAST_FUNC 1

// Some debug options
// Dump functions to get intermediate results
#define DUMP_OCTAVE_IMAGE 0
#define DUMP_GAUSSIAN_PYRAMID_IMAGE 0
#define DUMP_DOG_IMAGE 0

// Print out matched keypoint pairs in match_keypoints() function.
#define PRINT_MATCH_KEYPOINTS 1

// Macro definition
#define PI 3.141592653589793f
#define _2PI 6.283185307179586f
#define PI_4 0.785398163397448f
#define PI_3_4 2.356194490192345f
#define SQRT2 1.414213562373095f

#define MAX(a, b) (a >= b ? a : b)
#define MIN(a, b) (a <= b ? a : b)

// Helper functions
    inline float my_log2(float n)
    {
        // Visual C++ does not have log2...
        return (float)((log(n)) / 0.69314718055994530941723212145818);
    }

// Debug log printing
    void inline logd(const char *format, ...)
    {
#ifdef ENABLE_DEBUG_PRINT
        va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
#endif
        return;
    }

    void inline flogd(FILE *fp, const char *format, ...)
    {
#ifdef ENABLE_DEBUG_FPRINT
        va_list args;
    va_start(args, format);
    vfprintf(fp, format, args);
    va_end(args);
#endif
        return;
    }

// Fast math functions
// Fast Atan2() function
#define EPSILON_F 1.19209290E-07F
    inline float fast_atan2_f(float y, float x)
    {
        float angle, r;
        float const c3 = 0.1821F;
        float const c1 = 0.9675F;
        float abs_y = fabsf(y) + EPSILON_F;

        if (x >= 0) {
            r = (x - abs_y) / (x + abs_y);
            angle = PI_4;
        }
        else {
            r = (x + abs_y) / (abs_y - x);
            angle = PI_3_4;
        }
        angle += (c3 * r * r - c1) * r;

        return (y < 0) ? _2PI - angle : angle;
    }

// Fast Sqrt() function
    inline float fast_resqrt_f(float x)
    {
        // 32-bit version
        union {
            float x;
            int i;
        } u;

        float xhalf = (float)0.5 * x;

        // convert floating point value in RAW integer
        u.x = x;

        // gives initial guess y0
        u.i = 0x5f3759df - (u.i >> 1);

        // two Newton steps
        u.x = u.x * ((float)1.5 - xhalf * u.x * u.x);
        u.x = u.x * ((float)1.5 - xhalf * u.x * u.x);
        return u.x;
    }

    inline float fast_sqrt_f(float x)
    {
        return (x < 1e-8) ? 0 : x * fast_resqrt_f(x);
    }

/****************************************
 * Utility functions
 ***************************************/
// Image operations
// Get pixel from an image with unsigned char datatype.
    inline unsigned char get_pixel(unsigned char *imageData, int w, int h, int r,
                                   int c)
    {
        unsigned char val;
        if (c >= 0 && c < w && r >= 0 && r < h) {
            val = imageData[r * w + c];
        }
        else if (c < 0) {
            val = imageData[r * w];
        }
        else if (c >= w) {
            val = imageData[r * w + w - 1];
        }
        else if (r < 0) {
            val = imageData[c];
        }
        else if (r >= h) {
            val = imageData[(h - 1) * w + c];
        }
        else {
            val = 0;
        }
        return val;
    }

// Get pixel value from an image with float data type.
    inline float get_pixel_f(float *imageData, int w, int h, int r, int c)
    {
        float val;
        if (c >= 0 && c < w && r >= 0 && r < h) {
            val = imageData[r * w + c];
        }
        else if (c < 0) {
            val = imageData[r * w];
        }
        else if (c >= w) {
            val = imageData[r * w + w - 1];
        }
        else if (r < 0) {
            val = imageData[c];
        }
        else if (r >= h) {
            val = imageData[(h - 1) * w + c];
        }
        else {
            val = 0.0f;
        }
        return val;
    }

    int read_pgm(const char *filename, unsigned char *&data, int &w, int &h)
    {
        unsigned char *_data;
        FILE *in_file;
        char ch, type;
        int i;

        in_file = fopen(filename, "rb");
        if (!in_file) {
            fprintf(stderr, "ERROR(0): Fail to open file %s\n", filename);
            return -1;
        }
        // Determine pgm image type (only type three can be used)
        ch = getc(in_file);
        if (ch != 'P') {
            printf("ERROR(1): Not valid pgm/ppm file type\n");
            return -1;
        }
        ch = getc(in_file);
        // Convert the one digit integer currently represented as a character to
        // an integer(48 == '0')
        type = ch - 48;
        if (type != 5) {
            printf("ERROR(2): this file type (P%d) is not supported!\n", type);
            return -1;
        }
        while (getc(in_file) != '\n')
            ;                          // Skip to end of line
        while (getc(in_file) == '#') { // Skip comment lines
            while (getc(in_file) != '\n')
                ;
        }
        fseek(in_file, -1, SEEK_CUR); // Backup one character

        fscanf(in_file, "%d", &w);
        fscanf(in_file, "%d", &h);
        fscanf(in_file, "%d", &i); // Skipped here
        while (getc(in_file) != '\n')
            ;
        _data = (unsigned char *)malloc((w) * (h) * sizeof(unsigned char));

        fread(_data, sizeof(unsigned char), (w) * (h), in_file);
        data = _data;

        return 0;
    }

    void write_pgm(const char *filename, unsigned char *data, int w, int h)
    {
        FILE *out_file;
        assert(w > 0);
        assert(h > 0);

        out_file = fopen(filename, "wb");
        if (!out_file) {
            fprintf(stderr, "Fail to open file: %s\n", filename);
            return;
        }

        fprintf(out_file, "P5\n");
        fprintf(out_file, "%d %d\n255\n", w, h);
        fwrite(data, sizeof(unsigned char), w * h, out_file);
        fclose(out_file);
    }

    void write_float_pgm(const char *filename, float *data, int w, int h, int mode)
    {
        int i, j;
        unsigned char *charImg;
        int tmpInt;
        charImg = (unsigned char *)malloc(w * h * sizeof(unsigned char));
        for (i = 0; i < h; i++) {
            for (j = 0; j < w; j++) {
                if (mode == 1) { // clop
                    if (data[i * w + j] >= 255.0) {
                        charImg[i * w + j] = 255;
                    }
                    else if (data[i * w + j] <= 0.0) {
                        charImg[i * w + j] = 0;
                    }
                    else {
                        charImg[i * w + j] = (int)data[i * w + j];
                    }
                }
                else if (mode == 2) { // abs, x10, clop
                    tmpInt = (int)(fabs(data[i * w + j]) * 10.0);
                    if (fabs(data[i * w + j]) >= 255) {
                        charImg[i * w + j] = 255;
                    }
                    else if (tmpInt <= 0) {
                        charImg[i * w + j] = 0;
                    }
                    else {
                        charImg[i * w + j] = tmpInt;
                    }
                }
                else {
                    return;
                }
            }
        }
        write_pgm(filename, charImg, w, h);
        free(charImg);
    }

    void setPixelRed(ImagePPM *img, int r, int c)
    {
        if ((r >= 0) && (r < img->h) && (c >= 0) && (c < img->w)) {
            img->img_r[r * img->w + c] = 0;
            img->img_g[r * img->w + c] = 0;
            img->img_b[r * img->w + c] = 255;
        }
    }

    void draw_red_circle(ImagePPM *imgPPM, int r, int c, int cR)
    {
        int cx = -cR, cy = 0, err = 2 - 2 * cR; // II. Quadrant
        do {
            setPixelRed(imgPPM, r - cx, c + cy); //   I. Quadrant
            setPixelRed(imgPPM, r - cy, c - cx); //  II. Quadrant
            setPixelRed(imgPPM, r + cx, c - cy); // III. Quadrant
            setPixelRed(imgPPM, r + cy, c + cx); //  IV. Quadrant
            cR = err;
            if (cR > cx)
                err += ++cx * 2 + 1; // e_xy+e_x > 0
            if (cR <= cy)
                err += ++cy * 2 + 1; // e_xy+e_y < 0
        } while (cx < 0);
    }

    void draw_circle(ImagePPM *imgPPM, int r, int c, int cR, float thickness)
    {
        int x, y;
        float f = thickness;
        for (x = -cR; x <= +cR; x++) // column
        {
            for (y = -cR; y <= +cR; y++) // row
            {
                if ((((x * x) + (y * y)) > (cR * cR) - (f / 2)) &&
                    (((x * x) + (y * y)) < (cR * cR) + (f / 2)))
                    setPixelRed(imgPPM, y + r, x + c);
            }
        }
    }

// http://en.wikipedia.org/wiki/Midpoint_circle_algorithm
    void rasterCircle(ImagePPM *imgPPM, int r, int c, int radius)
    {
        int f = 1 - radius;
        int ddF_x = 1;
        int ddF_y = -2 * radius;
        int x = 0;
        int y = radius;

        int x0 = r;
        int y0 = c;

        setPixelRed(imgPPM, x0, y0 + radius);
        setPixelRed(imgPPM, x0, y0 - radius);
        setPixelRed(imgPPM, x0 + radius, y0);
        setPixelRed(imgPPM, x0 - radius, y0);

        while (x < y) {
            // ddF_x == 2 * x + 1;
            // ddF_y == -2 * y;
            // f == x*x + y*y - radius*radius + 2*x - y + 1;
            if (f >= 0) {
                y--;
                ddF_y += 2;
                f += ddF_y;
            }
            x++;
            ddF_x += 2;
            f += ddF_x;
            setPixelRed(imgPPM, x0 + x, y0 + y);
            setPixelRed(imgPPM, x0 - x, y0 + y);
            setPixelRed(imgPPM, x0 + x, y0 - y);
            setPixelRed(imgPPM, x0 - x, y0 - y);
            setPixelRed(imgPPM, x0 + y, y0 + x);
            setPixelRed(imgPPM, x0 - y, y0 + x);
            setPixelRed(imgPPM, x0 + y, y0 - x);
            setPixelRed(imgPPM, x0 - y, y0 - x);
        }
    }

    void draw_red_orientation(ImagePPM *imgPPM, int x, int y, float ori, int cR)
    {
        int xe = (int)(x + cos(ori) * cR), ye = (int)(y + sin(ori) * cR);
        // Bresenham's line algorithm
        int dx = abs(xe - x), sx = x < xe ? 1 : -1;
        int dy = -abs(ye - y), sy = y < ye ? 1 : -1;
        int err = dx + dy, e2; /* error value e_xy */

        for (;;) { /* loop */
            setPixelRed(imgPPM, y, x);
            if (x == xe && y == ye)
                break;
            e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x += sx;
            } /* e_xy+e_x > 0 */
            if (e2 <= dx) {
                err += dx;
                y += sy;
            } /* e_xy+e_y < 0 */
        }
    }

    void skip_comment(FILE *fp)
    {
        int c;
        while (isspace(c = getc(fp)))
            ;
        if (c != '#') {
            ungetc(c, fp);
            return;
        }

        do {
            c = getc(fp);
        } while (c != '\n' && c != EOF);
    }

    int read_ppm(const char *filename, unsigned char *&data, int &w, int &h)
    {
        FILE *fp;

        if ((fp = fopen(filename, "rb")) == NULL) {
            printf("Could not read file: %s\n", filename);
            return -1;
        }

        int width, height, maxComp;
        char cookie[3];
        fscanf(fp, "%2s", cookie);
        if (strcmp("P6", cookie)) {
            printf("Wrong file type\n");
            fclose(fp);
            return -1;
        }
        skip_comment(fp);

        fscanf(fp, "%4d", &width);
        fscanf(fp, "%4d", &height);
        fscanf(fp, "%3d", &maxComp);
        fread(cookie, 1, 1, fp); // Read newline which follows maxval

        if (maxComp != 255) {
            printf("Data error: %d\n", maxComp);
            fclose(fp);
            return -1;
        }

        if (data == NULL) {
            data = new unsigned char[3 * width * height];
        }

        size_t res = fread(data, sizeof(unsigned char), 3 * width * height, fp);
        assert((int)res == 3 * width * height);
        fclose(fp);

        w = width;
        h = height;

        return 0;
    }

    void write_ppm(const char *filename, unsigned char *data, int w, int h)
    {
        FILE *fp;
        if ((fp = fopen(filename, "wb")) == NULL) {
            printf("Cannot write to file %s\n", filename);
            return;
        }

        /* Write header */
        fprintf(fp, "P6\n");
        fprintf(fp, "%d %d\n", w, h);
        fprintf(fp, "255\n");

        fwrite(data, sizeof(unsigned char), w * h * 3, fp);
        fclose(fp);
    }

    void write_rgb2ppm(const char *filename, unsigned char *r, unsigned char *g,
                       unsigned char *b, int w, int h)
    {
        FILE *out_file;
        int i;

        unsigned char *obuf =
                (unsigned char *)malloc(3 * w * h * sizeof(unsigned char));

        for (i = 0; i < w * h; i++) {
            obuf[3 * i + 0] = r[i];
            obuf[3 * i + 1] = g[i];
            obuf[3 * i + 2] = b[i];
        }
        out_file = fopen(filename, "wb");
        fprintf(out_file, "P6\n");
        fprintf(out_file, "%d %d\n255\n", w, h);
        fwrite(obuf, sizeof(unsigned char), 3 * w * h, out_file);
        fclose(out_file);
        free(obuf);
    }

//////////////////////
// Helper Functions //
//////////////////////

// Combine two images horizontally
    int combine_image(Image<unsigned char> &out_image,
                      const Image<unsigned char> &image1,
                      const Image<unsigned char> &image2)
    {
        int w1 = image1.w;
        int h1 = image1.h;
        int w2 = image2.w;
        int h2 = image2.h;
        int dstW = w1 + w2;
        int dstH = MAX(h1, h2);

        out_image.init(dstW, dstH);

        unsigned char *srcData1 = image1.data;
        unsigned char *srcData2 = image2.data;
        unsigned char *dstData = out_image.data;

        for (int r = 0; r < dstH; r++) {
            if (r < h1) {
                memcpy(dstData, srcData1, w1 * sizeof(unsigned char));
            }
            else {
                memset(dstData, 0, w1 * sizeof(unsigned char));
            }
            dstData += w1;

            if (r < h2) {
                memcpy(dstData, srcData2, w2 * sizeof(unsigned char));
            }
            else {
                memset(dstData, 0, w2 * sizeof(unsigned char));
            }
            dstData += w2;
            srcData1 += w1;
            srcData2 += w2;
        }

        return 0;
    }

// Helper callback function for merge match list.
    bool same_match_pair(const MatchPair &first, const MatchPair &second)
    {
        if (first.c1 == second.c1 && first.r1 == second.r1 &&
            first.c2 == second.c2 && first.r2 == second.r2)
            return true;
        else
            return false;
    }

// Match keypoints from two images, using brutal force method.
// Use Euclidean distance as matching score.
    int match_keypoints(std::list<SiftKeypoint> &kpt_list1,
                        std::list<SiftKeypoint> &kpt_list2,
                        std::list<MatchPair> &match_list)
    {
        std::list<SiftKeypoint>::iterator kpt1;
        std::list<SiftKeypoint>::iterator kpt2;

        for (kpt1 = kpt_list1.begin(); kpt1 != kpt_list1.end(); kpt1++) {
            // Position of the matched feature.
            int r1 = (int)kpt1->r;
            int c1 = (int)kpt1->c;

            float *descr1 = kpt1->descriptors;
            float score1 = (std::numeric_limits<float>::max)(); // highest score
            float score2 = (std::numeric_limits<float>::max)(); // 2nd highest score

            // Position of the matched feature.
            int r2 = 0, c2 = 0;
            for (kpt2 = kpt_list2.begin(); kpt2 != kpt_list2.end(); kpt2++) {
                float score = 0;
                float *descr2 = kpt2->descriptors;
                float dif;
                for (int i = 0; i < DEGREE_OF_DESCRIPTORS; i++) {
                    dif = descr1[i] - descr2[i];
                    score += dif * dif;
                }

                if (score < score1) {
                    score2 = score1;
                    score1 = score;
                    r2 = (int)kpt2->r;
                    c2 = (int)kpt2->c;
                }
                else if (score < score2) {
                    score2 = score;
                }
            }

#if (USE_FAST_FUNC == 1)
            if (fast_sqrt_f(score1 / score2) < SIFT_MATCH_NNDR_THR)
#else
                if (sqrtf(score1 / score2) < SIFT_MATCH_NNDR_THR)
#endif
            {
                MatchPair mp;
                mp.r1 = r1;
                mp.c1 = c1;
                mp.r2 = r2;
                mp.c2 = c2;

                match_list.push_back(mp);
            }
        }

        match_list.unique(same_match_pair);

#if PRINT_MATCH_KEYPOINTS
        std::list<MatchPair>::iterator p;
        int match_idx = 0;
        for (p = match_list.begin(); p != match_list.end(); p++) {
            printf("\tMatch %3d: (%4d, %4d) -> (%4d, %4d)\n", match_idx, p->r1,
                   p->c1, p->r2, p->c2);
            match_idx++;
        }
#endif

        return 0;
    }

    void draw_keypoints_to_ppm_file(const char *out_filename,
                                    const Image<unsigned char> &image,
                                    std::list<SiftKeypoint> kpt_list)
    {
        std::list<SiftKeypoint>::iterator it;
        ImagePPM imgPPM;
        int w = image.w;
        int h = image.h;
        int r, c;

        /*******************************
         * cR:
         * radius of the circle
         * cR = sigma * 4 * (2^O)
         *******************************/
        int cR;

        // initialize the imgPPM
        imgPPM.w = w;
        imgPPM.h = h;
        imgPPM.img_r = new unsigned char[w * h];
        imgPPM.img_g = new unsigned char[w * h];
        imgPPM.img_b = new unsigned char[w * h];

        int i, j;
        unsigned char *data = image.data;
        // Copy gray PGM images to color PPM images
        for (i = 0; i < h; i++) {
            for (j = 0; j < w; j++) {
                imgPPM.img_r[i * w + j] = data[i * w + j];
                imgPPM.img_g[i * w + j] = data[i * w + j];
                imgPPM.img_b[i * w + j] = data[i * w + j];
            }
        }

        for (it = kpt_list.begin(); it != kpt_list.end(); it++) {
            // derive circle radius cR
            cR = (int)it->scale;
            if (cR <= 1) { // avoid zero radius
                cR = 1;
            }
            r = (int)it->r;
            c = (int)it->c;
            //  draw_red_circle(&imgPPM, r, c, cR);
            rasterCircle(&imgPPM, r, c, cR);
            rasterCircle(&imgPPM, r, c, cR + 1);
            float ori = it->ori;
            draw_red_orientation(&imgPPM, c, r, ori, cR);
        }

        // write rendered image to output
        write_rgb2ppm(out_filename, imgPPM.img_r, imgPPM.img_g, imgPPM.img_b, w, h);

        // free allocated memory
        delete[] imgPPM.img_r;
        delete[] imgPPM.img_g;
        delete[] imgPPM.img_b;
        imgPPM.img_r = imgPPM.img_g = imgPPM.img_b = NULL;
    }

    int export_kpt_list_to_file(const char *filename,
                                std::list<SiftKeypoint> &kpt_list,
                                bool bIncludeDescpritor)
    {
        FILE *fp;
        fp = fopen(filename, "wb");
        if (!fp) {
            printf("Fail to open file: %s\n", filename);
            return -1;
        }

        fprintf(fp, "%u\t%d\n", static_cast<unsigned int>(kpt_list.size()), 128);

        std::list<SiftKeypoint>::iterator it;
        for (it = kpt_list.begin(); it != kpt_list.end(); it++) {
            fprintf(fp, "%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t", it->octave, it->layer,
                    it->r, it->c, it->scale, it->ori);
            if (bIncludeDescpritor) {
                for (int i = 0; i < 128; i++) {
                    fprintf(fp, "%d\t", (int)(it->descriptors[i]));
                }
            }
            fprintf(fp, "\n");
        }

        fclose(fp);
        return 0;
    }

// Draw a while line on a gray-scale image.
// MatchPair mp indicates the coordinates of the start point
// and the end point.
    int draw_line_to_image(Image<unsigned char> &image, MatchPair &mp)
    {
        int w = image.w;
        int r1 = mp.r1;
        int c1 = mp.c1;
        int r2 = mp.r2;
        int c2 = mp.c2 + w / 2;

        float k = (float)(r2 - r1) / (float)(c2 - c1);
        for (int c = c1; c < c2; c++) {
            // Line equation
            int r = (int)(k * (c - c1) + r1);
            image.data[r * w + c] = 190; // Draw a gray line.
        }
        return 0;
    }

// Draw a line on the RGB color image.
    int draw_line_to_rgb_image(unsigned char *&data, int w, int h, MatchPair &mp)
    {
        int r1 = mp.r1;
        int c1 = mp.c1;
        int r2 = mp.r2;
        int c2 = mp.c2;

        float k = (float)(r2 - r1) / (float)(c2 - c1);
        for (int c = c1; c < c2; c++) {
            // Line equation
            int r = (int)(k * (c - c1) + r1);

            // Draw a blue line
            data[r * w * 3 + 3 * c] = 0;
            data[r * w * 3 + 3 * c + 1] = 0;
            data[r * w * 3 + 3 * c + 2] = 255;
        }

        return 0;
    }

// Draw match lines between matched keypoints between two images.
    int draw_match_lines_to_ppm_file(const char *filename,
                                     Image<unsigned char> &image1,
                                     Image<unsigned char> &image2,
                                     std::list<MatchPair> &match_list)
    {
        Image<unsigned char> tmpImage;
        combine_image(tmpImage, image1, image2);

        int w = tmpImage.w;
        int h = tmpImage.h;
        unsigned char *srcData = tmpImage.data;
        unsigned char *dstData = new unsigned char[w * h * 3];

        for (int i = 0; i < w * h; i++) {
            dstData[i * 3 + 0] = srcData[i];
            dstData[i * 3 + 1] = srcData[i];
            dstData[i * 3 + 2] = srcData[i];
        }

        std::list<MatchPair>::iterator p;
        for (p = match_list.begin(); p != match_list.end(); p++) {
            MatchPair mp;
            mp.r1 = p->r1;
            mp.c1 = p->c1;
            mp.r2 = p->r2;
            mp.c2 = p->c2 + image1.w;
            draw_line_to_rgb_image(dstData, w, h, mp);
        }

        write_ppm(filename, dstData, w, h);

        delete[] dstData;
        dstData = NULL;
        return 0;
    }

// Member function definition
    template <typename T>
    Image<T>::Image()
    {
        w = 0;
        h = 0;
        data = NULL;
    }

    template <typename T>
    Image<T>::Image(int _w, int _h)
    {
        w = _w;
        h = _h;
        data = new T[w * h];
    }

// Copy construction function
    template <typename T>
    Image<T>::Image(const Image<T> &img)
    {
        w = img.w;
        h = img.h;
        data = new T[w * h];
        memcpy(data, img.data, w * h * sizeof(T));
    }

    template <typename T>
    Image<T>::~Image()
    {
        if (data) {
            delete[] data;
            data = NULL;
        }
    }

    template <typename T>
    Image<T> &Image<T>::operator=(const Image<T> &img)
    {
        init(img.w, img.h);
        memcpy(data, img.data, img.w * img.h * sizeof(T));
        return *this;
    }

    template <typename T>
    void Image<T>::init(int _w, int _h)
    {
        w = _w;
        h = _h;
        data = new T[w * h];
    }

    template <typename T>
    void Image<T>::reinit(int _w, int _h)
    {
        w = _w;
        h = _h;
        if (data) {
            delete[] data;
        }
        data = new T[w * h];
    }

    template <typename T>
    void Image<T>::release()
    {
        w = 0;
        h = 0;
        if (data) {
            delete[] data;
            data = NULL;
        }
    }

    template <typename T>
    int Image<T>::read_pgm(const char *filename)
    {
        FILE *in_file;
        char ch, type;
        int dummy;
        unsigned char *_data;

        in_file = fopen(filename, "rb");
        if (!in_file) {
            fprintf(stderr, "ERROR(0): Fail to open file %s\n", filename);
            return -1;
        }
        // Determine pgm image type (only type three can be used)
        ch = getc(in_file);
        if (ch != 'P') {
            printf("ERROR(1): Not valid pgm/ppm file type\n");
            return -1;
        }
        ch = getc(in_file);
        // Convert the one digit integer currently represented as a character to
        // an integer(48 == '0')
        type = ch - 48;
        if (type != 5) {
            printf("ERROR(2): this file type (P%d) is not supported!\n", type);
            return -1;
        }
        while (getc(in_file) != '\n')
            ;                          // Skip to end of line
        while (getc(in_file) == '#') { // Skip comment lines
            while (getc(in_file) != '\n')
                ;
        }
        fseek(in_file, -1, SEEK_CUR); // Backup one character

        fscanf(in_file, "%d", &w);
        fscanf(in_file, "%d", &h);
        fscanf(in_file, "%d", &dummy); // Skipped here
        while (getc(in_file) != '\n')
            ;

        init(w, h);
        if (typeid(T) == typeid(unsigned char)) {
            fread(data, sizeof(unsigned char), (w) * (h), in_file);
        }
        else {
            _data = new unsigned char[w * h];
            fread(_data, sizeof(unsigned char), (w) * (h), in_file);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    data[i * w + j] = (T)(_data[i * w + j]);
                }
            }
            delete[] _data;
        }

        return 0;
    }

    template <typename T>
    int Image<T>::write_pgm(const char *filename)
    {
        FILE *out_file;

        if (w <= 0 || h <= 0) {
            fprintf(stderr, "write_pgm(%s):Invalid image width or height\n",
                    filename);
            return -1;
        }

        out_file = fopen(filename, "wb");
        if (!out_file) {
            fprintf(stderr, "Fail to open file: %s\n", filename);
            return -1;
        }

        fprintf(out_file, "P5\n");
        fprintf(out_file, "%d %d\n255\n", w, h);

        Image<unsigned char> tmpImage;
        tmpImage = this->to_uchar();
        fwrite(tmpImage.data, sizeof(unsigned char), w * h, out_file);

        fclose(out_file);
        return 0;
    }

    template <typename T>
    Image<unsigned char> Image<T>::to_uchar() const
    {
        Image<unsigned char> dstImage(w, h);

        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                // Negative number, truncate to zero.
                float temp = data[r * w + c];
                dstImage.data[r * w + c] = temp >= 0 ? (unsigned char)temp : 0;
            }
        }
        return dstImage;
    }

    template <typename T>
    Image<float> Image<T>::to_float() const
    {
        Image<float> dstImage(w, h);

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                dstImage.data[i * w + j] = (float)this->data[i * w + j];
            }
        }
        return dstImage;
    }

// Upsample the image by 2x, linear interpolation.
    template <typename T>
    Image<T> Image<T>::upsample_2x() const
    {
        float scale = 2.0f;

        int srcW = w, srcH = h;
        int dstW = srcW << 1, dstH = srcH << 1;
        Image<T> out_image(dstW, dstH);

        T *srcData = data;
        T *dstData = out_image.data;

        for (int r = 0; r < dstH; r++) {
            for (int c = 0; c < dstW; c++) {
                float ori_r = r / scale;
                float ori_c = c / scale;
                int r1 = (int)ori_r;
                int c1 = (int)ori_c;
                float dr = ori_r - r1;
                float dc = ori_c - c1;

                int idx = r1 * srcW + c1;
                dstData[r * dstW + c] =
                        (unsigned char)((1 - dr) * (1 - dc) * srcData[idx] +
                                        dr * (1 - dc) *
                                        (r1 < srcH - 1 ? srcData[idx + srcW]
                                                       : srcData[idx]) +
                                        (1 - dr) * dc *
                                        (c1 < srcW - 1 ? srcData[idx + 1]
                                                       : srcData[idx]) +
                                        dr * dc *
                                        ((c1 < srcW - 1 && r1 < srcH - 1)
                                         ? srcData[idx + srcW + 1]
                                         : srcData[idx]));
            }
        }
        return out_image;
    }

// Downsample the image by 2x, nearest neighbor interpolation.
    template <typename T>
    Image<T> Image<T>::downsample_2x() const
    {
        int srcW = w, srcH = h;
        int dstW = srcW >> 1, dstH = srcH >> 1;
        Image<T> out_image(dstW, dstH);

        T *srcData = data;
        T *dstData = out_image.data;

        for (int r = 0; r < dstH; r++) {
            for (int c = 0; c < dstW; c++) {
                int ori_r = r << 1;
                int ori_c = c << 1;
                dstData[r * dstW + c] = srcData[ori_r * srcW + ori_c];
            }
        }
        return out_image;
    }

// Init sift parameters
    void init_sift_parameters(bool doubleFirstOctave, float contrast_threshold,
                              float edge_threshold, float match_NDDR_threshold)
    {
        SIFT_IMG_DBL = doubleFirstOctave;
        SIFT_CONTR_THR = contrast_threshold;
        SIFT_CURV_THR = edge_threshold;
        SIFT_MATCH_NNDR_THR = match_NDDR_threshold;
    }

// Set up first Octave
// doubleFirstOctave = true, firstOcative=-1;
// doubleFirstOctave = false, firstOcative=0;
    void double_original_image(bool doubleFirstOctave)
    {
        SIFT_IMG_DBL = doubleFirstOctave;
        return;
    }

// Compute octaves to build Gaussian Pyramid.
    int build_octaves(const Image<unsigned char> &image,
                      std::vector<Image<unsigned char> > &octaves, int firstOctave,
                      int nOctaves)
    {
        int w = image.w;
        int h = image.h;
        if (firstOctave == -1) {
            w = image.w * 2;
            h = image.h * 2;
        }

        for (int i = 0; i < nOctaves; i++) {
            if (i == 0 && firstOctave == -1) {
                octaves[i] = image.upsample_2x();
            }
            else if ((i == 0 && firstOctave != -1) ||
                     (i == 1 && firstOctave == -1)) {
                octaves[i] = image;
            }
            else {
                octaves[i] = octaves[(i - 1)].downsample_2x();
            }
            w = w / 2;
            h = h / 2;
        }
        return 0;
    }

// Apply Gaussian row filter to image, then transpose the image.
    int row_filter_transpose(float *src, float *dst, int w, int h, float *coef1d,
                             int gR)
    {
        float *row_buf = new float[w + gR * 2];
        float *row_start;
        int elemSize = sizeof(float);

        float *srcData = src;
        float *dstData = dst + w * h - 1;
        float partialSum = 0.0f;
        float *coef = coef1d;
        float *prow;

        float firstData, lastData;
        for (int r = 0; r < h; r++) {
            row_start = srcData + r * w;
            memcpy(row_buf + gR, row_start, elemSize * w);
            firstData = *(row_start);
            lastData = *(row_start + w - 1);
            for (int i = 0; i < gR; i++) {
                row_buf[i] = firstData;
                row_buf[i + w + gR] = lastData;
            }

            prow = row_buf;
            dstData = dstData - w * h + 1;
            for (int c = 0; c < w; c++) {
                partialSum = 0.0f;
                coef = coef1d;

                for (int i = -gR; i <= gR; i++) {
                    partialSum += (*coef++) * (*prow++);
                }

                prow -= 2 * gR;
                *dstData = partialSum;
                dstData += h;
            }
        }
        delete[] row_buf;
        row_buf = NULL;

        return 0;
    }

// Improved Gaussian Blurring Function
    int gaussian_blur(const Image<float> &in_image, Image<float> &out_image,
                      std::vector<float> coef1d)
    {
        int w = in_image.w;
        int h = in_image.h;
        int gR = static_cast<int>(coef1d.size()) / 2;

        Image<float> img_t(h, w);
        row_filter_transpose(in_image.data, img_t.data, w, h, &coef1d[0], gR);
        row_filter_transpose(img_t.data, out_image.data, h, w, &coef1d[0], gR);

        return 0;
    }

// For build_gaussian_pyramid()
    std::vector<std::vector<float> > compute_gaussian_coefs(int nOctaves,
                                                            int nGpyrLayers)
    {
        // Compute all sigmas for different layers
        int nLayers = nGpyrLayers - 3;
        float sigma, sigma_pre;
        float sigma0 = SIFT_SIGMA;
        float k = powf(2.0f, 1.0f / nLayers);

        std::vector<float> sig(nGpyrLayers);
        sigma_pre = SIFT_IMG_DBL ? 2.0f * SIFT_INIT_SIGMA : SIFT_INIT_SIGMA;
        sig[0] = sqrtf(sigma0 * sigma0 - sigma_pre * sigma_pre);
        for (int i = 1; i < nGpyrLayers; i++) {
            sigma_pre = powf(k, (float)(i - 1)) * sigma0;
            sigma = sigma_pre * k;
            sig[i] = sqrtf(sigma * sigma - sigma_pre * sigma_pre);
        }

        std::vector<std::vector<float> > gaussian_coefs(nGpyrLayers);
        for (int i = 0; i < nGpyrLayers; i++) {
            // Compute Gaussian filter coefficients
            float factor = SIFT_GAUSSIAN_FILTER_RADIUS;
            int gR = (sig[i] * factor > 1.0f) ? (int)ceilf(sig[i] * factor) : 1;
            int gW = gR * 2 + 1;

            gaussian_coefs[i].resize(gW);
            float accu = 0.0f;
            float tmp;
            for (int j = 0; j < gW; j++) {
                tmp = (float)((j - gR) / sig[i]);
                gaussian_coefs[i][j] = expf(tmp * tmp * -0.5f) * (1 + j / 1000.0f);
                accu += gaussian_coefs[i][j];
            }
            for (int j = 0; j < gW; j++) {
                gaussian_coefs[i][j] = gaussian_coefs[i][j] / accu;
            } // End compute Gaussian filter coefs
        }
        return gaussian_coefs;
    }

// Build Gaussian pyramid using recursive method.
// The first layer is downsampled from last octave, layer=nLayers.
// All n-th layer is Gaussian blur from (n-1)-th layer.
    int build_gaussian_pyramid(std::vector<Image<unsigned char> > &octaves,
                               std::vector<Image<float> > &gpyr, int nOctaves,
                               int nGpyrLayers)
    {
        int nLayers = nGpyrLayers - 3;
        std::vector<std::vector<float> > gaussian_coefs =
                compute_gaussian_coefs(nOctaves, nGpyrLayers);

        int w, h;
        for (int i = 0; i < nOctaves; i++) {
            w = octaves[i].w;
            h = octaves[i].h;
            for (int j = 0; j < nGpyrLayers; j++) {
                if (i == 0 && j == 0) {
                    gpyr[0].init(w, h);
                    gaussian_blur(octaves[0].to_float(), gpyr[0],
                                  gaussian_coefs[j]);
                }
                else if (i > 0 && j == 0) {
                    gpyr[i * nGpyrLayers] =
                            gpyr[(i - 1) * nGpyrLayers + nLayers].downsample_2x();
                }
                else {
                    gpyr[i * nGpyrLayers + j].init(w, h);
                    gaussian_blur(gpyr[i * nGpyrLayers + j - 1],
                                  gpyr[i * nGpyrLayers + j], gaussian_coefs[j]);
                }
            }
        }
        // Release octaves memory.
        octaves.clear();
        return 0;
    }

// Build difference of Gaussian pyramids.
    int build_dog_pyr(std::vector<Image<float> > &gpyr,
                      std::vector<Image<float> > &dogPyr, int nOctaves,
                      int nDogLayers)
    {
        int nGpyrLayers = nDogLayers + 1;

        int w, h;
        float *srcData1; // always data2-data1.
        float *srcData2;
        float *dstData;
        int index = 0;

        for (int i = 0; i < nOctaves; i++) {
            int row_start = i * nGpyrLayers;
            w = gpyr[row_start].w;
            h = gpyr[row_start].h;

            for (int j = 0; j < nDogLayers; j++) {
                dogPyr[i * nDogLayers + j].init(w, h);
                dstData = dogPyr[i * nDogLayers + j].data;

                srcData1 = gpyr[row_start + j].data;
                srcData2 = gpyr[row_start + j + 1].data;

                index = 0;
                while (index++ < w * h)
                    *(dstData++) = *(srcData2++) - *(srcData1++);
            }
        }

        return 0;
    }

// Build gradient pyramids.
    int build_grd_rot_pyr(std::vector<Image<float> > &gpyr,
                          std::vector<Image<float> > &grdPyr,
                          std::vector<Image<float> > &rotPyr, int nOctaves,
                          int nLayers)
    {
        int nGpyrLayers = nLayers + 3;
        int w, h;
        float dr, dc;
        float angle;

        float *srcData;
        float *grdData;
        float *rotData;

        for (int i = 0; i < nOctaves; i++) {
            // We only use gradient information from layers 1~n Layers.
            // Since keypoints only occur at these layers.

            w = gpyr[i * nGpyrLayers].w;
            h = gpyr[i * nGpyrLayers].h;
            for (int j = 1; j <= nLayers; j++) {
                int layer_index = i * nGpyrLayers + j;
                grdPyr[layer_index].init(w, h);
                rotPyr[layer_index].init(w, h);

                srcData = gpyr[layer_index].data;
                grdData = grdPyr[layer_index].data;
                rotData = rotPyr[layer_index].data;

                for (int r = 0; r < h; r++) {
                    for (int c = 0; c < w; c++) {
                        dr = get_pixel_f(srcData, w, h, r + 1, c) -
                             get_pixel_f(srcData, w, h, r - 1, c);
                        dc = get_pixel_f(srcData, w, h, r, c + 1) -
                             get_pixel_f(srcData, w, h, r, c - 1);

#if (USE_FAST_FUNC == 1)
                        grdData[r * w + c] = fast_sqrt_f(dr * dr + dc * dc);
                        angle = fast_atan2_f(dr, dc); // atan2f(dr, dc + FLT_MIN);
#else
                        grdData[r * w + c] = sqrtf(dr * dr + dc * dc);
                    angle = atan2f(dr, dc + FLT_MIN);
                    angle = angle < 0 ? angle + _2PI : angle;
#endif
                        rotData[r * w + c] = angle;
                    }
                }
            }
        }
        return 0;
    }

// Compute orientation histogram for keypoint detection.
// Gradient information is computed in this function.
    float compute_orientation_hist(const Image<float> &image, SiftKeypoint &kpt,
                                   float *&hist)
    {
        int nBins = SIFT_ORI_HIST_BINS;

        float kptr = kpt.ri;
        float kptc = kpt.ci;
        float kpt_scale = kpt.layer_scale;

        int kptr_i = (int)(kptr + 0.5f);
        int kptc_i = (int)(kptc + 0.5f);
        float d_kptr = kptr - kptr_i;
        float d_kptc = kptc - kptc_i;

        float sigma = SIFT_ORI_SIG_FCTR * kpt_scale;
        int win_radius = (int)(SIFT_ORI_RADIUS * kpt_scale);

        float *data = image.data;
        int w = image.w;
        int h = image.h;

        int r, c;
        float dr, dc;
        float magni, angle, weight;
        int bin;
        float fbin; // float point bin

        float *tmpHist = new float[nBins];
        memset(tmpHist, 0, nBins * sizeof(float));

        for (int i = -win_radius; i <= win_radius; i++) // rows
        {
            r = kptr_i + i;
            if (r <= 0 || r >= h - 1) // Cannot calculate dy
                continue;
            for (int j = -win_radius; j <= win_radius; j++) // columns
            {
                c = kptc_i + j;
                if (c <= 0 || c >= w - 1)
                    continue;

                dr = get_pixel_f(data, w, h, r + 1, c) -
                     get_pixel_f(data, w, h, r - 1, c);
                dc = get_pixel_f(data, w, h, r, c + 1) -
                     get_pixel_f(data, w, h, r, c - 1);

#if (USE_FAST_FUNC == 1)
                magni = fast_sqrt_f(dr * dr + dc * dc);
                angle = fast_atan2_f(dr, dc); // Unit: degree, range: [-PI, PI]
#else
                magni = sqrtf(dr * dr + dc * dc);
            angle = atan2f(dr, dc); // Unit: degree, range: [-PI, PI]
            angle = angle < 0 ? angle + _2PI : angle;
#endif
                fbin = angle * nBins / _2PI;
                weight = expf(
                        -1.0f *
                        ((i - d_kptr) * (i - d_kptr) + (j - d_kptc) * (j - d_kptc)) /
                        (2.0f * sigma * sigma));

#define SIFT_ORI_BILINEAR
#ifdef SIFT_ORI_BILINEAR
                bin = (int)(fbin - 0.5f);
                float d_fbin = fbin - 0.5f - bin;
                tmpHist[(bin + nBins) % nBins] += (1 - d_fbin) * magni * weight;
                tmpHist[(bin + 1) % nBins] += d_fbin * magni * weight;
#else
                bin = (int)(fbin);
            tmpHist[bin] += magni * weight;
#endif
            }
        }

#define TMPHIST(idx)                                                           \
    (idx < 0 ? tmpHist[0] : (idx >= nBins ? tmpHist[nBins - 1] : tmpHist[idx]))

#define USE_SMOOTH1 1
#if USE_SMOOTH1
        // Smooth the histogram. Algorithm comes from OpenCV.
        for (int i = 0; i < nBins; i++) {
            hist[i] = (TMPHIST(i - 2) + TMPHIST(i + 2)) * 1.0f / 16.0f +
                      (TMPHIST(i - 1) + TMPHIST(i + 1)) * 4.0f / 16.0f +
                      TMPHIST(i) * 6.0f / 16.0f;
        }
#else
        // Yet another smooth function
    // Smoothing algorithm comes from vl_feat implementation.
    for (int iter = 0; iter < 6; iter++) {
        float prev = TMPHIST(nBins - 1);
        float first = TMPHIST(0);
        int i;
        for (i = 0; i < nBins - 1; i++) {
            float newh = (prev + TMPHIST(i) + TMPHIST(i + 1)) / 3.0f;
            prev = hist[i];
            hist[i] = newh;
        }
        hist[i] = (prev + hist[i] + first) / 3.0f;
    }
#endif

        // Find the maximum item of the histogram
        float maxitem = hist[0];
        int max_i = 0;
        for (int i = 0; i < nBins; i++) {
            if (maxitem < hist[i]) {
                maxitem = hist[i];
                max_i = i;
            }
        }

        kpt.ori = max_i * _2PI / nBins;

        delete[] tmpHist;
        tmpHist = NULL;
        return maxitem;
    }

// Compute orientation histogram for keypoint detection.
// using pre-computed gradient information.
    float compute_orientation_hist_with_gradient(const Image<float> &grdImage,
                                                 const Image<float> &rotImage,
                                                 SiftKeypoint &kpt, float *&hist)
    {
        int nBins = SIFT_ORI_HIST_BINS;

        float kptr = kpt.ri;
        float kptc = kpt.ci;
        float kpt_scale = kpt.layer_scale;

        int kptr_i = (int)(kptr + 0.5f);
        int kptc_i = (int)(kptc + 0.5f);
        float d_kptr = kptr - kptr_i;
        float d_kptc = kptc - kptc_i;

        float sigma = SIFT_ORI_SIG_FCTR * kpt_scale;
        int win_radius = (int)(SIFT_ORI_RADIUS * kpt_scale);
        float exp_factor = -1.0f / (2.0f * sigma * sigma);

        float *grdData = grdImage.data;
        float *rotData = rotImage.data;
        int w = grdImage.w;
        int h = grdImage.h;

        int r, c;
        float magni, angle, weight;
        int bin;
        float fbin; // float point bin

        float *tmpHist = new float[nBins];
        memset(tmpHist, 0, nBins * sizeof(float));

        for (int i = -win_radius; i <= win_radius; i++) // rows
        {
            r = kptr_i + i;
            if (r <= 0 || r >= h - 1) // Cannot calculate dy
                continue;
            for (int j = -win_radius; j <= win_radius; j++) // columns
            {
                c = kptc_i + j;
                if (c <= 0 || c >= w - 1)
                    continue;

                magni = grdData[r * w + c];
                angle = rotData[r * w + c];

                fbin = angle * nBins / _2PI;
                weight = expf(
                        ((i - d_kptr) * (i - d_kptr) + (j - d_kptc) * (j - d_kptc)) *
                        exp_factor);

#define SIFT_ORI_BILINEAR
#ifdef SIFT_ORI_BILINEAR
                bin = (int)(fbin - 0.5f);
                float d_fbin = fbin - 0.5f - bin;

                float mw = weight * magni;
                float dmw = d_fbin * mw;
                tmpHist[(bin + nBins) % nBins] += mw - dmw;
                tmpHist[(bin + 1) % nBins] += dmw;
#else
                bin = (int)(fbin);
            tmpHist[bin] += magni * weight;
#endif
            }
        }

#define TMPHIST(idx)                                                           \
    (idx < 0 ? tmpHist[0] : (idx >= nBins ? tmpHist[nBins - 1] : tmpHist[idx]))

#define USE_SMOOTH1 1
#if USE_SMOOTH1

        // Smooth the histogram. Algorithm comes from OpenCV.
        hist[0] = (tmpHist[0] + tmpHist[2]) * 1.0f / 16.0f +
                  (tmpHist[0] + tmpHist[1]) * 4.0f / 16.0f +
                  tmpHist[0] * 6.0f / 16.0f;
        hist[1] = (tmpHist[0] + tmpHist[3]) * 1.0f / 16.0f +
                  (tmpHist[0] + tmpHist[2]) * 4.0f / 16.0f +
                  tmpHist[1] * 6.0f / 16.0f;
        hist[nBins - 2] = (tmpHist[nBins - 4] + tmpHist[nBins - 1]) * 1.0f / 16.0f +
                          (tmpHist[nBins - 3] + tmpHist[nBins - 1]) * 4.0f / 16.0f +
                          tmpHist[nBins - 2] * 6.0f / 16.0f;
        hist[nBins - 1] = (tmpHist[nBins - 3] + tmpHist[nBins - 1]) * 1.0f / 16.0f +
                          (tmpHist[nBins - 2] + tmpHist[nBins - 1]) * 4.0f / 16.0f +
                          tmpHist[nBins - 1] * 6.0f / 16.0f;

        for (int i = 2; i < nBins - 2; i++) {
            hist[i] = (tmpHist[i - 2] + tmpHist[i + 2]) * 1.0f / 16.0f +
                      (tmpHist[i - 1] + tmpHist[i + 1]) * 4.0f / 16.0f +
                      tmpHist[i] * 6.0f / 16.0f;
        }

#else
        // Yet another smooth function
    // Algorithm comes from the vl_feat implementation.
    for (int iter = 0; iter < 6; iter++) {
        float prev = TMPHIST(nBins - 1);
        float first = TMPHIST(0);
        int i;
        for (i = 0; i < nBins - 1; i++) {
            float newh = (prev + TMPHIST(i) + TMPHIST(i + 1)) / 3.0f;
            prev = hist[i];
            hist[i] = newh;
        }
        hist[i] = (prev + hist[i] + first) / 3.0f;
    }
#endif

        // Find the maximum item of the histogram
        float maxitem = hist[0];
        int max_i = 0;
        for (int i = 0; i < nBins; i++) {
            if (maxitem < hist[i]) {
                maxitem = hist[i];
                max_i = i;
            }
        }

        kpt.ori = max_i * _2PI / nBins;

        delete[] tmpHist;
        tmpHist = NULL;
        return maxitem;
    }

// Refine local keypoint extrema.
    bool refine_local_extrema(std::vector<Image<float> > &dogPyr, int nOctaves,
                              int nDogLayers, SiftKeypoint &kpt)
    {
        int nGpyrLayers = nDogLayers + 1;

        int w = 0, h = 0;
        int layer_idx = 0;
        int octave = kpt.octave;
        int layer = kpt.layer;
        int r = (int)kpt.ri;
        int c = (int)kpt.ci;

        float *currData = NULL;
        float *lowData = NULL;
        float *highData = NULL;

        int xs_i = 0, xr_i = 0, xc_i = 0;
        float tmp_r = 0.0f, tmp_c = 0.0f, tmp_layer = 0.0f;
        float xr = 0.0f, xc = 0.0f, xs = 0.0f;
        float x_hat[3] = {xc, xr, xs};
        float dx = 0.0f, dy = 0.0f, ds = 0.0f;
        float dxx = 0.0f, dyy = 0.0f, dss = 0.0f, dxs = 0.0f, dys = 0.0f,
                dxy = 0.0f;

        tmp_r = (float)r;
        tmp_c = (float)c;
        tmp_layer = (float)layer;

        // Interpolation (x,y,sigma) 3D space to find sub-pixel accurate
        // location of keypoints.
        int i = 0;
        for (; i < SIFT_MAX_INTERP_STEPS; i++) {
            c += xc_i;
            r += xr_i;

            layer_idx = octave * nDogLayers + layer;
            w = dogPyr[layer_idx].w;
            h = dogPyr[layer_idx].h;
            currData = dogPyr[layer_idx].data;
            lowData = dogPyr[layer_idx - 1].data;
            highData = dogPyr[layer_idx + 1].data;

            dx = (get_pixel_f(currData, w, h, r, c + 1) -
                  get_pixel_f(currData, w, h, r, c - 1)) *
                 0.5f;
            dy = (get_pixel_f(currData, w, h, r + 1, c) -
                  get_pixel_f(currData, w, h, r - 1, c)) *
                 0.5f;
            ds = (get_pixel_f(highData, w, h, r, c) -
                  get_pixel_f(lowData, w, h, r, c)) *
                 0.5f;
            float dD[3] = {-dx, -dy, -ds};

            float v2 = 2.0f * get_pixel_f(currData, w, h, r, c);
            dxx = (get_pixel_f(currData, w, h, r, c + 1) +
                   get_pixel_f(currData, w, h, r, c - 1) - v2);
            dyy = (get_pixel_f(currData, w, h, r + 1, c) +
                   get_pixel_f(currData, w, h, r - 1, c) - v2);
            dss = (get_pixel_f(highData, w, h, r, c) +
                   get_pixel_f(lowData, w, h, r, c) - v2);
            dxy = (get_pixel_f(currData, w, h, r + 1, c + 1) -
                   get_pixel_f(currData, w, h, r + 1, c - 1) -
                   get_pixel_f(currData, w, h, r - 1, c + 1) +
                   get_pixel_f(currData, w, h, r - 1, c - 1)) *
                  0.25f;
            dxs = (get_pixel_f(highData, w, h, r, c + 1) -
                   get_pixel_f(highData, w, h, r, c - 1) -
                   get_pixel_f(lowData, w, h, r, c + 1) +
                   get_pixel_f(lowData, w, h, r, c - 1)) *
                  0.25f;
            dys = (get_pixel_f(highData, w, h, r + 1, c) -
                   get_pixel_f(highData, w, h, r - 1, c) -
                   get_pixel_f(lowData, w, h, r + 1, c) +
                   get_pixel_f(lowData, w, h, r - 1, c)) *
                  0.25f;

            // The scale in two sides of the equation should cancel each other.
            float H[3][3] = {{dxx, dxy, dxs}, {dxy, dyy, dys}, {dxs, dys, dss}};
            float Hinvert[3][3];
            float det;

            // Matrix inversion
            // INVERT_3X3 = DETERMINANT_3X3, then SCALE_ADJOINT_3X3;
            // Using INVERT_3X3(Hinvert, det, H) is more convenient;
            // but using separate ones, we can check det==0 easily.
            float tmp;
            DETERMINANT_3X3(det, H);
            if (fabsf(det) < (std::numeric_limits<float>::min)())
                break;
            tmp = 1.0f / (det);
            // INVERT_3X3(Hinvert, det, H);
            SCALE_ADJOINT_3X3(Hinvert, tmp, H);
            MAT_DOT_VEC_3X3(x_hat, Hinvert, dD);

            xs = x_hat[2];
            xr = x_hat[1];
            xc = x_hat[0];

            // Update tmp data for keypoint update.
            tmp_r = r + xr;
            tmp_c = c + xc;
            tmp_layer = layer + xs;

            // Make sure there is room to move for next iteration.
            xc_i = ((xc >= SIFT_KEYPOINT_SUBPiXEL_THR && c < w - 2) ? 1 : 0) +
                   ((xc <= -SIFT_KEYPOINT_SUBPiXEL_THR && c > 1) ? -1 : 0);

            xr_i = ((xr >= SIFT_KEYPOINT_SUBPiXEL_THR && r < h - 2) ? 1 : 0) +
                   ((xr <= -SIFT_KEYPOINT_SUBPiXEL_THR && r > 1) ? -1 : 0);

            if (xc_i == 0 && xr_i == 0 && xs_i == 0)
                break;
        }

        // We MIGHT be able to remove the following two checking conditions.
        // Condition 1
        if (i >= SIFT_MAX_INTERP_STEPS)
            return false;
        // Condition 2.
        if (fabsf(xc) >= 1.5 || fabsf(xr) >= 1.5 || fabsf(xs) >= 1.5)
            return false;

        // If (r, c, layer) is out of range, return false.
        if (tmp_layer < 0 || tmp_layer > (nGpyrLayers - 1) || tmp_r < 0 ||
            tmp_r > h - 1 || tmp_c < 0 || tmp_c > w - 1)
            return false;

        {
            float value = get_pixel_f(currData, w, h, r, c) +
                          0.5f * (dx * xc + dy * xr + ds * xs);
            if (fabsf(value) < SIFT_CONTR_THR)
                return false;

            float trH = dxx + dyy;
            float detH = dxx * dyy - dxy * dxy;
            float response =
                    (SIFT_CURV_THR + 1) * (SIFT_CURV_THR + 1) / (SIFT_CURV_THR);

            if (detH <= 0 || (trH * trH / detH) >= response)
                return false;
        }

        // Coordinates in the current layer.
        kpt.ci = tmp_c;
        kpt.ri = tmp_r;
        kpt.layer_scale = SIFT_SIGMA * powf(2.0f, tmp_layer / SIFT_INTVLS);

        int firstOctave = SIFT_IMG_DBL ? -1 : 0;
        float norm = powf(2.0f, (float)(octave + firstOctave));
        // Coordinates in the normalized format (compared to the original image).
        kpt.c = tmp_c * norm;
        kpt.r = tmp_r * norm;
        kpt.rlayer = tmp_layer;
        kpt.layer = layer;

        // Formula: Scale = sigma0 * 2^octave * 2^(layer/S);
        kpt.scale = kpt.layer_scale * norm;

        return true;
    }

// Keypoint detection.
    int detect_keypoints(std::vector<Image<float> > &dogPyr,
                         std::vector<Image<float> > &grdPyr,
                         std::vector<Image<float> > &rotPyr, int nOctaves,
                         int nDogLayers, std::list<SiftKeypoint> &kpt_list)
    {
        float *currData;
        float *lowData;
        float *highData;

        SiftKeypoint kpt;

        int w, h;
        int layer_index;
        int index;
        float val;

        int nBins = SIFT_ORI_HIST_BINS;
        float *hist = new float[nBins];
        int nGpyrLayers = nDogLayers + 1;

        // Some paper uses other thresholds, for example 3.0f for all cases
        // In Lowe's paper, |D(x)|<0.03 will be rejected.
        float contr_thr = 0.8f * SIFT_CONTR_THR;

        for (int i = 0; i < nOctaves; i++) {
            w = dogPyr[i * nDogLayers].w;
            h = dogPyr[i * nDogLayers].h;

            for (int j = 1; j < nDogLayers - 1; j++) {
                layer_index = i * nDogLayers + j;

                highData = dogPyr[layer_index + 1].data;
                currData = dogPyr[layer_index].data;
                lowData = dogPyr[layer_index - 1].data;

                for (int r = SIFT_IMG_BORDER; r < h - SIFT_IMG_BORDER; r++) {
                    for (int c = SIFT_IMG_BORDER; c < w - SIFT_IMG_BORDER; c++) {
                        index = r * w + c;
                        val = currData[index];

                        bool bExtrema =
                                (val >= contr_thr && val > highData[index - w - 1] &&
                                 val > highData[index - w] &&
                                 val > highData[index - w + 1] &&
                                 val > highData[index - 1] && val > highData[index] &&
                                 val > highData[index + 1] &&
                                 val > highData[index + w - 1] &&
                                 val > highData[index + w] &&
                                 val > highData[index + w + 1] &&
                                 val > currData[index - w - 1] &&
                                 val > currData[index - w] &&
                                 val > currData[index - w + 1] &&
                                 val > currData[index - 1] &&
                                 val > currData[index + 1] &&
                                 val > currData[index + w - 1] &&
                                 val > currData[index + w] &&
                                 val > currData[index + w + 1] &&
                                 val > lowData[index - w - 1] &&
                                 val > lowData[index - w] &&
                                 val > lowData[index - w + 1] &&
                                 val > lowData[index - 1] && val > lowData[index] &&
                                 val > lowData[index + 1] &&
                                 val > lowData[index + w - 1] &&
                                 val > lowData[index + w] &&
                                 val > lowData[index + w + 1]) || // Local min
                                (val <= -contr_thr && val < highData[index - w - 1] &&
                                 val < highData[index - w] &&
                                 val < highData[index - w + 1] &&
                                 val < highData[index - 1] && val < highData[index] &&
                                 val < highData[index + 1] &&
                                 val < highData[index + w - 1] &&
                                 val < highData[index + w] &&
                                 val < highData[index + w + 1] &&
                                 val < currData[index - w - 1] &&
                                 val < currData[index - w] &&
                                 val < currData[index - w + 1] &&
                                 val < currData[index - 1] &&
                                 val < currData[index + 1] &&
                                 val < currData[index + w - 1] &&
                                 val < currData[index + w] &&
                                 val < currData[index + w + 1] &&
                                 val < lowData[index - w - 1] &&
                                 val < lowData[index - w] &&
                                 val < lowData[index - w + 1] &&
                                 val < lowData[index - 1] && val < lowData[index] &&
                                 val < lowData[index + 1] &&
                                 val < lowData[index + w - 1] &&
                                 val < lowData[index + w] &&
                                 val < lowData[index + w + 1]);

                        if (bExtrema) {
                            kpt.octave = i;
                            kpt.layer = j;
                            kpt.ri = (float)r;
                            kpt.ci = (float)c;

                            bool bGoodKeypoint = refine_local_extrema(
                                    dogPyr, nOctaves, nDogLayers, kpt);

                            if (!bGoodKeypoint)
                                continue;

                            float max_mag = compute_orientation_hist_with_gradient(
                                    grdPyr[i * nGpyrLayers + kpt.layer],
                                    rotPyr[i * nGpyrLayers + kpt.layer], kpt, hist);

                            float threshold = max_mag * SIFT_ORI_PEAK_RATIO;

                            for (int ii = 0; ii < nBins; ii++) {
#define INTERPOLATE_ORI_HIST

#ifndef INTERPOLATE_ORI_HIST
                                if (hist[ii] >= threshold) {
                                kpt.mag = hist[ii];
                                kpt.ori = ii * _2PI / nBins;
                                kpt_list.push_back(kpt);
                            }
#else
                                // Use 3 points to fit a curve and find the accurate
                                // location of a keypoints
                                int left = ii > 0 ? ii - 1 : nBins - 1;
                                int right = ii < (nBins - 1) ? ii + 1 : 0;
                                float currHist = hist[ii];
                                float lhist = hist[left];
                                float rhist = hist[right];
                                if (currHist > lhist && currHist > rhist &&
                                    currHist > threshold) {
                                    // Refer to here:
                                    // http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
                                    float accu_ii =
                                            ii + 0.5f * (lhist - rhist) /
                                                 (lhist - 2.0f * currHist + rhist);

                                    // Since bin index means the starting point of a
                                    // bin, so the real orientation should be bin
                                    // index plus 0.5. for example, angles in bin 0
                                    // should have a mean value of 5 instead of 0;
                                    accu_ii += 0.5f;
                                    accu_ii = accu_ii < 0 ? (accu_ii + nBins)
                                                          : accu_ii >= nBins
                                                            ? (accu_ii - nBins)
                                                            : accu_ii;
                                    // The magnitude should also calculate the max
                                    // number based on fitting But since we didn't
                                    // actually use it in image matching, we just
                                    // lazily use the histogram value.
                                    kpt.mag = currHist;
                                    kpt.ori = accu_ii * _2PI / nBins;
                                    kpt_list.push_back(kpt);
                                }
#endif
                            }
                        }
                    }
                }
            }
        }

        delete[] hist;
        hist = NULL;
        return 0;
    }

// Extract descriptor
// 1. Unroll the tri-linear part.
    int extract_descriptor(std::vector<Image<float> > &grdPyr,
                           std::vector<Image<float> > &rotPyr, int nOctaves,
                           int nGpyrLayers, std::list<SiftKeypoint> &kpt_list)
    {
        // Number of subregions, default 4x4 subregions.
        // The width of subregion is determined by the scale of the keypoint.
        // Or, in Lowe's SIFT paper[2004], width of subregion is 16x16.
        int nSubregion = SIFT_DESCR_WIDTH;
        int nHalfSubregion = nSubregion >> 1;

        // Number of histogram bins for each descriptor subregion.
        int nBinsPerSubregion = SIFT_DESCR_HIST_BINS;
        float nBinsPerSubregionPerDegree = (float)nBinsPerSubregion / _2PI;

        // 3-D structure for histogram bins (rbin, cbin, obin);
        // (rbin, cbin, obin) means (row of hist bin, column of hist bin,
        // orientation bin) In Lowe's paper, 4x4 histogram, each has 8 bins. that
        // means for each (rbin, cbin), there are 8 bins in the histogram.

        // In this implementation, histBin is a circular buffer.
        // we expand the cube by 1 for each direction.
        int nBins = nSubregion * nSubregion * nBinsPerSubregion;
        int nHistBins =
                (nSubregion + 2) * (nSubregion + 2) * (nBinsPerSubregion + 2);
        int nSliceStep = (nSubregion + 2) * (nBinsPerSubregion + 2);
        int nRowStep = (nBinsPerSubregion + 2);
        float *histBin = new float[nHistBins];

        float exp_scale = -2.0f / (nSubregion * nSubregion);

        for (std::list<SiftKeypoint>::iterator kpt = kpt_list.begin();
             kpt != kpt_list.end(); kpt++) {
            // Keypoint information
            int octave = kpt->octave;
            int layer = kpt->layer;

            float kpt_ori = kpt->ori;
            float kptr = kpt->ri;
            float kptc = kpt->ci;
            float kpt_scale = kpt->layer_scale;

            // Nearest coordinate of keypoints
            int kptr_i = (int)(kptr + 0.5f);
            int kptc_i = (int)(kptc + 0.5f);
            float d_kptr = kptr_i - kptr;
            float d_kptc = kptc_i - kptc;

            int layer_index = octave * nGpyrLayers + layer;
            int w = grdPyr[layer_index].w;
            int h = grdPyr[layer_index].h;
            float *grdData = grdPyr[layer_index].data;
            float *rotData = rotPyr[layer_index].data;

            // Note for Gaussian weighting.
            // OpenCV and vl_feat uses non-fixed size of subregion.
            // But they all use (0.5 * 4) as the Gaussian weighting sigma.
            // In Lowe's paper, he uses 16x16 sample region,
            // partition 16x16 region into 16 4x4 subregion.
            float subregion_width = SIFT_DESCR_SCL_FCTR * kpt_scale;
            int win_size =
                    (int)(SQRT2 * subregion_width * (nSubregion + 1) * 0.5f + 0.5f);

            // Normalized cos() and sin() value.
            float sin_t = sinf(kpt_ori) / (float)subregion_width;
            float cos_t = cosf(kpt_ori) / (float)subregion_width;

            // Re-init histBin
            memset(histBin, 0, nHistBins * sizeof(float));

            // Start to calculate the histogram in the sample region.
            float rr, cc;
            float mag, angle, gaussian_weight;

            // Used for tri-linear interpolation.
            // int rbin_i, cbin_i, obin_i;
            float rrotate, crotate;
            float rbin, cbin, obin;
            float d_rbin, d_cbin, d_obin;

            // Boundary of sample region.
            int r, c;
            int left = MAX(-win_size, 1 - kptc_i);
            int right = MIN(win_size, w - 2 - kptc_i);
            int top = MAX(-win_size, 1 - kptr_i);
            int bottom = MIN(win_size, h - 2 - kptr_i);

            for (int i = top; i <= bottom; i++) // rows
            {
                for (int j = left; j <= right; j++) // columns
                {
                    // Accurate position relative to (kptr, kptc)
                    rr = i + d_kptr;
                    cc = j + d_kptc;

                    // Rotate the coordinate of (i, j)
                    rrotate = (cos_t * cc + sin_t * rr);
                    crotate = (-sin_t * cc + cos_t * rr);

                    // Since for a bin array with 4x4 bins, the center is actually
                    // at (1.5, 1.5)
                    rbin = rrotate + nHalfSubregion - 0.5f;
                    cbin = crotate + nHalfSubregion - 0.5f;

                    // rbin, cbin range is (-1, d); if outside this range, then the
                    // pixel is counted.
                    if (rbin <= -1 || rbin >= nSubregion || cbin <= -1 ||
                        cbin >= nSubregion)
                        continue;

                    // All the data need for gradient computation are valid, no
                    // border issues.
                    r = kptr_i + i;
                    c = kptc_i + j;
                    mag = grdData[r * w + c];
                    angle = rotData[r * w + c] - kpt_ori;
                    float angle1 = (angle < 0) ? (_2PI + angle)
                                               : angle; // Adjust angle to [0, 2PI)
                    obin = angle1 * nBinsPerSubregionPerDegree;

                    int x0, y0, z0;
                    int x1, y1, z1;
                    y0 = (int)floor(rbin);
                    x0 = (int)floor(cbin);
                    z0 = (int)floor(obin);
                    d_rbin = rbin - y0;
                    d_cbin = cbin - x0;
                    d_obin = obin - z0;
                    x1 = x0 + 1;
                    y1 = y0 + 1;
                    z1 = z0 + 1;

                    // Gaussian weight relative to the center of sample region.
                    gaussian_weight =
                            expf((rrotate * rrotate + crotate * crotate) * exp_scale);

                    // Gaussian-weighted magnitude
                    float gm = mag * gaussian_weight;
                    // Tri-linear interpolation

                    float vr1, vr0;
                    float vrc11, vrc10, vrc01, vrc00;
                    float vrco110, vrco111, vrco100, vrco101, vrco010, vrco011,
                            vrco000, vrco001;

                    vr1 = gm * d_rbin;
                    vr0 = gm - vr1;
                    vrc11 = vr1 * d_cbin;
                    vrc10 = vr1 - vrc11;
                    vrc01 = vr0 * d_cbin;
                    vrc00 = vr0 - vrc01;
                    vrco111 = vrc11 * d_obin;
                    vrco110 = vrc11 - vrco111;
                    vrco101 = vrc10 * d_obin;
                    vrco100 = vrc10 - vrco101;
                    vrco011 = vrc01 * d_obin;
                    vrco010 = vrc01 - vrco011;
                    vrco001 = vrc00 * d_obin;
                    vrco000 = vrc00 - vrco001;

                    // int idx =  y0  * nSliceStep + x0  * nRowStep + z0;
                    // All coords are offseted by 1. so x=[1, 4], y=[1, 4];
                    // data for -1 coord is stored at position 0;
                    // data for 8 coord is stored at position 9.
                    // z doesn't need to move.
                    int idx = y1 * nSliceStep + x1 * nRowStep + z0;
                    histBin[idx] += vrco000;

                    idx++;
                    histBin[idx] += vrco001;

                    idx += nRowStep - 1;
                    histBin[idx] += vrco010;

                    idx++;
                    histBin[idx] += vrco011;

                    idx += nSliceStep - nRowStep - 1;
                    histBin[idx] += vrco100;

                    idx++;
                    histBin[idx] += vrco101;

                    idx += nRowStep - 1;
                    histBin[idx] += vrco110;

                    idx++;
                    histBin[idx] += vrco111;
                }
            }

            // Discard all the edges for row and column.
            // Only retrive edges for orientation bins.
            float *dstBins = new float[nBins];
            for (int i = 1; i <= nSubregion; i++) // slice
            {
                for (int j = 1; j <= nSubregion; j++) // row
                {
                    int idx = i * nSliceStep + j * nRowStep;
                    // comments: how this line works.
                    // Suppose you want to write w=width, y=1, due to circular
                    // buffer, we should write it to w=0, y=1; since we use a
                    // circular buffer, it is written into w=width, y=1. Now, we
                    // fectch the data back.
                    histBin[idx] = histBin[idx + nBinsPerSubregion];

                    // comments: how this line works.
                    // Suppose you want to write x=-1 y=1, due to circular, it
                    // should be at y=1, x=width-1; since we use circular buffer,
                    // the value goes to y=0, x=width, now, we need to get it back.
                    if (idx != 0)
                        histBin[idx + nBinsPerSubregion + 1] = histBin[idx - 1];

                    int idx1 = ((i - 1) * nSubregion + j - 1) * nBinsPerSubregion;
                    for (int k = 0; k < nBinsPerSubregion; k++) {
                        dstBins[idx1 + k] = histBin[idx + k];
                    }
                }
            }

            // Normalize the histogram
            float sum_square = 0.0f;
            for (int i = 0; i < nBins; i++)
                sum_square += dstBins[i] * dstBins[i];

#if (USE_FAST_FUNC == 1)
            float thr = fast_sqrt_f(sum_square) * SIFT_DESCR_MAG_THR;
#else
            float thr = sqrtf(sum_square) * SIFT_DESCR_MAG_THR;
#endif

            float tmp = 0.0;
            sum_square = 0.0;
            // Cut off the numbers bigger than 0.2 after normalized.
            for (int i = 0; i < nBins; i++) {
                tmp = fmin(thr, dstBins[i]);
                dstBins[i] = tmp;
                sum_square += tmp * tmp;
            }

// Re-normalize
// The numbers are usually too small to store, so we use
// a constant factor to scale up the numbers.
#if (USE_FAST_FUNC == 1)
            float norm_factor = SIFT_INT_DESCR_FCTR / fast_sqrt_f(sum_square);
#else
            float norm_factor = SIFT_INT_DESCR_FCTR / sqrtf(sum_square);
#endif
            for (int i = 0; i < nBins; i++)
                dstBins[i] = dstBins[i] * norm_factor;

            memcpy(kpt->descriptors, dstBins, nBins * sizeof(float));

            if (dstBins) {
                delete[] dstBins;
                dstBins = NULL;
            }
        }

        if (histBin) {
            delete[] histBin;
            histBin = NULL;
        }

        return 0;
    }

    int sift_cpu(const Image<unsigned char> &image,
                 std::list<SiftKeypoint> &kpt_list, bool bExtractDescriptors)
    {
        // Index of the first octave.
        int firstOctave = (SIFT_IMG_DBL) ? -1 : 0;
        // Number of layers in one octave; same as s in the paper.
        int nLayers = SIFT_INTVLS;
        // Number of Gaussian images in one octave.
        int nGpyrLayers = nLayers + 3;
        // Number of DoG images in one octave.
        int nDogLayers = nLayers + 2;
        // Number of octaves according to the size of image.
        int nOctaves = (int)my_log2((float)fmin(image.w, image.h)) - 3 -
                       firstOctave; // 2 or 3, need further research

        // Build image octaves
        std::vector<Image<unsigned char> > octaves(nOctaves);
        build_octaves(image, octaves, firstOctave, nOctaves);

#if (DUMP_OCTAVE_IMAGE == 1)
        char foctave[256];
    for (int i = 0; i < nOctaves; i++) {
        sprintf(foctave, "octave_Octave-%d.pgm", i);
        write_pgm(foctave, octaves[i].data, octaves[i].w, octaves[i].h);
    }
#endif

        // Build Gaussian pyramid
        std::vector<Image<float> > gpyr(nOctaves * nGpyrLayers);
        build_gaussian_pyramid(octaves, gpyr, nOctaves, nGpyrLayers);

#if (DUMP_GAUSSIAN_PYRAMID_IMAGE == 1)
        char fgpyr[256];
    for (int i = 0; i < nOctaves; i++) {
        for (int j = 0; j < nGpyrLayers; j++) {
            sprintf(fgpyr, "gpyr-%d-%d.pgm", i, j);
            write_float_pgm(fgpyr, gpyr[i * nGpyrLayers + j].data,
                            gpyr[i * nGpyrLayers + j].w,
                            gpyr[i * nGpyrLayers + j].h, 1);
        }
    }
#endif

        // Build DoG pyramid
        std::vector<Image<float> > dogPyr(nOctaves * nDogLayers);
        build_dog_pyr(gpyr, dogPyr, nOctaves, nDogLayers);

#if (DUMP_DOG_IMAGE == 1)
        char fdog[256];
    Image<unsigned char> img_dog_t;
    for (int i = 0; i < nOctaves; i++) {
        for (int j = 0; j < nDogLayers; j++) {
            sprintf(fdog, "dog_Octave-%d_Layer-%d.pgm", i, j);
            img_dog_t = dogPyr[i * nDogLayers + j].to_unsigned char();
            write_pgm(fdog, img_dog_t.data, img_dog_t.w, img_dog_t.h);
        }
    }
#endif

        // Build gradient and rotation pyramids
        std::vector<Image<float> > grdPyr(nOctaves * nGpyrLayers);
        std::vector<Image<float> > rotPyr(nOctaves * nGpyrLayers);
        build_grd_rot_pyr(gpyr, grdPyr, rotPyr, nOctaves, nLayers);

        // Detect keypoints
        detect_keypoints(dogPyr, grdPyr, rotPyr, nOctaves, nDogLayers, kpt_list);

        // Extract descriptor
        if (bExtractDescriptors)
            extract_descriptor(grdPyr, rotPyr, nOctaves, nGpyrLayers, kpt_list);

        return 0;
    }

} // end namespace ezsift


#define USE_FIX_FILENAME 0
int main(int argc, char *argv[])
{

#if USE_FIX_FILENAME
    char *file1 = "img1.pgm";
#else
    if (argc != 2) {
        std::cerr
            << "Please input an input image name.\nUsage: feature_extract img"
            << std::endl;
        return -1;
    }
    char file1[255];
    memcpy(file1, argv[1], sizeof(char) * strlen(argv[1]));
    file1[strlen(argv[1])] = 0;
#endif

    ezsift::Image<unsigned char> image;
    if (ezsift::read_pgm(file1, image.data, image.w, image.h) != 0) {
        std::cerr << "Failed to open input image." << std::endl;
        return -1;
    }
    std::cout << "Image size: " << image.w << "x" << image.h << std::endl;

    bool bExtractDescriptor = true;
    std::list<ezsift::SiftKeypoint> kpt_list;

    // Double the original image as the first octive.
    ezsift::double_original_image(true);

    // Perform SIFT computation on CPU.
    std::cout << "Start SIFT detection ..." << std::endl;
    ezsift::sift_cpu(image, kpt_list, bExtractDescriptor);

    // Generate output image with keypoints drawing
    char filename[255];
    sprintf(filename, "%s_sift_output.ppm", file1);
    ezsift::draw_keypoints_to_ppm_file(filename, image, kpt_list);

    // Generate keypoints list
    sprintf(filename, "%s_sift_key.key", file1);
    ezsift::export_kpt_list_to_file(filename, kpt_list, bExtractDescriptor);

    std::cout << "\nTotal keypoints number: \t\t"
              << static_cast<unsigned int>(kpt_list.size()) << std::endl;

    return 0;
}
