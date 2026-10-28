/*********************                                                        */
/*! \file ConvolutionMode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __CONVOLUTION_MODE_H__
#define __CONVOLUTION_MODE_H__

namespace NLR
{

// Mirrors auto_LiRPA's conv_mode
enum class ConvMode
{
    MATRIX,
    PATCHES
};

} // namespace NLR

#endif // __CONVOLUTION_MODE_H__
