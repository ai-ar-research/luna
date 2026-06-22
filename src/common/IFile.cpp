/*********************                                                        */
/*! \file IFile.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "IFile.h"

#include "MString.h"
#include "T/sys/stat.h"
#include "T/unistd.h"

bool IFile::exists( const String &path )
{
    struct stat DONT_CARE;
    return T::stat( path.ascii(), &DONT_CARE ) == 0;
}
