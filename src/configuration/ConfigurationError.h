/*********************                                                        */
/*! \file ConfigurationError.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __ConfigurationError_h__
#define __ConfigurationError_h__

#include "Error.h"

class ConfigurationError : public Error
{
public:
    enum Code {
        OPTION_KEY_DOESNT_EXIST = 0,
        INCOMPTATIBLE_OPTIONS = 1,
    };

    ConfigurationError( ConfigurationError::Code code )
        : Error( "ConfigurationError", (int)code )
    {
    }

    ConfigurationError( ConfigurationError::Code code, const char *userMessage )
        : Error( "ConfigurationError", (int)code, userMessage )
    {
    }
};

#endif // __ConfigurationError_h__
