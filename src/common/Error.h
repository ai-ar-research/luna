/*********************                                                        */
/*! \file Error.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __Error_h__
#define __Error_h__

#include <exception>

class Error : public std::exception
{
public:
    Error( const char *errorClass, int code );
    Error( const char *errorClass, int code, const char *userMessage );
    int getErrno() const;
    int getCode() const;
    void setUserMessage( const char *userMessage );
    const char *getErrorClass() const;
    const char *getUserMessage() const;
    const char *what() const noexcept override;

private:
    enum {
        BUFFER_SIZE = 2048,
    };

    char _errorClass[BUFFER_SIZE];
    int _code;
    int _errno;
    char _userMessage[BUFFER_SIZE];
};

#endif // __Error_h__
