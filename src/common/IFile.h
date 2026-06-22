/*********************                                                        */
/*! \file IFile.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __IFile_h__
#define __IFile_h__

class HeapData;
class String;

class IFile
{
public:
    enum Mode {
        MODE_READ,
        MODE_WRITE_APPEND,
        MODE_WRITE_TRUNCATE,
    };

    virtual void open( Mode openMode ) = 0;
    virtual void write( const String &line ) = 0;
    virtual String readLine( char lineSeparatingChar = '\n' ) = 0;
    virtual void read( HeapData &buffer, unsigned maxReadSize ) = 0;
    virtual void close() = 0;

    static bool exists( const String &path );

    virtual ~IFile()
    {
    }
};

#endif // __IFile_h__
